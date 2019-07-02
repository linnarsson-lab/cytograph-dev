import logging
import os
import shutil
import subprocess
import sys
from typing import List, Dict, Optional, Set, Union

from .config import ExecutionConfig, config, merge_config
from .punchcards import PunchcardDeck, PunchcardSubset, PunchcardView


def is_task_complete(task: str) -> bool:
	if not os.path.exists(os.path.join(config.paths.build, "data", task + ".loom")):
		return False
	if not os.path.exists(os.path.join(config.paths.build, "data", task + ".agg.loom")):
		return False
	if not os.path.exists(os.path.join(config.paths.build, "exported", task)):
		return False
	return True


class Engine:
	'''
	An execution engine, which takes a :class:`PunchcardDeck` and calculates an execution plan in the form
	of a dependency graph. The Engine itself does not actually execute the graph. This is the job of
	subclasses such as :class:`LocalEngine`` and :class:`CondorEngine`, which take the execution plan and
	executes it in some manner (e.g. locally and serially, or on a cluster and in parallel).
	'''

	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		self.deck = deck
		self.dryrun = dryrun
	
	def build_execution_dag(self) -> Dict[str, List[str]]:
		"""
		Build an execution plan in the form of a dependency graph, encoded as a dictionary.

		Returns:
			Dictionary mapping tasks to their dependencies
	
		Remarks:
			The tasks are named for the punchcard subset they involve (using long subset names),
			or the view name ("View_xxx"), and the pooling task is denoted by the special task name 'Pool'.
		"""
		stack = self.deck.root.get_leaves()
		if len(stack) > 1:
			tasks = {"Pool": [s.longname() for s in stack]}
		else:
			tasks = {}
		while len(stack) > 0:
			s = stack.pop()
			if s.longname() in tasks:
				continue
			dep = s.dependency()
			if dep is not None:
				dep_subset = self.deck.get_subset(dep)
				if dep_subset is None:
					logging.error(f"Dependency '{dep}' of '{s.longname()}' was not found in punchcard deck.")
					sys.exit(1)
				stack.append(dep_subset)
				tasks[s.longname()] = [dep]
			else:
				tasks[s.longname()] = []
		# Add views
		for view in self.deck.views:
			for i in view.include:
				if i not in tasks.keys():
					logging.error(f"Dependency '{i}' of view '{view.name}' was not found in punchcard deck.")
					sys.exit(1)
			tasks[view.name] = view.include
		return tasks
	
	def execute(self) -> None:
		pass


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
	result: List[str] = []
	seen: Set[str] = set()

	def recursive_helper(node: str) -> None:
		for neighbor in graph.get(node, []):
			if neighbor not in seen:
				seen.add(neighbor)
				recursive_helper(neighbor)
		if node not in result:
			result.append(node)

	for key in graph.keys():
		recursive_helper(key)
	return result



class LocalEngine(Engine):
	"""
	An execution engine that executes tasks serially and locally.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

	def execute(self) -> None:
		tasks = self.build_execution_dag()
		for task, deps in tasks.items():
			if len(deps) > 0:
				logging.debug(f"Task {task} depends on {','.join(deps)}")
			else:
				logging.debug(f"Task {task} has no dependencies")

		# Figure out a linear execution order consistent with the DAG
		ordered_tasks = topological_sort(tasks)
		logging.debug(f"Execution order: {','.join(ordered_tasks)}")

		# Figure out which tasks have already been completed
		filtered_tasks = [t for t in ordered_tasks if not is_task_complete(t)]

		# Now we have the tasks ordered by the DAG, and run them
		if self.dryrun:
			logging.info("Dry run only, with the following execution plan")
		for ix, task in enumerate(filtered_tasks):
			if task == "Pool":
				if not self.dryrun:
					logging.info(f"\033[1;32;40mBuild step {ix + 1} of {len(filtered_tasks)}: cytograph pool\033[0m")
					subprocess.run(["cytograph", "--hide-message", "pool"])
				else:
					logging.info("cytograph pool")
			else:
				if not self.dryrun:
					logging.info(f"\033[1;32;40mBuild step {ix + 1} of {len(filtered_tasks)}: cytograph process {task}\033[0m")
					subprocess.run(["cytograph", "--hide-message", "process", task])
				else:
					logging.info(f"cytograph process {task}")


class CondorEngine(Engine):
	"""
	An engine that executes tasks in parallel on a HTCondor cluster, using the DAGman functionality
	of condor. Tasks will be executed in parallel as much as possible while respecting the
	dependency graph.
	"""
	def __init__(self, deck: PunchcardDeck, dryrun: bool = True) -> None:
		super().__init__(deck, dryrun)

	def execute(self) -> None:
		tasks = self.build_execution_dag()
		# Make job files
		exdir = os.path.abspath(os.path.join(config.paths.build, "condor"))
		if os.path.exists(exdir):
			logging.warn("Removing previous build logs from 'condor' directory.")
			shutil.rmtree(exdir)
		os.mkdir(exdir)
		for task in tasks.keys():
			if is_task_complete(task):
				continue
			cmd = ""
			excfg: Optional[ExecutionConfig] = None
			# Get the right execution configuration for the task (CPUs etc.)
			if task == "Pool":
				cfg_file = os.path.join(config.paths.build, "pool_config.yaml")
				if os.path.exists(cfg_file):
					merge_config(config, cfg_file)
				excfg = config.execution
				cmd = "pool"
			else:
				subset: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = self.deck.get_subset(task)
				if subset is None:
					subset = self.deck.get_view(task)
					if subset is None:
						logging.error(f"Subset or view {task} not found.")
						sys.exit(1)
				config.execution.merge(subset.execution)
				excfg = config.execution
				cmd = f"process {task}"

			# Generate the condor submit file for the task
			cytograph_exe = shutil.which('cytograph')
			if cytograph_exe is None:
				logging.error("The 'cytograph' command-line tool was not found.")
				sys.exit(1)
			# Must set 'request_gpus' only if non-zero, because even asking for zero GPUs requires a node that has GPUs (weirdly)
			request_gpus = f"request_gpus = {excfg.n_gpus}" if excfg.n_gpus > 0 else ""
			with open(os.path.join(exdir, task + ".condor"), "w") as f:
				f.write(f"""
getenv       = true
executable   = {os.path.abspath(cytograph_exe)}
arguments    = "{cmd}"
log          = {os.path.join(exdir, task)}.log
output       = {os.path.join(exdir, task)}.out
error        = {os.path.join(exdir, task)}.error
request_cpus = {excfg.n_cpus}
{request_gpus}
request_memory = {excfg.memory * 1024}
queue 1\n
""")

		with open(os.path.join(exdir, "_dag.condor"), "w") as f:
			for task in tasks.keys():
				if is_task_complete(task):
					continue
				f.write(f"JOB {task} {os.path.join(exdir, task)}.condor DIR {config.paths.build}\n")
			for task, deps in tasks.items():
				filtered_deps = [d for d in deps if not is_task_complete(d)]
				if len(filtered_deps) == 0:
					continue
				f.write(f"PARENT {' '.join(filtered_deps)} CHILD {task}\n")

		if not self.dryrun:
			logging.info(f"condor_submit_dag {os.path.join(exdir, '_dag.condor')}")
			subprocess.run(["condor_submit_dag", os.path.join(exdir, "_dag.condor")])
		else:
			logging.info(f"(Dry run) condor_submit_dag {os.path.join(exdir, '_dag.condor')}")

# TODO: SlurmEngine using job dependencies (https://hpc.nih.gov/docs/job_dependencies.html)
# TODO: SgeEngine using job dependencies (https://arc.leeds.ac.uk/using-the-systems/why-have-a-scheduler/advanced-sge-job-dependencies/)
