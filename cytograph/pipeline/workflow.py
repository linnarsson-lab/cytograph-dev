import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

import cytograph.plotting as cgplot
import loompy
from cytograph.clustering import ClusterValidator
from cytograph.preprocessing import Scrublet, doublet_finder
from cytograph.species import Species

from .aggregator import Aggregator
from .config import config
from .cytograph import Cytograph
from .punchcards import Punchcard, PunchcardDeck, PunchcardSubset
from .utils import Tempname

#
# Overview of the cytograph2 pipeline
#
# sample1 ----\                /--> First_First.loom --------------------------------------|
# sample2 -----> First.loom --<                                                            |
# sample3 ----/                \--> First_Second.loom -------------------------------------|
#                                                                                          |
#                                                                                           >---> Pool.loom
#                                                                                          |
# sample4 ----\                 /--> Second_First.loom ------------------------------------|
# sample5 -----> Second.loom --<                                                           |
# sample6 ----/                 \                          /--> Second_Second_First.loom --|
#                               \--> Second_Second.loom --<                                |
#                                                          \--> Second_Second_Second.loom -|
#


class Metadata:
	"""
	Parse a semicolon-delimited metadata table
	"""
	def __init__(self, path: str) -> None:
		self.sid = "SampleID"
		self.metadata = None
		if path is not None and os.path.exists(path):
			# Special handling of typed column names for our database
			with open(path) as f:
				line = f.readline()
				if "SampleID:string" in line:
					self.sid = "SampleID:string"
			try:
				self.metadata = pd.read_csv(path, delimiter=";", index_col=self.sid, engine="python")
			except Exception as e:
				logging.error(f"Failed to load metadata because: {e}")

	def get(self, sample: str) -> Dict:
		"""
		Return metadata for the given sample as a dictionary
		"""
		if self.metadata is not None:
			if self.sid == "SampleID:string":
				return {k.split(":")[0]: v for k, v in self.metadata.loc[sample].items()}
			else:
				return {k: v for k, v in self.metadata.loc[sample].items()}
		else:
			return {}


class Workflow:
	"""
	Shared workflow for every task, implementing cytograph, aggregation and plotting

	Subclasses implement workflows that vary by the way cells are collected
	"""
	def __init__(self, deck: PunchcardDeck, name: str) -> None:
		self.deck = deck
		self.name = name
		self.loom_file = os.path.join(config.paths.build, "data", name + ".loom")
		self.agg_file = os.path.join(config.paths.build, "data", name + ".agg.loom")
		self.export_dir = os.path.join(config.paths.build, "exported", name)

	def collect_cells(self, out_file: str) -> loompy.LoomConnection:
		# Override this in subclasses
		pass

	def compute_subsets(self, card: Punchcard) -> None:
		logging.info(f"Computing subset assignments for {card.name}")
		with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".loom"), mode="r+") as ds:
			subset_per_cell = np.zeros(ds.shape[1], dtype=object)
			taken = np.zeros(ds.shape[1], dtype=bool)
			with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".agg.loom"), mode="r") as dsagg:
				for subset in card.subsets.values():
					selected = np.zeros(ds.shape[1], dtype=bool)
					if len(subset.include) > 0:
						# Include clusters that have any of the given auto-annotations
						for aa in subset.include:
							for ix in range(dsagg.shape[1]):
								if aa in dsagg.ca.AutoAnnotation[ix].split(" "):
									selected = selected | (ds.ca.Clusters == ix)
					else:
						selected = ~taken
					# Exclude cells that don't match the onlyif expression
					if subset.onlyif != "" and subset.onlyif is not None:
						selected = selected & eval(subset.onlyif, globals(), ds.ca)
					# Don't include cells that were already taken
					selected = selected & ~taken
					subset_per_cell[selected] = subset.name
			ds.ca.Subset = subset_per_cell

	def process(self) -> None:
		# STEP 1: build the .loom file and perform manifold learning (Cytograph)
		# Maybe we're already done?
		if os.path.exists(self.loom_file):
			logging.info(f"Skipping '{self.name}.loom' because it was already complete.")
		else:
			with Tempname(self.loom_file) as out_file:
				self.collect_cells(out_file)
				with loompy.connect(out_file) as ds:
					logging.info(f"Collected {ds.shape[1]} cells")
					Cytograph(steps=config.steps).fit(ds)

		# STEP 2: aggregate and create the .agg.loom file
		if os.path.exists(self.agg_file):
			logging.info(f"Skipping '{self.name}.agg.loom' because it was already complete.")
		else:
			with loompy.connect(self.loom_file) as dsout:
				with Tempname(self.agg_file) as out_file:
					Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, out_file=out_file)

		# STEP 3: export plots
		if os.path.exists(self.export_dir):
			logging.info(f"Skipping 'exported/{self.name}' because it was already complete.")
		else:
			pool = self.name
			logging.info(f"Exporting plots for {pool}")
			with Tempname(self.export_dir) as out_dir:
				os.mkdir(out_dir)
				with loompy.connect(self.loom_file) as ds:
					with loompy.connect(self.agg_file) as dsagg:
						cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.aa.png"), list(dsagg.ca.AutoAnnotation))
						cgplot.manifold(ds, os.path.join(out_dir, pool + "_TSNE_manifold.aaa.png"), list(dsagg.ca.MarkerGenes))
						cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.aaa.png"), list(dsagg.ca.MarkerGenes), embedding="UMAP")
						cgplot.markerheatmap(ds, dsagg, n_markers_per_cluster=10, out_file=os.path.join(out_dir, pool + "_heatmap.pdf"))
						cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
						cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
						cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
						cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
						cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
						cgplot.embedded_velocity(ds, out_file=os.path.join(out_dir, f"{pool}_velocity.png"))
						cgplot.TFs(ds, dsagg, out_file_root=os.path.join(out_dir, pool))
						if "cluster-validation" in config.steps:
							ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))

		# If there's a punchcard for this subset, go ahead and compute the subsets for that card
		card_for_subset = self.deck.get_card(self.name)
		if card_for_subset is not None:
			self.compute_subsets(card_for_subset)

		logging.info("Done.")


class RootWorkflow(Workflow):
	"""
	A workflow for the root, which collects its cells directly from input samples
	"""
	def __init__(self, deck: PunchcardDeck, subset: PunchcardSubset) -> None:
		super().__init__(deck, subset.longname())
		self.subset = subset
		self.deck = deck

	def collect_cells(self, out_file: str) -> None:
		# Make sure all the sample files exist
		err = False
		for batch in self.subset.include:
			for sample_id in batch:
				full_path = os.path.join(config.paths.samples, sample_id + ".loom")
				if not os.path.exists(full_path):
					logging.error(f"Sample file '{full_path}' not found")
					err = True
		if err and not config.params.skip_missing_samples:
			sys.exit(1)

		metadata = Metadata(config.paths.metadata)
		logging.info(f"Collecting cells for {self.name}")
		logging.debug(out_file)
		batch_id = 0
		n_cells = 0
		files: List[str] = []
		selections: List[str] = []
		new_col_attrs: List[Dict[str, np.ndarray]] = []
		for batch in self.subset.include:
			replicate_id = 0
			for sample_id in batch:
				full_path = os.path.join(config.paths.samples, sample_id + ".loom")
				if not os.path.exists(full_path):
					continue
				logging.info(f"Adding {sample_id}.loom")
				with loompy.connect(full_path, "r") as ds:
					species = Species.detect(ds).name
					col_attrs = dict(ds.ca)
					for key, val in metadata.get(sample_id).items():
						col_attrs[key] = np.array([val] * ds.shape[1])
					if "Species" not in col_attrs:
						col_attrs["Species"] = np.array([species] * ds.shape[1])
					col_attrs["SampleID"] = np.array([sample_id] * ds.shape[1])
					col_attrs["Batch"] = np.array([batch_id] * ds.shape[1])
					col_attrs["Replicate"] = np.array([replicate_id] * ds.shape[1])
					logging.info("Scoring doublets using Scrublet")
					if "doublets" in config.steps:
						if config.params.doublets_method == "scrublet":
							data = ds[:, :].T
							doublet_scores, predicted_doublets = Scrublet(data, expected_doublet_rate=0.05).scrub_doublets()
							col_attrs["ScrubletScore"] = doublet_scores
							if predicted_doublets is None:  # Sometimes scrublet gives up and returns None
								predicted_doublets = np.zeros(ds.shape[1], dtype=bool)
								col_attrs["ScrubletFlag"] = np.zeros(ds.shape[1])
							else:
								col_attrs["ScrubletFlag"] = predicted_doublets.astype("int")
						elif config.params.doublets_method == "doublet-finder":
							logging.info("Scoring doublets using DoubletFinder")
							col_attrs["DoubletFinderScore"] = doublet_finder(ds)
					logging.info(f"Computing total UMIs")
					(totals, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
					col_attrs["TotalUMI"] = totals
					col_attrs["NGenes"] = genes
					good_cells = (totals >= config.params.min_umis)
					if config.params.doublets_method == "scrublet" and config.params.doublets_action == "remove":
						logging.info(f"Removing {predicted_doublets.sum()} doublets and {(~good_cells).sum()} cells with <{config.params.min_umis} UMIs")
						good_cells = good_cells & (~predicted_doublets)
					else:
						logging.info(f"Removing {(~good_cells).sum()} cells with <{config.params.min_umis} UMIs")
					if good_cells.sum() / ds.shape[1] > config.params.min_fraction_good_cells:
						logging.info(f"Including {good_cells.sum()} of {ds.shape[1]} cells")
						n_cells += good_cells.sum()
						files.append(full_path)
						selections.append(good_cells)
						new_col_attrs.append(col_attrs)
					else:
						logging.warn(f"Skipping {sample_id}.loom because only {good_cells.sum()} of {ds.shape[1]} cells (less than {config.params.min_fraction_good_cells * 100}%) passed QC.")
				replicate_id += 1
			batch_id += 1
		logging.info(f"Collecting a total of {n_cells} cells.")
		loompy.combine_faster(files, out_file, {}, selections, skip_attrs=["_X", "_Y", "Clusters"])
		logging.info(f"Adding column attributes")
		with loompy.connect(out_file) as ds:
			for attr in new_col_attrs[0].keys():
				if all([attr in attrs for attrs in new_col_attrs]):
					ds.ca[attr] = np.concatenate([x[attr][sel] for x, sel in zip(new_col_attrs, selections)])
				else:
					logging.warn(f"Skipping column attribute {attr} because it was only present in some of the inputs")


class SubsetWorkflow(Workflow):
	"""
	A workflow for intermediate punchcards, which collects its cells from a previous punchcard subset
	"""
	def __init__(self, deck: PunchcardDeck, subset: PunchcardSubset) -> None:
		super().__init__(deck, subset.longname())
		self.subset = subset
		self.deck = deck

	def collect_cells(self, out_file: str) -> None:
		# Verify that the previous punchard subset exists
		parent = os.path.join(config.paths.build, "data", self.subset.card.name + ".loom")
		if not os.path.exists(parent):
			logging.error(f"Punchcard file '{parent}' was missing.")
			sys.exit(1)

		# Verify that there are some cells in the subset
		with loompy.connect(parent, mode="r") as ds:
			if (ds.ca.Subset == self.subset.name).sum() == 0:
				logging.info(f"Skipping {self.name} because the subset was empty")
				sys.exit(0)

		logging.info(f"Collecting cells for {self.name}")
		with loompy.new(out_file) as dsout:
			# Collect from a previous punchard subset
			with loompy.connect(parent, mode="r") as ds:
				for (ix, selection, view) in ds.scan(items=(ds.ca.Subset == self.subset.name), axis=1, key="Accession"):
					dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)


class PoolWorkflow(Workflow):
	"""
	Workflow for the final pooling step, which collects its cells from all the leaf subsets
	"""
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck, "Pool")
		self.deck = deck

	def collect_cells(self, out_file: str) -> None:
		logging.info(f"Collecting cells for 'Pool.loom'")
		punchcards: List[str] = []
		clusters: List[int] = []
		punchcard_clusters: List[int] = []
		next_cluster = 0

		# Check that all the inputs exist
		err = False
		for subset in self.deck.get_leaves():
			if not os.path.exists(os.path.join(config.paths.build, "data", subset.longname() + ".loom")):
				logging.error(f"Punchcard file 'data/{subset.longname()}.loom' is missing")
				err = True
		if err:
			sys.exit(1)

		with loompy.new(out_file) as dsout:
			for subset in self.deck.get_leaves():
				with loompy.connect(os.path.join(config.paths.build, "data", subset.longname() + ".loom"), mode="r") as ds:
					punchcards = punchcards + [subset.longname()] * ds.shape[1]
					punchcard_clusters = punchcard_clusters + list(ds.ca.Clusters)
					clusters = clusters + list(ds.ca.Clusters + next_cluster)
					next_cluster = max(clusters) + 1
					for (ix, selection, view) in ds.scan(axis=1, key="Accession"):
						dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
			ds.ca.Punchcard = punchcards
			ds.ca.PunchcardClusters = punchcard_clusters
			ds.ca.Clusters = clusters
