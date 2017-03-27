from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class SplitAndPool(luigi.Task):
	"""
	Luigi Task to split the results of level 1 analysis by major cell class, and pool each class separately

	If tissue is "All", all tissues will be pooled.
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		tissues = cg.PoolSpec().tissues_for_project(self.project)
		if self.tissue == "All":
			return [cg.PrepareTissuePool(tissue=tissue) for tissue in tissues]
		else:
			return [cg.PrepareTissuePool(tissue=self.tissue)]

	def output(self) -> luigi.Target:
		if self.project == "Development":
			return luigi.LocalTarget(os.path.join("loom_builds", "Development_All.loom"))
		else:
			return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for clustered in self.input():
				ds = loompy.connect(clustered.fn)
				labels = ds.col_attrs["Class"]
				for (ix, selection, vals) in ds.batch_scan(axis=1):
					if self.project == "Adolescent":
						subset = np.intersect1d(np.where(labels == self.major_class)[0], selection)
						if subset.shape[0] == 0:
							continue
						m = vals[:, subset - ix]
					else:
						subset = selection
						m = vals
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][subset]
					if dsout is None:
						dsout = loompy.create(out_file, m, ds.row_attrs, ca)
					else:
						dsout.add_columns(m, ca)