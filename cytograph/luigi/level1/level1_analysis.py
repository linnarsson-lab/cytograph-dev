from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class Level1Analysis(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 1 analyses
	"""

	project = luigi.Parameter(default="Adolescent")

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project(self.project)
		for tissue in tissues:
			yield cg.PlotCVMeanL1(tissue=tissue)
			yield cg.PlotGraphL1(tissue=tissue)
			yield cg.PlotClassesL1(tissue=tissue)
			yield cg.MarkerEnrichmentL1(tissue=tissue)