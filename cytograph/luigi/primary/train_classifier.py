from typing import *
import os
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class TrainClassifier(luigi.Task):
	"""
	Luigi Task to train a classifier
	"""

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "classifier.pickle"))

	def run(self) -> None:
		with self.output().temporary_path() as fname:
			logging.info("Retraining classifier")
			pathname = os.path.join(cg.paths().build, "classified")
			clf = cg.Classifier(os.path.join(cg.paths().build, "classified"), n_per_cluster=100)
			clf.generate()
			ds_training = loompy.connect(os.path.join(pathname, "classified.loom"))
			clf.fit(ds_training)
			with open(fname, "wb") as f:
				pickle.dump(clf, f)
