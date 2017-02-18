from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class Classify(luigi.Task):
	"""
	Luigi Task to classify major cell types
	"""
	tissue = luigi.Parameter()

	def requires(self) -> List[luigi.Task]:
		return [cg.TrainClassifier(), cg.PrepareTissuePool(tissue=self.tissue)]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".classes.txt"))

	def run(self) -> None:
		with open(self.input()[0].fn, "rb") as f:
			clf = pickle.load(f)
		with self.output().temporary_path() as fname:
			logging.info("Classification of major cell types")
			logging.info("Note: as side-effect, the column attribute 'Class' will be set")
			ds = loompy.connect(self.input()[1].fn)
			(probs, labels) = clf.predict_proba(ds)
			labels = [x.replace("-", "_") for x in labels]
			ds.set_attr("Class", labels[np.argmax(probs, axis=1)], axis=1)
			for ix, label in enumerate(labels):
				ds.set_attr("Class_" + label, probs[:, ix], axis=1)
			np.savetxt(fname, probs)
