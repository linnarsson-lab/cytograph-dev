from typing import *
import os
import logging
from math import exp, lgamma, log
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg


class Trinarizer:
	def __init__(self, f: float = 0.2) -> None:
		self.f = f
		self.trinary_prob = None  # type: np.ndarray
		self.genes = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
		cells = np.where(ds.col_attrs["Clusters"] >= 0)[0]
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = np.max(labels) + 1
		logging.info("n_labels %d", n_labels)
		self.trinary_prob = np.empty((ds.shape[0], n_labels))
		self.genes = ds.Gene

		j = 0
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
			for j, row in enumerate(selection):
				data = np.round(vals[j, :])
				self.trinary_prob[row, :] = self._betabinomial_trinarize_array(data, labels, self.f, n_labels)

		return self.trinary_prob

	def save(self, fname: str) -> None:
		# Save the result
		with open(fname, "w") as f:
			f.write("Gene\t")
			for ix in range(self.trinary_prob.shape[1]):
				f.write(str(ix) + "\t")
			f.write("\n")

			for row in range(self.trinary_prob.shape[0]):
				f.write(self.genes[row] + "\t")
				for ix in range(self.trinary_prob.shape[1]):
					f.write(str(self.trinary_prob[row, ix]) + "\t")
				f.write("\n")

	def _betabinomial_trinarize_array(self, array: np.ndarray, labels: np.ndarray, f: float, n_labels: int = None) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Trinarize a vector, grouped by labels, using a beta binomial model

		Args:
			array (ndarray of ints):	The input vector of ints
			labels (ndarray of ints):	Group labels 0, 1, 2, ....

		Returns:
			ps (ndarray of float):		The posterior probability of expression in at least a fraction f

		Remarks:
			We calculate probability p that at least half the cells express (in each group),
			and compare with pep, setting the binary pattern to 1 if p > pep,
			-1 if p < (1 - pep) and 0 otherwise.
		"""
		def p_half(k: int, n: int, f: float) -> float:
			"""
			Return probability that at least half the cells express, if we have observed k of n cells expressing

			Args:
				k (int):	Number of observed positive cells
				n (int):	Total number of cells

			Remarks:
				Probability that at least a fraction f of the cells express, when we observe k positives among n cells is:

					p|k,n = 1-(betainc(1+k, 1-k+n, f)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

			Note:
				The formula was derived in Mathematica by computing

					Probability[x > f, {x \[Distributed] BetaDistribution[1 + k, 1 + n - k]}]
			"""

			# These are the prior hyperparameters beta(a,b)
			a = 1.5
			b = 2

			# We really want to calculate this:
			# p = 1-(betainc(a+k, b-k+n, 0.5)*beta(a+k, b-k+n)*gamma(a+b+n)/(gamma(a+k)*gamma(b-k+n)))
			#
			# But it's numerically unstable, so we need to work on log scale (and special-case the incomplete beta)

			incb = betainc(a + k, b - k + n, f)
			if incb == 0:
				p = 1.0
			else:
				p = 1.0 - exp(log(incb) + betaln(a + k, b - k + n) + lgamma(a + b + n) - lgamma(a + k) - lgamma(b - k + n))
			return p

		if n_labels is None:
			n_labels = np.max(labels) + 1
		n_by_label = np.bincount(labels, minlength=n_labels)
		k_by_label = np.zeros(n_labels)
		for lbl in range(n_labels):
			if np.sum(labels == lbl) == 0:
				continue
			k_by_label[lbl] = np.count_nonzero(array[np.where(labels == lbl)[0]])

		vfunc = np.vectorize(p_half)
		ps = vfunc(k_by_label, n_by_label, f)

		return ps
