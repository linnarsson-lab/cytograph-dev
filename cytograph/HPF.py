from typing import *
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from scipy.special import gammaln, digamma


def make_nonzero(a: np.ndarray) -> np.ndarray:
	"""
	Make the array nonzero in place, by replacing zeros with 1e-30

	Returns:
		a	The input array
	"""
	a[a == 0.0] = 1e-30
	return a


class HPF:
	"""
	Bayesian scalable hierarchical Poisson matrix factorization
	See https://arxiv.org/pdf/1311.1704.pdf

	This implementation requires https://github.com/linnarsson-lab/hgaprec to be installed and in the $PATH
	"""
	def __init__(self, k: int, a: float = 0.3, b: float = 0.3, c: float = 0.3, d: float = 0.3) -> None:
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.beta: np.ndarray = None
		self.theta: np.ndarray = None

		self._cache: Dict[str, str] = {}

	def fit_python(self, X: sparse.coo_matrix) -> None:
		"""
		Fit an HPF model to the data matrix

		Args:
			X      			Data matrix, shape (n_users, n_items)

		Remarks:
			TODO check this
			After fitting, the factor matrices beta and theta are available as self.beta of shape
			(n_users, k) and self.theta of shape (k, n_items)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")
		
		(n_users, n_items) = X.shape
		(a, b, c, d) = (self.a, self.b, self.c, self.d)
		k = self.k
		(u, i, y) = (X.row, X.col, X.data)

		kappa_shape = np.zeros(n_users) + a + k * b
		kappa_rate = np.zeros(n_users)
		gamma_shape = np.zeros((n_users, k))  # TODO: add prior
		gamma_rate = np.zeros((n_users, k))   # TODO: add prior

		tau_shape = np.zeros(n_items) + c + k * d
		tau_rate = np.zeros(n_items)
		lambda_shape = np.zeros((n_items, k))  # TODO: add prior
		lambda_rate = np.zeros((n_items, k))   # TODO: add prior

		while True:
			make_nonzero(gamma_shape)
			make_nonzero(gamma_rate)
			make_nonzero(lambda_shape)
			make_nonzero(lambda_rate)
			
			# Compute y * phi only for the nonzero values, which are indexed by u and i in the sparse matrix
			# phi is calculated on log scale from expectations of the gammas, then exponentiated, hence the digamma and log terms
			# Shape of phi will be (nnz, k)
			y_phi = y * np.exp(digamma(gamma_shape[u, :]) - np.log(gamma_rate[u, :]) + digamma(lambda_shape[i, :]) - np.log(lambda_rate[i, :]))

			# Upate the variational parameters corresponding to theta (the users)
			y_phi_sum_u = sparse.coo_matrix((u, i, y_phi[0]), X.shape).sum(axis=0)
			for ix in range(1, k):
				y_phi_sum_u += sparse.coo_matrix((u, i, y_phi[ix]), X.shape).sum(axis=0)
			gamma_shape = a + y_phi_sum_u
			gamma_rate = (kappa_shape / kappa_rate)[:, None] + (lambda_shape / lambda_rate).sum(axis=0).T
			kappa_rate = b + (gamma_shape / gamma_rate).sum(axis=1)

			# Upate the variational parameters corresponding to beta (the items)
			y_phi_sum_i = sparse.coo_matrix((u, i, y_phi[0]), X.shape).sum(axis=1)
			for ix in range(1, k):
				y_phi_sum_i += sparse.coo_matrix((u, i, y_phi[ix]), X.shape).sum(axis=1)
			lambda_shape = c + y_phi_sum_i
			lambda_rate = (tau_shape / tau_rate)[:, None] + (gamma_shape / gamma_rate).sum(axis=0).T
			tau_rate = d + (lambda_shape / lambda_rate).sum(axis=1)

			# Compute the log likelihood and assess convergence
			egamma = make_nonzero(gamma_shape / gamma_rate)
			elambda = make_nonzero(lambda_shape / lambda_rate)
			# Sum over k for the expectations
			# This is really a dot product but we're only computing it for the nonzeros (indexed by u and i)
			s = (egamma[u] * elambda[i]).sum(axis=1)
			log_likelihood = np.sum(y * np.log(s) - s - gammaln(y + 1))

	def fit(self, X: sparse.coo_matrix) -> None:
		"""
		Fit an HPF model to the data matrix

		Args:
			X      			Data matrix, shape (n_samples, n_features)
			test_size       Fraction to use for test dataset, or None to use X for training, test and validation
			validation_size Fraction to use for validation dataset

		Remarks:
			After fitting, the factor matrices beta and theta are available as self.beta of shape
			(n_samples, k) and self.theta of shape (k, n_features)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)
			# Save to TSV file
			np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")

			# Run hgaprec
			bnpf_p = Popen((
				"hgaprec",
				"-dir", tmpdirname,
				"-m", str(X.shape[1]),
				"-n", str(X.shape[0]),
				"-k", str(self.k),
				"-a", str(self.a),
				"-b", str(self.b),
				"-c", str(self.c),
				"-d", str(self.d),
				"-hier"
			), cwd=tmpdirname)
			bnpf_p.wait()
			if bnpf_p.returncode != 0:
				logging.error("HPF failed to execute external binary 'hgaprec' (check $PATH)")
				raise RuntimeError()
			sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb"

			# Format of these is (row, col, )
			self.theta = np.loadtxt(os.path.join(tmpdirname, sf, "htheta.tsv"))[:, 2:]
			temp = np.loadtxt(os.path.join(tmpdirname, sf, "hbeta.tsv"))
			self.beta = temp[:, 2:][np.argsort(temp[:, 1]), :]  # the beta matrix with the correct rows ordering

			# Cache the beta branch of the model, so we can refit later with fixed beta
			for file in ["betarate.tsv", "betarate_rate.tsv", "betarate_shape.tsv", "hbeta.tsv", "hbeta_rate.tsv", "hbeta_shape.tsv"]:
				with open(os.path.join(tmpdirname, sf, file)) as f:
					self._cache[file] = f.read()

	def transform(self, X: sparse.coo_matrix) -> np.ndarray:
		if self.beta is None:
			raise NotFittedError("Cannot transform without first fitting the model")
		if X.shape[1] != self.beta.shape[0]:
			raise ValueError(f"X must have exactly {self.beta.shape[0]} columns")
		if np.any(np.sum(X, axis=0) == 0):
			raise ValueError("Every feature (column) must have at least one non-zero sample (row)")
		if np.any(np.sum(X, axis=1) == 0):
			raise ValueError("Every sample (row) must have at least one non-zero feature (column)")
		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)

			# Save to TSV file
			np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")

			# Write the previously saved beta samples for reuse
			# sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb-beta-precomputed"
			# if not os.path.exists(os.path.join(tmpdirname, sf)):
			# 	os.mkdir(os.path.join(tmpdirname, sf))
				
			for file in self._cache.keys():
				with open(os.path.join(tmpdirname, file), "w") as f:
					f.write(self._cache[file])

			# Run hgaprec
			bnpf_p = Popen((
				"hgaprec",
				"-dir", tmpdirname,
				"-m", str(X.shape[1]),
				"-n", str(X.shape[0]),
				"-k", str(self.k),
				"-a", str(self.a),
				"-b", str(self.b),
				"-c", str(self.c),
				"-d", str(self.d),
				"-beta-precomputed"
			), cwd=tmpdirname)
			bnpf_p.wait()
			if bnpf_p.returncode != 0:
				logging.error("HPF failed to execute external binary 'hgaprec' (check $PATH)")
				raise RuntimeError()
			sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb-beta-precomputed"
			return np.loadtxt(os.path.join(tmpdirname, sf, "htheta.tsv"))[:, 2:]
