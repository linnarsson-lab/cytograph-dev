from typing import *
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from scipy.special import gammaln, digamma, psi
from scipy.misc import logsumexp
from tqdm import trange
import ctypes


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
	Bayesian Hierarchical Poisson Factorization
	Implementation of https://arxiv.org/pdf/1311.1704.pdf
	"""
	def __init__(self, k: int, a: float = 1, b: float = 5, c: float = 1, d: float = 5, max_iter: int = 1000, stop_interval: int = 10) -> None:
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.max_iter = max_iter
		self.stop_interval = stop_interval

		self.beta: np.ndarray = None
		self.theta: np.ndarray = None

		self.log_likelihoods: List[float] = []

		self._tau_rate: np.ndarray = None
		self._tau_shape: np.ndarray = None
		self._lambda_rate: np.ndarray = None
		self._lambda_shape: np.ndarray = None

	def fit(self, X: sparse.coo_matrix) -> Any:
		"""
		Fit an HPF model to the data matrix

		Args:
			X	Data matrix, shape (n_cells, n_genes)

		Remarks:
			After fitting, the factor matrices beta and theta are available as self.theta of shape
			(n_cells, k) and self.beta of shape (k, n_genes)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		(beta, theta) = self._fit(X)

		self.beta = beta
		self.theta = theta
		return self

	def _fit(self, X: sparse.coo_matrix, beta_precomputed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		# Create local variables for convenience
		(n_users, n_items) = X.shape
		(a, b, c, d) = (self.a, self.b, self.c, self.d)
		k = self.k
		# u and i are indices of the nonzero entries; y are the values of those entries
		(u, i, y) = (X.row, X.col, X.data)

		# Initialize the variational parameters with priors
		kappa_shape = np.full(n_users, a) + np.random.uniform(0, 0.1, n_users)
		kappa_rate = np.full(n_users, b + k)
		gamma_shape = np.full((n_users, k), a) + np.random.uniform(0, 0.1, (n_users, k))
		gamma_rate = np.full((n_users, k), b) + np.random.uniform(0, 0.1, (n_users, k))

		if beta_precomputed:
			tau_shape = self._tau_shape
			tau_rate = self._tau_rate
			lambda_shape = self._lambda_shape
			lambda_rate = self._lambda_rate
		else:
			tau_shape = np.full(n_items, c) + np.random.uniform(0, 0.1, n_items)
			tau_rate = np.full(n_items, d + k)
			lambda_shape = np.full((n_items, k), c) + np.random.uniform(0, 0.1, (n_items, k))
			lambda_rate = np.full((n_items, k), d) + np.random.uniform(0, 0.1, (n_items, k))

		self.log_likelihoods = []
		with trange(self.max_iter) as t:
			t.set_description(f"HPF.fit(X.shape={X.shape})")
			for n_iter in t:
				make_nonzero(gamma_shape)
				make_nonzero(gamma_rate)
				make_nonzero(lambda_shape)
				make_nonzero(lambda_rate)

				# Compute y * phi only for the nonzero values, which are indexed by u and i in the sparse matrix
				# phi is calculated on log scale from expectations of the gammas, hence the digamma and log terms
				# Shape of phi will be (nnz, k)
				phi = (digamma(gamma_shape) - np.log(gamma_rate))[u, :] + (digamma(lambda_shape) - np.log(lambda_rate))[i, :]
				# Multiply y by phi normalized (in log space) along the k axis
				y_phi = y[:, None] * np.exp(phi - logsumexp(phi, axis=1)[:, None])
				
				# Upate the variational parameters corresponding to theta (the users)
				# Sum of y_phi over users, for each k
				y_phi_sum_u = np.zeros((n_users, k))
				for ix in range(k):
					y_phi_sum_u[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=1).A.T[0]
				gamma_shape = a + y_phi_sum_u
				gamma_rate = (kappa_shape / kappa_rate)[:, None] + (lambda_shape / lambda_rate).sum(axis=0)
				kappa_rate = b + (gamma_shape / gamma_rate).sum(axis=1)

				if not beta_precomputed:
					# Upate the variational parameters corresponding to beta (the items)
					# Sum of y_phi over items, for each k
					y_phi_sum_i = np.zeros((n_items, k))
					for ix in range(k):
						y_phi_sum_i[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=0).A
					lambda_shape = c + y_phi_sum_i
					lambda_rate = (tau_shape / tau_rate)[:, None] + (gamma_shape / gamma_rate).sum(axis=0)
					tau_rate = d + (lambda_shape / lambda_rate).sum(axis=1)

				if (n_iter + 1) % self.stop_interval == 0:
					# Compute the log likelihood and assess convergence
					# Expectations
					egamma = make_nonzero(gamma_shape / gamma_rate)
					elambda = make_nonzero(lambda_shape / lambda_rate)
					# Sum over k for the expectations
					# This is really a dot product but we're only computing it for the nonzeros (indexed by u and i)
					s = (egamma[u] * elambda[i]).sum(axis=1)
					# We use gammaln to compute the log factorial, hence the "y + 1"
					log_likelihood = np.sum(y * np.log(s) - s - gammaln(y + 1))
					self.log_likelihoods.append(log_likelihood)

					# Check for convergence
					# TODO: allow for small fluctuations?
					if len(self.log_likelihoods) > 1:
						prev_ll = self.log_likelihoods[-2]
						diff = abs((log_likelihood - prev_ll) / prev_ll)
						t.set_postfix(ll=log_likelihood, diff=diff)
						if diff < 0.00001:
							break
					else:
						t.set_postfix(ll=log_likelihood)

		# End of the main fitting loop
		if not beta_precomputed:
			# Save these for future use in self.transform()
			self._tau_shape = tau_shape
			self._tau_rate = tau_rate
			self._lambda_shape = lambda_shape
			self._lambda_rate = lambda_rate

		# Compute beta and theta, which are given by the expectations, i.e. shape / rate
		beta = lambda_shape / lambda_rate
		theta = gamma_shape / gamma_rate
		return (beta, theta)


	def transform(self, X: sparse.coo_matrix) -> np.ndarray:
		"""
		Transform the data matrix using an already fitted HPF model

		Args:
			X      Data matrix, shape (n_cells, n_genes)

		Returns:
			Factor matrix theta of shape (n_cells, k)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		(beta, theta) = self._fit(X, beta_precomputed=True)

		return theta
