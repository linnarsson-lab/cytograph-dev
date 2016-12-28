
import numpy as np
import copy
import logging
from typing import *
from numpy_groupies import aggregate_numba as agg


class Facet:
	def __init__(self, name: str, k: int=2, n_genes: int=100, genes: List[int]=[], adaptive: bool=False) -> None:
		"""
		Create a Facet object

		Args:
			k (int):				Number of (initial) clusters in this facet
			n (int): 				Number of genes to allocate to this Facet
			genes (List[string]):	Genes to use to initialize this Facet
			adaptive (bool):		If true, the number of clusters is increased until BIC is minimized
		"""
		self.k = k
		self.n_genes = n_genes
		self.name = name
		self.genes = genes
		self.adaptive = adaptive

		# fields used during fitting
		self.labels = None  # type: np.ndarray
		self.S = None  # type: np.ndarray
		self.pi_k = None  # type: np.ndarray
		self.y_g = None  # type: np.ndarray


class FacetLearning:
	def __init__(self, facets: List[Facet], r: float = 2.0, alpha: float = 1.0, max_iter: int = 100) -> None:
		"""
		Create a FacetLearning object

		Args:
			facets (List[Facet]):	The facet definitions
			r (float):				The overdispersion
			alpha (float):			The regularization factor
			max_iter (int):			The number of EM iterations to run
		"""
		self.facets = facets
		self.r = r
		self.alpha = alpha
		self.max_iter = max_iter

	def fit(self, X: np.ndarray) -> None:
		for facet in self.facets:
			facet.labels = np.random.randint(facet.k, size=X.shape[0])
			facet.S = np.random.choice(X.shape[1], size=facet.n_genes, replace=False)
			if len(facet.genes) > 0:
				facet.S[:len(facet.genes)] = facet.genes
			facet.pi_k = np.ones(facet.k) / facet.k

		for _ in range(self.max_iter):
			self._E_step(X)
			self._M_step(X)

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		self.fit(X)
		return self.transform()

	def transform(self) -> np.ndarray:
		labels = []
		for facet in self.facets:
			labels.append(facet.labels)
		return np.array(labels).T

	def _E_step(self, X: np.ndarray) -> None:
		for facet in self.facets:
			X_S = X[:, facet.S]
			n_cells = X.shape[0]
			# (n_cells, k)
			z_ck = np.zeros((n_cells, facet.k))
			# (k, n_S)
			mu_gk = agg.aggregate(facet.labels, X_S, func='mean', fill_value=0, size=facet.k, axis=0) + 0.01
			# (k, n_S)
			p_gk = mu_gk / (mu_gk + self.r)
			# (n_cells, k)
			# z_ck += X_S.dot((np.log(p_gk) + self.r*np.log(1-p_gk)).transpose())
			z_ck += np.log(facet.pi_k)
			z_ck += np.log(p_gk).dot(X_S.transpose()).transpose()
			z_ck += np.sum(self.r * np.log(1 - p_gk), axis=1)
			# (n_cells)
			facet.labels = np.argmax(z_ck, axis=1)
			# Add 1 to each as a pseudocount to avoid zeros
			facet.pi_k = (np.bincount(facet.labels, minlength=facet.k) + 1) / (n_cells + facet.k)

	def _M_step(self, X: np.ndarray) -> None:
		n_genes = X.shape[1]
		n_cells = X.shape[0]

		all_yg = np.zeros((n_genes, len(self.facets)))
		for i, facet in enumerate(self.facets):
			# (n_genes)
			facet.y_g = np.zeros(n_genes)
			# (k, n_genes)
			mu_gk = agg.aggregate(facet.labels, X, func='mean', fill_value=0, size=facet.k, axis=0) + 0.01
			# (k, n_genes)
			p_gk = mu_gk / (mu_gk + self.r)
			# (n_genes)
			mu_g0 = X.mean(axis=0) + 0.01
			# (n_genes)
			p_g0 = mu_g0 / (mu_g0 + self.r)
			for c in range(n_cells):
				p_gkc = p_gk[facet.labels[c], :]
				facet.y_g += X[c, :] * (np.log(p_gkc) - np.log(p_g0))
				facet.y_g += self.r * np.log(1 - p_gkc) - np.log(1 - p_g0)

			all_yg[:, i] = facet.y_g

		# Compute the regularized likelihood gains
		all_yg_sum = np.sum(all_yg, axis=1)
		all_yg_regularized = 2 * all_yg - self.alpha * all_yg_sum[:, None]

		for i, facet in enumerate(self.facets):
			if len(facet.genes) > 0:
				facet.S = np.argsort(all_yg_regularized[:, i], axis=0)[-(facet.n_genes - len(facet.genes)):]
				facet.S[:len(facet.genes)] = facet.genes
			else:
				facet.S = np.argsort(all_yg_regularized[:, i], axis=0)[-facet.n_genes:]

	def _suggest_splits(self, X: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Calculate maximum likelihood gain for each gene when cells are split in two

		Args:
			X:			The data matrix (n_cells, n_genes)
			cells:		The cells to consider

		Returns:
			gains:		The likelihood gain for each gene (n_genes)
			thetas:		The optimal split point for each gene (n_genes)

		Remarks:
			This code closely follows the original MATLAB code by Kenneth Harris
		"""
		logging.debug("Calculating optimal splits for %d cells", cells.shape[0])
		n_cells = cells.shape[0]
		xs = X[cells, :].sort(axis=0)

		# cumulative sums for top and bottom halves of expression
		# - to evaluate splitting each gene in each position
		cx1 = np.cumsum(xs, axis=0)
		cx2 = cx1[::-1, :] - cx1

		# mean expression for top and bottom halves
		# n1 = 1..n_cells
		n1 = np.arange(n_cells) + 1
		n2 = n_cells - n1
		regN = 1e-4
		regD = 1
		mu1 = (cx1 + regN) / (n1 + regD)
		mu2 = (cx2 + regN) / (n2 + regD)

		# nbin parameters
		p1 = mu1 / (mu1 + self.r)
		p2 = mu2 / (mu2 + self.r)

		L1 = cx1 * np.log(p1) + self.r * (np.log(1 - p1) * n1)
		L2 = cx2 * np.log(p2) + self.r * (np.log(1 - p2) * n2)

		dL = (L1 + L2) - L1[::-1, :]
		split_pos = np.argmax(dL, axis=0)
		gains = dL[split_pos]
		thetas = np.choose(split_pos, xs)

		return (gains, thetas)

	def _evaluate_splits(self, facet: Facet, k: int, genes: np.ndarray, thetas: np.ndarray) -> Tuple[np.ndarray, int, float]:
		"""
		Evaluate how well it works to split this particular cluster by each of the given genes

		Args:
			facet:			The facet to work with
			k:				The cluster to split
			genes:			The genes to consider
			thetas:			The values to split by

		Returns:
			best_classes:	The classes of the best split (0s and 1s only)
			best_gene:		The best gene for splitting
			best_score:		The score of the best gene
		"""
		cells = np.where(facet.labels == k)[0]
		f0 = copy.copy(facet)
		f0.k = 1
		f0.adaptive = False
		fl = FacetLearning([f0], self.r, self.alpha, max_iter=1)
		fl.fit

# Deal with differences in cell size (in p_gk)
# Select S/k genes per group

# Bregman divergence

# px = x./(x + r); % negbin parameter for x
# py = y./(y + r); % negbin parameter for y

# bxy = x.*(log(px)-log(py)) + r*(log(1-px)-log(1-py));

# Note this is undefined if x or y=0, so you really need to compute image011.png, where image012.png is a
# regularization factor (0.1 seems to work well). The parameter r measures the amount of variability,
# 2 seems to work well for your data.
