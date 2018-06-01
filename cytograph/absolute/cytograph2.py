import cytograph as cg
import numpy as np
import scipy.sparse as sparse
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from typing import *
import os
import igraph


class Cytograph2:
	def __init__(self, n_genes: int = 1000, n_factors: int = 200, k: int = 25, k_smoothing: int = 10, max_iter: int = 200) -> None:
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_smoothing = k_smoothing
		self.k = k
		self.max_iter = max_iter
		self.smoothing = (True if k_smoothing > 0 else False)

	def fit(self, ds: loompy.LoomConnection) -> None:
		if self.smoothing:
			# Select genes
			logging.info(f"Selecting {self.n_genes} genes")
			normalizer = cg.Normalizer(False)
			normalizer.fit(ds)
			genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
			self.genes = genes
			data = ds.sparse(rows=genes).T

			# HPF factorization
			logging.info(f"HPF to {self.n_factors} latent factors")
			hpf = cg.HPF(a=1, b=10, c=1, d=10, k=self.n_factors, max_iter=self.max_iter)
			hpf.fit(data)

			# KNN in HPF space
			logging.info(f"Computing KNN (k={self.k_smoothing}) in latent space")
			theta = np.log(hpf.theta)
			theta = (theta - theta.min(axis=0))
			theta = theta / theta.max(axis=0)
			hpfn = normalize(theta)  # This converts euclidean distances to cosine distances (ball_tree doesn't directly support cosine)
			nn = NearestNeighbors(self.k_smoothing, algorithm="ball_tree", metric='euclidean', n_jobs=4)
			nn.fit(hpfn)
			knn = nn.kneighbors_graph(hpfn, mode='connectivity')
			knn.setdiag(1)

			# Poisson smoothing (in place)
			logging.info(f"Poisson smoothing")
			for (ix, indexes, view) in ds.scan(axis=0):
				ds[indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = ds.sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(a=1, b=10, c=1, d=10, k=self.n_factors, max_iter=self.max_iter)
		hpf.fit(data)

		logging.info(f"Saving normalized latent factors")
		beta = np.log(hpf.beta)
		beta = (beta - beta.min(axis=0))
		beta = beta / beta.max(axis=0)
		beta_all = np.zeros((ds.shape[0], beta.shape[1]))
		beta_all[genes] = beta
		ds.ra.HPF = beta_all

		theta = np.log(hpf.theta)
		theta = (theta - theta.min(axis=0))
		theta = theta / theta.max(axis=0)
		totals = ds.map([np.sum], axis=1)[0]
		theta = (theta.T / totals).T * np.median(totals)
		ds.ca.HPF = theta

		logging.info(f"tSNE embedding from latent space")
		tsne = TSNE(metric="cosine").fit_transform(theta)
		ds.ca.TSNE = tsne

		logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		hpfn = normalize(theta)  # This makes euclidean distances equivalent to cosine distances (ball_tree doesn't support cosine)
		bnn = cg.BalancedKNN(k=self.k, metric="euclidean", maxl=2 * self.k, sight_k=2 * self.k)
		bnn.fit(hpfn)
		knn = bnn.kneighbors_graph(mode='connectivity')
		mknn = knn.minimum(knn.transpose())
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info(f"Poisson resampling and HPF projection to latent space")
		data.data = np.random.poisson(data.data)  # this replaces the non-zero value with poisson samples of the same mean
		hpf.transform(data)
		theta = np.log(hpf.theta)
		theta = (theta - theta.min(axis=0))
		theta = theta / theta.max(axis=0)
		totals = ds.map([np.sum], axis=1)[0]
		theta = (theta.T / totals).T * np.median(totals)

		logging.info(f"Computing KNN (k = 10) of Poisson samples in latent space")
		hpfn = normalize(theta)  # This makes euclidean distances equivalent to cosine distances (ball_tree doesn't support cosine)
		nn = NearestNeighbors(n_neighbors=10, metric="euclidean", algorithm="ball_tree")
		nn.fit(hpfn)
		knnp = nn.kneighbors_graph(mode='connectivity')
		ds.col_graphs.KNNP = knnp

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain()
		labels = pl.fit_predict(ds, "KNN")
		ds.ca.Clusters = labels + 1
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")