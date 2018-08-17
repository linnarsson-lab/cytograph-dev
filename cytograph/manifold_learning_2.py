from typing import *
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import numpy as np
import random
import logging
import cytograph as cg
import loompy


class ManifoldLearning2:
	def __init__(self, *, n_genes: int = 1000, k: int = 100, gtsne: bool = True, alpha: float = 1, genes: np.ndarray = None, filter_cellcycle: str = None, layer: str=None, max_iter: int = 1000) -> None:
		self.n_genes = n_genes
		self.k = k
		self.gtsne = gtsne
		self.alpha = alpha
		self.genes = genes
		self.filter_cellcycle = filter_cellcycle
		self.layer = layer
		self.max_iter = max_iter

	def fit(self, ds: loompy.LoomConnection, initial_pos: np.ndarray = None, nng: np.ndarray = None, blocked_genes: np.ndarray = None) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, np.ndarray]:
		"""
		Discover the manifold
		Args:
			n_genes			Number of genes to use for manifold learning (ignored if genes is not None)
			gtsnse			Use graph t-SNE for layout (default: standard tSNE)
			alpha			The scale parameter for multiscale KNN
			genes			List of genes to use for manifold learning
			initial_pos		Use this initial layout, shape (ds.shape[1], 2)
			nng				Non-neuronal genes, set these to zero in neurons (mask array)
			blocked_gens	Don't use these genes (mask array)

		Returns:
			knn		The multiscale knn graph as a sparse matrix, with k = 100
			mknn	Mutual knn subgraph, with k = 20
			pos		2D projection (t-SNE or gt-SNE) as ndarray with shape (n_cells, 2)
		"""
		n_cells = ds.shape[1]
		logging.info("Processing all %d cells", n_cells)
		logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		if self.filter_cellcycle is not None:
			cell_cycle_genes = np.array(open(self.filter_cellcycle).read().split())
			mask = np.in1d(ds.ra.Gene, cell_cycle_genes)
			if np.sum(mask) == 0:
				logging.warn("None cell cycle genes where filtered, check your gene list")
		else:
			mask = None
		
		if blocked_genes is not None:
			if mask is None:
				mask = blocked_genes
			else:
				mask = mask & blocked_genes

		if self.genes is None:
			logging.info("Selecting up to %d genes", self.n_genes)
			genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd, mask=mask)

			n_components = min(50, n_cells)
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca_transformed = pca.fit_transform(ds, normalizer)
			transformed = pca_transformed

			logging.info("Generating balanced KNN graph")
			np.random.seed(0)
			k = min(self.k, n_cells - 1)
			bnn = cg.BalancedKNN(k=k, maxl=2 * k, sight_k=2 * k)
			bnn.fit(transformed)
			knn = bnn.kneighbors_graph(mode='connectivity')
			knn = knn.tocoo()
			mknn = knn.minimum(knn.transpose()).tocoo()

			logging.info("MKNN-Louvain clustering with outliers")
			(a, b, w) = (mknn.row, mknn.col, mknn.data)
			random.seed(13)
			lj = cg.LouvainJaccard(resolution=1, jaccard=False)
			labels = lj.fit_predict(knn)
			bigs = np.where(np.bincount(labels) >= 10)[0]
			mapping = {k: v for v, k in enumerate(bigs)}
			labels = np.array([mapping[x] if x in bigs else -1 for x in labels])

			n_labels = np.max(labels) + 1
			logging.info("Found " + str(n_labels) + " clusters")

			logging.info("Marker selection")
			temp = None
			if "Clusters" in ds.ca:
				temp = ds.ca.Clusters
			ds.ca.Clusters = labels - labels.min()
			(genes, _, _) = cg.MarkerSelection(n_markers=int(500 / n_labels), mask=mask, findq=False).fit(ds)
			if temp is not None:
				ds.ca.Clusters = temp
		else:
			genes = self.genes

		temp = np.zeros(ds.shape[0], dtype='bool')
		temp[genes] = True
		ds.ra._Selected = temp.astype('int')
		logging.info("%d genes selected", temp.sum())

		if self.genes is None:
			# Select cells across clusters more uniformly, preventing a single cluster from dominating the PCA
			cells_adjusted = cg.cap_select(labels - labels.min(), np.arange(n_cells), int(n_cells * 0.2))
			n_components = min(50, cells_adjusted.shape[0])
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca.fit(ds, normalizer, cells=cells_adjusted)
		else:
			n_components = min(50, n_cells)
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca.fit(ds, normalizer)
			
		# Note that here we're transforming all cells; we just did the fit on the selection
		transformed = pca.transform(ds, normalizer)

		k = min(self.k, n_cells - 1)
		logging.info("Generating multiscale KNN graph (k = %d)", k)
		bnn = cg.BalancedKNN(k=k, maxl=2 * k, sight_k=2 * k)
		bnn.fit(transformed)
		knn = bnn.kneighbors(mode='connectivity')[1][:, 1:]
		n_cells = knn.shape[0]
		a = np.tile(np.arange(n_cells), k)
		b = np.reshape(knn.T, (n_cells * k,))
		w = np.repeat(1 / np.power(np.arange(1, k + 1), self.alpha), n_cells)
		knn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells))
		threshold = w > 0.025
		mknn = sparse.coo_matrix((w[threshold], (a[threshold], b[threshold])), shape=(n_cells, n_cells))
		mknn = mknn.minimum(mknn.transpose()).tocoo()

		perplexity = min(k, (n_cells - 1) / 3 - 1)
		if self.gtsne:
			logging.info("gt-SNE layout")
			# Note that perplexity argument is ignored in this case, but must still be given
			# because bhtsne will check that it has a valid value
			tsne_pos = cg.TSNE(perplexity=perplexity, max_iter=self.max_iter).layout(transformed, knn=knn.tocsr(), initial_pos=initial_pos)
		else:
			logging.info("t-SNE layout")
			tsne_pos = cg.TSNE(perplexity=perplexity, max_iter=self.max_iter).layout(transformed, initial_pos=initial_pos)

		return (knn, mknn, tsne_pos)
