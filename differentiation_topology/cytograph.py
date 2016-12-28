import copy
import json
import logging
import os
from datetime import datetime
from typing import *
from multiprocessing import Pool
import loompy
import matplotlib.pyplot as plt
import numpy as np
from palettable.tableau import Tableau_20
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from annoy import AnnoyIndex
import networkx as nx
# import community
import differentiation_topology as dt

colors20 = np.array(Tableau_20.mpl_colors)

default_config = {
	"sample_dir": "/Users/Sten/loom-datasets/Whole brain/",
	"build_dir": "/Users/Sten/builds/build_20161127_180745",
	"tissue": "Dentate gyrus",
	"samples": ["10X43_1", "10X46_1"],

	"preprocessing": {
		"do_validate_genes": True,
		"make_doublets": False
	},
	"knn": {
		"k": 50,
		"n_trees": 50,
		"mutual": True,
		"min_cells": 10
	},
	"louvain_jaccard": {
		"cache_n_columns": 5000,
		"n_components": 50,
		"n_genes": 2000,
		"normalize": True,
		"standardize": False
	},
	"prommt": {
		"n_genes": 1000,
		"n_S": 100,
		"k": 5,
		"max_iter": 100
	},
	"annotation": {
		"pep": 0.05,
		"f": 0.2,
		"annotation_root": "/Users/Sten/Dropbox (Linnarsson Group)/Code/autoannotation/"
	}
}


def cytograph(config: Dict = default_config) -> Any:
	skip_preprocessing = False
	if config["build_dir"] is None:
		config["build_dir"] = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S")
		os.mkdir(config["build_dir"])
	else:
		skip_preprocessing = True
	build_dir = config["build_dir"]
	tissue = config["tissue"]
	samples = config["samples"]
	sample_dir = config["sample_dir"]
	fname = os.path.join(build_dir, tissue.replace(" ", "_") + ".loom")

	config_json = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
	with open(os.path.join(build_dir, config_json), 'w') as f:
		f.write(json.dumps(config))

	# logging.basicConfig(filename=os.path.join(build_dir, tissue.replace(" ", "_") + ".log"))
	logging.info("Processing: " + tissue)

	# Preprocessing
	if not skip_preprocessing:
		logging.info("Preprocessing")
		dt.preprocess(sample_dir, samples, fname, {"title": tissue}, False, True)

	ds = loompy.connect(fname)
	n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
	n_total = ds.shape[1]
	logging.info("%d of %d cells were valid", n_valid, n_total)
	logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])

	# KNN graph generation and clustering
	cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
	logging.info("PCA projection")
	transformed = pca_projection(ds, cells, config["louvain_jaccard"])
	logging.info("Generating mutual KNN graph")
	(knn, ok_cells) = make_knn(transformed, ds.shape[1], cells, config["knn"])

	# logging.info("ProMMT clustering")
	# labels = prommt(ds, ok_cells, config["prommt"])
	# n_labels = max(labels) + 1
	# logging.info("Found %d clusters", n_labels)

	logging.info("Facet learning")
	labels = facets(ds, cells)
	logging.info(labels.shape)
	n_labels = np.max(labels, axis=0) + 1
	logging.info("Found " + str(n_labels) + " clusters")
	# Make labels for excluded cells == -1
	labels_all = np.zeros((ds.shape[1], labels.shape[1]), dtype='int') + -1
	labels_all[cells, :] = labels

	# Layout
	logging.info("t-SNE layout")
	tsne = TSNE().fit_transform(transformed)
	tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne, axis=0)
	tsne_all[cells] = tsne

	logging.info("Marker enrichment and trinarization")
	f = config["annotation"]["f"]
	pep = config["annotation"]["pep"]
	(enrichment, trinary_prob, trinary_pat) = dt.expression_patterns(ds, labels_all[ok_cells, 2] + 1, pep, f, ok_cells)
	save_diff_expr(ds, build_dir, tissue, enrichment, trinary_pat, trinary_prob)

	# Auto-annotation
	logging.info("Auto-annotating cell types and states")
	aa = dt.AutoAnnotator(ds, root=config["annotation"]["annotation_root"])
	(tags, annotations) = aa.annotate(ds, trinary_prob)
	sizes = np.bincount(labels_all[:, 2] + 1)
	save_auto_annotation(build_dir, tissue, sizes, annotations, tags)

	logging.info("Plotting clusters on mutual-KNN graph")
	for i in range(labels.shape[1]):
		pl = False
		if i == 2:
			pl = True
		plot_clusters(knn, labels_all[:, i] + 1, tsne_all, tags, annotations, title=tissue, plt_labels=pl, outfile=os.path.join(build_dir, tissue + "_" + str(i)))

	logging.info("Saving attributes")
	ds.set_attr("tSNE_X", tsne_all[:, 0], axis=1)
	ds.set_attr("tSNE_Y", tsne_all[:, 1], axis=1)
	ds.set_attr("Louvain_Jaccard", labels_all, axis=1)
	logging.info("Done.")
	return (knn, tsne_all, labels_all, ok_cells, tags, annotations, enrichment, trinary_pat, trinary_prob)


def save_auto_annotation(build_dir: str, tissue: str, sizes: np.ndarray, annotations: np.ndarray, tags: np.ndarray) -> None:
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_annotations.tab"), "w") as f:
		f.write("\t")
		for j in range(annotations.shape[1]):
			f.write(str(j + 1) + " (" + str(sizes[j]) + ")\t")
		f.write("\n")
		for ix, tag in enumerate(tags):
			f.write(str(tag) + "\t")
			for j in range(annotations.shape[1]):
				f.write(str(annotations[ix, j]) + "\t")
			f.write("\n")


def save_diff_expr(ds: loompy.LoomConnection, build_dir: str, tissue: str, enrichment: np.ndarray, trinary_pat: np.ndarray, trinary_prob: np.ndarray) -> None:
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_diffexpr.tab"), "w") as f:
		f.write("Gene\t")
		f.write("Valid\t")
		for ix in range(enrichment.shape[1]):
			f.write("Enr_" + str(ix + 1) + "\t")
		for ix in range(trinary_pat.shape[1]):
			f.write("Trin_" + str(ix + 1) + "\t")
		for ix in range(trinary_prob.shape[1]):
			f.write("Prob_" + str(ix + 1) + "\t")
		f.write("\n")

		for row in range(enrichment.shape[0]):
			f.write(ds.Gene[row] + "\t")
			really_valid = 1
			if "_Valid" in ds.row_attrs and not ds.row_attrs["_Valid"][row] == 1:
				really_valid = 0
			if "_Excluded" in ds.row_attrs and not ds.row_attrs["_Excluded"][row] == 0:
				really_valid = 0
			f.write(str(really_valid) + "\t")
			for ix in range(enrichment.shape[1]):
				f.write(str(enrichment[row, ix]) + "\t")
			for ix in range(trinary_pat.shape[1]):
				f.write(str(trinary_pat[row, ix]) + "\t")
			for ix in range(trinary_prob.shape[1]):
				f.write(str(trinary_prob[row, ix]) + "\t")
			f.write("\n")


def facets(ds: loompy.LoomConnection, cells: np.ndarray, config: Dict={"n_genes": 5000}) -> np.ndarray:
	n_genes = config["n_genes"]

	# Compute an initial gene set
	logging.info("Selecting genes for Facet Learning")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, _, _) = feature_selection(ds, n_genes, cells)
	facet_genes = np.where(np.in1d(ds.Gene, ["Xist", "Tsix", "Top2a", "Cdk1", "Plk1", "Cenpe"]))[0]
	genes = np.union1d(genes, facet_genes)

	logging.info("Loading data (in batches)")
	m = np.zeros((cells.shape[0], genes.shape[0]), dtype='int')
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=5000):
		vals = vals[genes, :].transpose()
		n_cells_in_batch = selection.shape[0]
		m[j:j + n_cells_in_batch, :] = vals
		j += n_cells_in_batch

	logging.info("Facet learning with three facets")

	# Get indexes for genes
	def gix(names: List[str]) -> List[int]:
		return [np.where(ds.Gene[genes] == n)[0][0] for n in names]
	f0 = dt.Facet("sex", k=2, n_genes=10, genes=gix(["Xist", "Tsix"]), adaptive=False)
	f1 = dt.Facet("cell cycle", k=2, n_genes=20, genes=gix(["Top2a", "Cdk1", "Plk1", "Cenpe"]), adaptive=False)
	f2 = dt.Facet("cell type", k=15, n_genes=100, genes=[], adaptive=True)
	labels = dt.FacetLearning([f0, f1, f2], r=2, max_iter=100).fit_transform(m)
	return labels


def prommt(ds: loompy.LoomConnection, cells: np.ndarray, config: Dict) -> np.ndarray:
	n_genes = config["n_genes"]
	n_S = config["n_S"]
	k = config["k"]
	max_iter = config["max_iter"]

	# Compute an initial gene set
	logging.info("Selecting genes for ProMMT")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, _, _) = feature_selection(ds, n_genes, cells)

	logging.info("Loading data (in batches)")
	m = np.zeros((cells.shape[0], genes.shape[0]), dtype='int')
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=5000):
		vals = vals[genes, :].transpose()
		n_cells_in_batch = selection.shape[0]
		m[j:j + n_cells_in_batch, :] = vals
		j += n_cells_in_batch

	logging.info("ProMMT clustering")
	labels = dt.ProMMT(n_S=n_S, k=k, max_iter=max_iter).fit_transform(m)
	return labels

def sfdp(knn, )

class Normalizer(object):
	def __init__(self, ds: loompy.LoomConnection, config: Dict, mu: np.ndarray = None, sd: np.ndarray = None) -> None:
		if (mu is None) or (sd is None):
			(self.sd, self.mu) = ds.map([np.std, np.mean], axis=0)
		else:
			self.sd = sd
			self.mu = mu
		self.totals = ds.map(np.sum, axis=1)
		self.config = config

	def normalize(self, vals: np.ndarray, cells: np.ndarray) -> np.ndarray:
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values
		"""
		if self.config["normalize"]:
			# Adjust total count per cell to 10,000
			vals = vals / (self.totals[cells] + 1) * 10000
		# Log transform
		vals = np.log(vals + 1)
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		if self.config["standardize"]:
			# Scale to unit standard deviation per gene
			vals = self._div0(vals, self.sd[:, None])
		return vals

	def _div0(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
		with np.errstate(divide='ignore', invalid='ignore'):
			c = np.true_divide(a, b)
			c[~np.isfinite(c)] = 0  # -inf inf NaN
		return c


def pca_projection(ds: loompy.LoomConnection, cells: np.ndarray, config: Dict) -> np.ndarray:
	"""
	Memory-efficient PCA projection of the dataset

	Args:
		ds (LoomConnection): 	Dataset
		cells (ndaray of int):	Indices of cells to project
		config (dict):			Dict of settings

	Returns:
		The dataset transformed by the top principal components
		Shape: (n_samples, n_components), where n_samples = cells.shape[0]
	"""
	cache_n_columns = config["cache_n_columns"]
	n_genes = config["n_genes"]
	n_cells = cells.shape[0]
	n_components = config["n_components"]

	# Compute an initial gene set
	logging.info("Selecting genes")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, mu, sd) = feature_selection(ds, n_genes, cells)

	# Perform PCA based on the gene selection and the cell sample
	logging.info("Computing aggregate statistics for normalization")
	normalizer = Normalizer(ds, config, mu, sd)

	logging.info("Incremental PCA in batches of %d", cache_n_columns)
	pca = IncrementalPCA(n_components=n_components)
	for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=cache_n_columns):
		vals = normalizer.normalize(vals, ix + selection)
		pca.partial_fit(vals[genes, :].transpose())		# PCA on the selected genes

	logging.info("Projecting cells to PCA space (in batches)")
	transformed = np.zeros((cells.shape[0], pca.n_components_))
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=cache_n_columns):
		vals = normalizer.normalize(vals, selection)
		n_cells_in_batch = selection.shape[0]
		temp = pca.transform(vals[genes, :].transpose())
		transformed[j:j + n_cells_in_batch, :] = pca.transform(vals[genes, :].transpose())
		j += n_cells_in_batch

	return transformed


def make_knn(m: np.ndarray, d: int, cells: np.ndarray, config: Dict) -> Tuple[sparse.coo_matrix, np.ndarray]:
	k = config["k"]
	n_trees = config["n_trees"]
	mutual = config["mutual"]
	min_cells = config["min_cells"]
	n_components = m.shape[1]
	logging.info("Creating approximate nearest neighbors model (annoy)")
	annoy = AnnoyIndex(n_components, metric="euclidean")
	for ix, cell in enumerate(cells):
		annoy.add_item(cell, m[ix, :])
	annoy.build(n_trees)

	logging.info("Computing mutual nearest neighbors")
	I = np.empty(d * k)
	J = np.empty(d * k)
	V = np.empty(d * k)
	for i in range(d):
		(nn, w) = annoy.get_nns_by_item(i, k, include_distances=True)
		w = np.array(w)
		I[i * k:(i + 1) * k] = [i] * k
		J[i * k:(i + 1) * k] = nn
		V[i * k:(i + 1) * k] = w

	# k nearest neighbours
	knn = sparse.coo_matrix((V, (I, J)), shape=(d, d))

	data = knn.data
	rows = knn.row
	cols = knn.col

	# Convert to similarities by rescaling and subtracting from 1
	data = data / data.max()
	data = 1 - data

	knn = sparse.coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()

	if mutual:
		# Compute Mutual knn
		# This removes all edges that are not reciprocal
		knn = knn.minimum(knn.transpose())
	else:
		# Make all edges reciprocal
		# This duplicates all edges that are not reciprocal
		knn = knn.maximum(knn.transpose())

	# Find and remove disconnected components
	logging.info("Identifying cells in small components")
	(_, labels) = sparse.csgraph.connected_components(knn, directed='False')
	sizes = np.bincount(labels)
	ok_cells = np.where((sizes > min_cells)[labels])[0]
	logging.info("Small components contained %d cells", cells.shape[0] - ok_cells.shape[0])

	return (knn, ok_cells)


def feature_selection(ds: loompy.LoomConnection, n_genes: int, cells: np.ndarray = None, cache: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Fits a noise model (CV vs mean)

	Args:
		ds (LoomConnection):	Dataset
		n_genes (int):	number of genes to include
		cells (ndarray): cells to include when computing mean and CV (or None)
		cache (ndarray): dataset corresponding to the selected cells (or None)

	Returns:
		ndarray of selected genes (list of ints)
	"""
	(mu, std) = ds.map((np.mean, np.std), axis=0, selection=cells)

	valid = np.logical_and(
		np.logical_and(
			ds.row_attrs["_Valid"] == 1,
			ds.row_attrs["Gene"] != "Xist"
		),
		ds.row_attrs["Gene"] != "Tsix"
	).astype('int')

	ok = np.logical_and(mu > 0, std > 0)
	cv = std[ok] / mu[ok]
	log2_m = np.log2(mu[ok])
	log2_cv = np.log2(cv)

	svr_gamma = 1000. / len(mu[ok])
	clf = SVR(gamma=svr_gamma)
	clf.fit(log2_m[:, np.newaxis], log2_cv)
	fitted_fun = clf.predict
	# Score is the relative position with respect of the fitted curve
	score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
	score = score * valid[ok]
	top_genes = np.where(ok)[0][np.argsort(score)][-n_genes:]

	logging.debug("Keeping %i genes" % top_genes.shape[0])
	logging.info(str(sorted(ds.Gene[top_genes[:50]])))
	return (top_genes, mu, std)


def louvain_jaccard(knn: sparse.coo_matrix, jaccard: bool = False, cooling_step: float = 0.95) -> Tuple[Any, np.ndarray, np.ndarray]:
	"""
	From knn, make a graph-tool Graph object, a Louvain partitioning and a layout position list

	Args:
		knn (COO sparse matrix):	knn adjacency matrix
		jaccard (bool):				If true, replace knn edge weights with Jaccard similarities

	Returns:
		g (graph.tool Graph):		the Graph corresponding to the knn matrix
		labels (ndarray of int): 	Louvain partition label for each node in the graph
		sfdp (ndarray matrix):		X,Y position for each node
	"""
	logging.info("Creating graph")
	g = gt.Graph(directed=False)

	# Keep only half the edges, so the result is undirected
	sel = np.where(knn.row < knn.col)[0]
	logging.info("Graph has %d edges", sel.shape[0])

	g.add_vertex(n=knn.shape[0])
	edges = np.stack((knn.row[sel], knn.col[sel]), axis=1)
	g.add_edge_list(edges)
	w = g.new_edge_property("double")
	if jaccard:
		js = []
		knncsr = knn.tocsr()
		for i, j in edges:
			r = knncsr.getrow(i)
			c = knncsr.getrow(j)
			shared = r.minimum(c).nnz
			total = r.maximum(c).nnz
			js.append(shared / total)
		w.a = np.array(js)
	else:
		# use the input edge weights
		w.a = knn.data[sel]

	logging.info("Louvain partitioning")
	partitions = community.best_partition(nx.from_scipy_sparse_matrix(knn))
	labels = np.fromiter(partitions.values(), dtype='int')

	logging.info("Creating graph layout")
	# label_prop = g.new_vertex_property("int", vals=labels)
	sfdp = gt.sfdp_layout(g, eweight=w, epsilon=0.0001, cooling_step=cooling_step).get_2d_array([0, 1]).transpose()

	return (g, labels, sfdp)


def plot_clusters(knn: np.ndarray, labels: np.ndarray, pos: Dict[int, Tuple[int, int]], tags: np.ndarray, annotations: np.ndarray, title: str = None, plt_labels: bool = True, outfile: str = None) -> None:
	# Plot auto-annotation
	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(knn)
	ax = fig.add_subplot(111)

	# Draw the KNN graph first, with gray transparent edges
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw_networkx_edges(g, pos=pos, alpha=0.5, width=0.1, edge_color='gray')

	# Then draw the nodes, colored by label
	block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
	nx.draw(g, pos=pos, node_color=block_colors, node_size=10, alpha=0.5, width=0.1, linewidths=0)
	if plt_labels:
		for lbl in range(max(labels) + 1):
			if np.sum(labels == lbl) == 0:
				continue
			(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
			text_labels = ["#" + str(lbl + 1)]
			for ix, a in enumerate(annotations[:, lbl]):
				if a >= 0.5:
					text_labels.append(tags[ix].abbreviation)
			if len(text_labels) > 0:
				text = "\n".join(text_labels)
			else:
				text = str(lbl + 1)
			ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.2, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_annotated.pdf")
		plt.close()

if __name__ == "__main__":
	result = cytograph()
