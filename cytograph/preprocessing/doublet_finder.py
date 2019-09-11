# This function written by Kimberly Siletti and is based on doubletFinder.R as forwarded by the Allen Institute:
#
# "Doublet detection in single-cell RNA sequencing data
#
# This function generates artificial nearest neighbors from existing single-cell RNA
# sequencing data. First, real and artificial data are merged. Second, dimension reduction
# is performed on the merged real-artificial dataset using PCA. Third, the proportion of
# artificial nearest neighbors is defined for each real cell. Finally, real cells are rank-
# ordered and predicted doublets are defined via thresholding based on the expected number
# of doublets.
#
# @param seu A fully-processed Seurat object (i.e. after normalization, variable gene definition,
# scaling, PCA, and tSNE).
# @param expected.doublets The number of doublets expected to be present in the original data.
# This value can best be estimated from cell loading densities into the 10X/Drop-Seq device.
# @param porportion.artificial The proportion (from 0-1) of the merged real-artificial dataset
# that is artificial. In other words, this argument defines the total number of artificial doublets.
# Default is set to 25%, based on optimization on PBMCs (see McGinnis, Murrow and Gartner 2018, BioRxiv).
# @param proportion.NN The proportion (from 0-1) of the merged real-artificial dataset used to define
# each cell's neighborhood in PC space. Default set to 1%, based on optimization on PBMCs (see McGinnis,
# Murrow and Gartner 2018, BioRxiv).
# @return An updated Seurat object with metadata for pANN values and doublet predictions.
# @export
# @examples
# seu <- doubletFinder(seu, expected.doublets = 1000, proportion.artificial = 0.25, proportion.NN = 0.01)"


import logging

import numpy as np
from pynndescent import NNDescent
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import loompy
from cytograph.decomposition import HPF
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph.metrics import jensen_shannon_distance
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from unidip import UniDip
from sklearn.ensemble import IsolationForest


def doublet_finder(ds: loompy.LoomConnection, use_pca: bool = False, proportion_artificial: float = 0.20, k: int = None) -> np.ndarray:
	# Step 1: Generate artificial doublets from input
	logging.debug("Creating artificial doublets")
	n_real_cells = ds.shape[1]
	n_doublets = int(n_real_cells / (1 - proportion_artificial) - n_real_cells)
	doublets = np.zeros((ds.shape[0], n_doublets))
	for i in range(n_doublets):
		a = np.random.choice(ds.shape[1])
		b = np.random.choice(ds.shape[1])
		doublets[:, i] = ds[:, a] + ds[:, b]

	data_wdoublets = np.concatenate((ds[:, :], doublets), axis=1)

	logging.debug("Feature selection and dimensionality reduction")
	genes = FeatureSelectionByVariance(2000).fit(ds)
	if use_pca:
		# R function uses log2 counts/million
		f = np.divide(data_wdoublets.sum(axis=0), 10e6)
		norm_data = np.divide(data_wdoublets, f)
		norm_data = np.log(norm_data + 1)
		pca = PCA(n_components=50).fit_transform(norm_data[genes, :].T)
	else:
		data = sparse.coo_matrix(data_wdoublets[genes, :]).T
		hpf = HPF(k=64, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
	
	if k is None:
		k = int(np.min([100, ds.shape[1] * 0.01]))

	logging.info(f"Initialize NN structure with k = {k}")
	if use_pca:
		knn_result = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
		knn_result.fit(pca)
		knn_dist, knn_idx = knn_result.kneighbors(X=pca, return_distance=True)

		num = ds.shape[1]
		knn_result1 = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
		knn_result1.fit(pca[0:num, :])
		knn_dist1, knn_idx1 = knn_result1.kneighbors(X=pca[num + 1:, :], n_neighbors=10)
	else:
		knn_result = NNDescent(data=theta, metric=jensen_shannon_distance)
		knn_idx, knn_dist = knn_result.query(theta, k=k)

		num = ds.shape[1]
		knn_result1 = NNDescent(data=theta[0:num, :], metric=jensen_shannon_distance)
		knn_idx1, knn_dist1 = knn_result1.query(theta[num + 1:, :], k=k)

	dist_th = np.mean(knn_dist1.flatten()) + 1.64 * np.std(knn_dist1.flatten())

	doublet_freq = np.logical_and(knn_idx > ds.shape[1], knn_dist < dist_th)
	doublet_freq_A = doublet_freq[ds.shape[1]:ds.shape[1]+n_doublets, :]
	mean1 = doublet_freq_A.mean(axis=1)
	mean2 = doublet_freq_A[:, 0:int(np.ceil(k / 2))].mean(axis=1)
	doublet_score_A = np.maximum(mean1, mean2)
	
	doublet_freq = doublet_freq[0:ds.shape[1], :]
	mean1 = doublet_freq.mean(axis=1)
	mean2 = doublet_freq[:, 0:int(np.ceil(k / 2))].mean(axis=1)
	doublet_score = np.maximum(mean1, mean2)
	doublet_flag = np.zeros(ds.shape[1],int)
	#Infer TH from the data
	# instantiate and fit the KDE model
	kde = KernelDensity(bandwidth=0.1  , kernel='gaussian')
	kde.fit(doublet_score_A[:, None])

	# score_samples returns the log of the probability density
	xx = np.linspace(doublet_score_A.min(), doublet_score_A.max(), len(doublet_score_A)).reshape(-1,1)

	logprob = kde.score_samples(xx)
	#Check if the distribution is bimodal
	intervals = UniDip(np.exp(logprob)).run()
	if (len(intervals)>1):
		kmeans = KMeans(n_clusters=2).fit(doublet_score_A.reshape(len(doublet_score_A),1))
		high_cluster = np.where(kmeans.cluster_centers_==max(kmeans.cluster_centers_))[0][0]
		doublet_th1 = np.round(np.min(doublet_score_A[kmeans.labels_==high_cluster]),2)
	else:
		isolation_forest = IsolationForest(n_estimators=100)
		isolation_forest.fit(doublet_score_A.reshape(-1, 1))
		anomaly_score = isolation_forest.decision_function(xx)
		outlier = isolation_forest.predict(xx)
		ind_outliers = np.where((outlier==-1))[0]
		doublet_th1 = min(xx[ind_outliers[np.where(xx[ind_outliers]>0.2)[0]]])		
	
	#0.5% for every 1000 cells
	doublet_th2 = np.percentile(doublet_score,100-(5e-4*ds.shape[1]))
	if (len(np.where(doublet_score>=doublet_th1)[0])>(len(np.where(doublet_score>=doublet_th2)[0])+(0.03*ds.shape[1]))):
		doublet_th = doublet_th2
	else:
		doublet_th = doublet_th1
	logging.info(f"TH1: {doublet_th1} TH2: {doublet_th2} TH: {doublet_th}")
	doublet_flag[doublet_score>=doublet_th]=1
	logging.debug(f"doublet rate: {len(np.where(doublet_flag==1)[0])/ds.shape[1]}")
	return doublet_score,doublet_flag
