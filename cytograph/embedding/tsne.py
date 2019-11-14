import numpy as np
import warnings
from pynndescent import NNDescent
from numba import NumbaPendingDeprecationWarning, NumbaPerformanceWarning
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import _joint_probabilities_nn

from cytograph.metrics import jensen_shannon_distance, multinomial_subspace_distance


def tsne(X: np.ndarray, *, n_components: int = 2, metric: str = "js", dof: int = 1, perplexity: int = 30, distances_nn: np.ndarray = None, neighbors_nn: np.ndarray = None, radius: float = 0.4) -> np.ndarray:
	"""
	Exploit the internals of sklearn TSNE to efficiently compute TSNE with either a Multinomial
	subspace or Jensen-Shannon distance metric. This will make use of multiple cores for the nearest-neighbor search.

	Args:
		X					Input matrix (n_samples, n_features)
		n_components		Number of components in embedding (2 or 3)
		metric				"js" (Jensen-Shannon), or "mns" (Multinomial subspace), or any other metric accepted by NNDescent
		dof					Degrees of freedom (n_components - 1 in standard TSNE)
		perplexity			Perplexity
		distances_nn		Precomputed nearest-neighbor distances (or None)
		neighbors_nn		Precomputed nearest-neighbor indices (or None)
		radius				Max distance (only applies when metric is js or mns)
	
	Remarks:
		If distances_nn and neighbors_nn are given, they will be used as-is to calculate the t-SNE P matrix
		(this will also implictly set k)

		If Multinomial subspace metric is used (i.e. metric == "mns"), then the input matrix must have
		negative values for features (genes) not in the local subspace.
	"""
	n_samples = X.shape[0]

	if distances_nn is None or neighbors_nn is None:
		k = min(n_samples - 1, int(3. * perplexity + 1))
		if metric == "js":
			m = jensen_shannon_distance
		elif metric == "mns":
			m = multinomial_subspace_distance
		else:
			m = metric

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=NumbaPerformanceWarning)  # Suppress warnings about numba not being able to parallelize code
			warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)  # Suppress warnings about future deprecations

			nn = NNDescent(data=X, metric=m)
			queue_size = min(5.0, n_samples / k)
			indices_nn, distances_nn = nn.query(X, k=k, queue_size=queue_size)
			indices_nn = indices_nn[:, 1:]
			distances_nn = distances_nn[:, 1:]

	if metric == "js" or metric == "mns":
		distances_nn[distances_nn > radius] = 1
	P = _joint_probabilities_nn(distances_nn, indices_nn, perplexity, False)
	
	pca = PCA(n_components=n_components, svd_solver='randomized')
	X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)

	tsne = TSNE(n_components=n_components, perplexity=perplexity)
	degrees_of_freedom = max(n_components - 1, 1)
	return tsne._tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded, neighbors=indices_nn, skip_num_points=0)
