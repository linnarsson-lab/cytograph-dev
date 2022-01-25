import os
import logging
import loompy
import numpy as np
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
from ..plotting.colors import colorize
from ..pipeline import Tempname
import networkx as nx
from .paris import paris
import community


def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return min(cpus[idx], 56)


def merge_small_clusters(G, merged_labels):

    logging.info(f"Starting with {merged_labels.max() + 1} labels")

    # get cell numbers
    _, n_cells = np.unique(merged_labels, return_counts=True)
    total = len(merged_labels)

    # get adjacency matrix
    partition = dict(zip(np.arange(total), merged_labels))
    induced = community.induced_graph(partition, G)
    M = nx.linalg.graphmatrix.adjacency_matrix(induced).A

    # while any small clusters remain
    while np.any(n_cells / total < 0.01):
        # recalculate clusters
        clusters = np.arange(len(n_cells))
        # cycle through each
        for c in np.unique(clusters):
            # check if cluster is too small
            if n_cells[c] / total < 0.01:

                # use uniform weights
                sim = M[c, :]

                # find max similarity
                adj_max = np.argsort(sim)
                adj_max = adj_max[-1] if adj_max[-1] != c else adj_max[-2]

                # change cell sizes
                new_size = np.array([n_cells[c] + n_cells[adj_max]])
                n_cells = np.delete(n_cells, [c, adj_max])
                n_cells = np.concatenate((n_cells, new_size))

                # change adjacency matrix
                new_adjacency = np.array(M[c] + M[adj_max])
                new_adjacency = np.delete(new_adjacency, [c, adj_max])
                M = np.delete(M, [c, adj_max], axis=0)
                M = np.delete(M, [c, adj_max], axis=1)
                M = np.hstack((M, new_adjacency[:, None]))
                new_adjacency = np.concatenate((new_adjacency, np.array([0])))
                M = np.vstack((M, new_adjacency))

                # record clusters
                merged_labels[(merged_labels == c) | (merged_labels == adj_max)] = merged_labels.max() + 1
                _, merged_labels = np.unique(merged_labels, return_inverse=True)
                break

    logging.info(f"Finished with {merged_labels.max() + 1} labels")

    return merged_labels


def split_subset(config, subset: str, method: str = 'coverage', hierarchy: str = 'agg', thresh: float = None) -> None:

    loom_file = os.path.join(config.paths.build, "data", subset + ".loom")
    out_dir = os.path.join(config.paths.build, "exported", subset, method)

    with Tempname(out_dir) as exportdir:
        os.mkdir(exportdir)
        with loompy.connect(loom_file) as ds:

            # Stop if only one cluster is left
            if ds.ca.Clusters.max() == 0:
                logging.info("Only one cluster found.")
                return False

            if method == 'dendrogram':

                logging.info("Splitting by dendrogram")
                # split dendrogram into two and get new clusters
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = np.array([branch[x] for x in ds.ca.Clusters])
                # plot split
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'coverage':

                logging.info("Splitting by dendrogram if coverage is above threshold")
                logging.info(f"Hierarchy: {hierarchy}")

                # Split dendrogram into two and get new clusters
                # possibly to be replaced with Paris clustering on the KNN
                if hierarchy == 'agg':

                    # change thresh to 0.98 if not specified
                    if thresh is None:
                        thresh = 0.98

                    # load KNN graph
                    logging.info("Loading KNN graph")
                    G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)

                    # get dendrogram from agg file and cut
                    logging.info("Splitting dendrogram in .agg file")
                    agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                    with loompy.connect(agg_file, 'r') as dsagg:
                        Z = dsagg.attrs.linkage
                        branch = cut_tree(Z, 2).T[0]

                    # Calculate clusters based on the dendrogram cut
                    clusters = np.array([branch[x] for x in ds.ca.Clusters])

                    # Check sizes
                    total = len(clusters)
                    if np.any(np.bincount(clusters) / total < 0.01):
                        logging.info(f"A cluster is too small.")
                        return False

                if hierarchy == 'paris':

                    # change thresh to 0.999 if not specified
                    if thresh is None:
                        thresh = 0.999

                    # load, partition, and cluster graph
                    logging.info("Loading RNN graph")
                    G = nx.from_scipy_sparse_matrix(ds.col_graphs.RNN)
                    logging.info("Partitioning graph by Cytograph clusters")
                    partition = dict(zip(np.arange(ds.shape[1]), ds.ca.Clusters))
                    induced = community.induced_graph(partition, G)
                    logging.info("Using Paris clustering on partitioned graph")
                    Z = paris(induced)

                    # calculate coverage for a range of cuts
                    coverage = []
                    upper = min(Z.shape[0], 25)
                    logging.info(f"Calculating coverage between cuts into 2 and {upper} branches")
                    for n in np.arange(2, upper + 1):
                        branch = cut_tree(Z, n_clusters=n).flatten()
                        clusters = np.array([branch[x] for x in ds.ca.Clusters])
                        partition = []
                        for c in np.unique(clusters):
                            partition.append(set(np.where(clusters == c)[0]))
                        coverage.append(nx.algorithms.community.quality.coverage(G, partition))

                    # cut dendrogram into two or last cut with a coverage over threshold
                    if not np.any(np.array(coverage) > thresh):
                        n = 2
                    else:
                        n = np.where(np.array(coverage) >= thresh)[0][-1] + 2
                    logging.info(f"Cutting Paris dendrogram into {n} branches")

                    # Calculate clusters based on this dendrogram cut
                    # and merge back small clusters
                    branch = cut_tree(Z, n_clusters=n).flatten()
                    logging.info("Merging small clusters")
                    clusters = np.array([branch[x] for x in ds.ca.Clusters])
                    clusters = merge_small_clusters(G, clusters)

                # Stop if only one cluster left
                if clusters.max() == 0:
                    logging.info(f"Only one cluster.")
                    return False

                # Calculate coverage of this partition on the graph
                logging.info("Calculating coverage of this partition")
                partition = []
                for c in np.unique(clusters):
                    partition.append(set(np.where(clusters == c)[0]))
                cov = nx.algorithms.community.quality.coverage(G, partition)

                # Stop if coverage is below thresh
                ds.attrs.Coverage = cov
                logging.info(f"Coverage threshold set at {thresh}")
                if cov < thresh:
                    logging.info(f"Partition is not separable: {cov:.5f}.")
                    return False

                # Otherwise save and plot separable clusters
                logging.info(f"Partition is separable: {cov:.5f}.")
                logging.info(f"Plotting partition")
                _, clusters = np.unique(clusters, return_inverse=True)
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.axis('off')
                plt.title(f"Coverage: {cov:.5f}", fontsize=20)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'cluster':

                logging.info("Splitting by clusters.")
                clusters = ds.ca.Clusters
                ds.ca.Split = clusters

        # Calculate split sizes
        sizes = np.bincount(clusters)
        logging.info("Creating punchcard")
        with open(f'punchcards/{subset}.yaml', 'w') as f:
            for i in np.unique(clusters):
                # Calc cpu usage
                n_cpus = calc_cpu(sizes[i])
                if n_cpus > 50:
                    memory = 750
                else:
                    memory = config.execution.memory
                # Write to punchcard
                name = chr(i + 65) if i < 26 else chr(i + 39) * 2
                f.write(f'{name}:\n')
                f.write('  include: []\n')
                f.write(f'  onlyif: Split == {i}\n')
                if sizes[i] <= 50:
                    f.write('  params:\n')
                    f.write(f'    k: {int(sizes[i] / 3)}\n')
                    f.write(f'    features: variance\n')
                elif sizes[i] <= 1000:
                    f.write(f'  steps: nn, embeddings, clustering, aggregate, export\n')
                f.write('  execution:\n')
                f.write(f'    n_cpus: {n_cpus}\n')
                f.write(f'    memory: {memory}\n')

    return True
