from typing import *
import numpy as np
import numpy_groupies as npg
import pandas as pd
import loompy
from collections import defaultdict
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import logging
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker as ticker


def loompy2data(filename: str) -> pd.DataFrame:
	ds = loompy.connect(filename)
	return pd.DataFrame(data=ds[:, :], columns=ds.col_attrs['CellID'], index=ds.row_attrs['Gene']).astype(int)


def loompy2annot(filename: str) -> pd.DataFrame:
	ds = loompy.connect(filename)
	return pd.DataFrame(ds.col_attrs, index=ds.col_attrs['CellID']).T


def loompy2data_annot(filename: str) -> Tuple[loompy.LoomConnection, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	ds = loompy.connect(filename)
	ret= (ds,
			pd.DataFrame(data=ds[:, :],
						columns=ds.col_attrs['CellID'],
						index=ds.row_attrs['Accession']).astype(int),
			pd.DataFrame( ds.col_attrs,
						index=ds.col_attrs['CellID'] ).T,
			pd.DataFrame( ds.row_attrs,
						index=ds.row_attrs['Accession'] ).T)
	ds.close()
	return ret


def marker_table(df: pd.DataFrame, groups: np.ndarray, avg_N: int = 30) -> Tuple[DefaultDict, np.ndarray]:
	logging.debug("Computing Marker Table")
	X = df.values
	N = int(np.ceil(avg_N / 3))
	mus = npg.aggregate_numba.aggregate(groups, X, func="mean", axis=1)
	counts_per_group = npg.aggregate_numba.aggregate(groups,1)
	mu0 = np.sum((mus * counts_per_group) / len(groups), 1)  # faster than X.mean(1)
	fold = np.zeros_like(mus)
	iz = mu0 > 0.001  # avoid Nans and useless high precision calculations
	fold[iz, :] = mus[iz, :] / mu0[iz, None]
	fs = npg.aggregate_numba.aggregate(groups, X > 0, func="mean", axis=1)
	
	# Filters
	fold *= (mus > 1) * (fold > 1.5)
	fs *= (fs > 0.25)
	
	# Scores and sorting
	score00 = fold
	score05 = fold * fs**0.5
	score10 = fold * fs
	ix00 = np.argsort(score00, 0)[::-1, :]
	ix05 = np.argsort(score05, 0)[::-1, :]
	ix10 = np.argsort(score10, 0)[::-1, :]
	score10_df = pd.DataFrame(score10, columns=np.unique(groups), index=df.index)
	genes00 = df.index.values[ix00][:N, :]
	genes05 = df.index.values[ix05][:N, :]
	genes10 = df.index.values[ix10][:N, :]
	markers_array = np.vstack((genes00, genes05,genes10))
	markers = defaultdict(set)  # type: defaultdict
	for ct in range(mus.shape[1]):
		markers[ct] |= set(markers_array[:,ct])
	for ct in range(mus.shape[1]):
		for mk in markers[ct]:
			for ct2 in list(set(range(mus.shape[1])) - set([ct])):
				if score10_df.ix[mk,ct] >= score10_df.ix[mk, ct2]:
					markers[ct2] -= set([mk])
					
	return markers, mus

def prepare_heat_map(df: pd.DataFrame, cols_df: pd.DataFrame,
					rows_df: pd.DataFrame, marker_n: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
	'''
	Prepare all the inputs necessary to plot a marker heatmap
	
	Args
	----
	
	Return
	------
	
	'''
	
	# Reorganize the inputs
	logging.debug("Reorganizing the input")
	np.random.seed(15071990)
	cols_df.columns = cols_df.columns + np.random.randint(1000, 9999, size=cols_df.shape[1]).astype(str)
	cols_df.ix["Accession"] = cols_df.columns.values
	df.columns = cols_df.columns
	labels = cols_df.ix["Clusters"].values.astype(int)
	valid = cols_df.ix["_Valid"].values.astype(bool)
	df = df.ix[:,valid]
	labels = labels[valid]
	cols_df = cols_df.ix[:,valid]
	
	# Preare Table with top markers
	table, mus = marker_table(df, labels, marker_n)
	accession_selected = []  # type: List
	for i in table.values():
		accession_selected += list(i)
	mus_selected = mus[np.in1d(rows_df.columns.values, accession_selected),:]
	
	# Perform single linkage on correlation of the average pattern of the markers
	logging.debug("Sort the clusters by single linkage")
	z = linkage(mus_selected.T, 'single','correlation' )
	order = leaves_list(z)
	
	logging.debug("Preparing output")
	# Modify the values of the labels to respect the dendrogram order
	ixs = np.argsort(order, kind="mergesort")
	labels_updated = ixs[labels]  # reattribute the label on the basis of the linkage

	# Sort the cells based on the updated labels
	ix0 = np.argsort(labels_updated, kind="mergesort")
	labels_sorted = labels_updated[ix0]
	cols_df_sorted = cols_df.ix[:, ix0]
	df_sorted = df.ix[:, ix0]
	cols_df_sorted.ix["Total Molecules"] = df_sorted.sum(0).values

	# Generate a list of genes and gene cluster labels
	accession_list = []  # type: List
	gene_cluster = []  # type: List
	for i in order:
		accession_list += list(table[i])
		gene_cluster += [i]*len(table[i])
	gene_cluster = np.array(gene_cluster)
		
	rows_df_markers = rows_df.ix[:,accession_list]
	rows_df_markers.ix["Cluster"] = np.array(gene_cluster)
	
	return df_sorted.ix[accession_list, :], rows_df_markers, cols_df_sorted, accession_list, gene_cluster, mus


def generate_pcolor_args(attribute_values: np.ndarray, kind: str = "categorical", cmap: Any = None, custom_list: List = None) -> Tuple[np.ndarray, Any]:
	"""
	
	Args
	----
	
	attribute_values (np.ndarray) : values of the attrybute
	
	kind (str) : one of "categorical"(default), "continuous", "binary", "bool", "custom"
	
	cmap (mpl.color.Colormap) : colormap to be used. The default is 0.3*cm.prism + 0.7*cm.spectral
	
	custom_list (list, default None) : if kind=="custom" is the list of selected colors to attribute to the sorted
	attribute values
	
	Return
	------
	
	values (np.ndarray) : array ready to be passed as first argument to pcolorfast
	
	colormap (mpl.color.Colormap) : colormap ready to be passed as cmap argument to pcolorfast
	"""
	if kind == "categorical":
		attributes, _, attrs_ix = np.unique(attribute_values, return_index=True, return_inverse=True)
		n_attrs = len(attribute_values)
		if not cmap:
			def spectral_prism(x: np.ndarray) -> np.ndarray:
				return 0.3 * plt.cm.prism(x) + 0.7 * plt.cm.spectral(x)
			cmap = spectral_prism
		color_list = cmap(attrs_ix / np.max(attrs_ix))
		generated_cmap = matplotlib.colors.ListedColormap(color_list, name='from_list')
		values = np.arange(n_attrs)
	elif kind == "continuous":
		values = np.array( attribute_values )
		values -= np.min(values)
		values /= np.max(values)
		if cmap:
			generated_cmap = cmap
		else:
			generated_cmap = plt.cm.viridis
	elif kind == "bool":
		generated_cmap = plt.cm.gray
		values = attribute_values.astype(int)
	elif kind == "binary":
		generated_cmap = plt.cm.gray
		values = (attribute_values == attribute_values[0]).astype(int)
	elif kind == "custom":
		levels, ix = np.unique(attribute_values, return_inverse=True )
	else:
		raise NotImplementedError("kind '%s' is not supported" % kind)
		
	return values, generated_cmap

def calculate_intensities(df_markers: pd.DataFrame) -> pd.DataFrame:
	intensities = np.log2(df_markers + 1)
	intensities = intensities.sub(intensities.mean(1),axis="rows")
	return intensities.div(intensities.std(1),axis="rows")


def super_heatmap(intensities: pd.DataFrame,
                  cols_annot: pd.DataFrame,
				  rows_annot: pd.DataFrame,
				  col_attrs: List[Tuple] = [ ("SampleID",), ("Clusters", ), ("DonorID", ) ],
				  row_attrs: List[Tuple] = [ ("Cluster",)]) -> None:
	'''Plots an interactive and informative heat map
	
	Args
	----
	intensities: pd.DataFrame
	cols_annot: pd.DataFrame
	rows_annot: pd.DataFrame
	col_attrs: List[Tuple]
	row_attrs: List[Tuple]
	
	Returns
	-------
	Nothing, plots the heatmap
	
	'''
	e = 0.03
	h_col_bar = 0.019
	n_col_bars = 3
	w_row_bar = 0.03
	n_row_bars = 1
	delta_x = w_row_bar * n_row_bars
	delta_y = h_col_bar * n_col_bars

	fig = plt.figure(figsize=(12,9))
	left, bottom, width, height = delta_x + 3*e, 0 + e, 1 - delta_x - 4*e, 1 - delta_y - 2*e
	heatmap_bbox = [left, bottom, width, height]
	heatmap_ax = fig.add_axes(heatmap_bbox)
	heatmap_ax.pcolorfast(intensities.values, cmap=plt.cm.YlOrRd,\
		vmin=np.percentile(intensities,2.5), vmax=np.percentile(intensities,98.5))

	heatmap_ax.tick_params(axis='x', labeltop='off', labelbottom='off',bottom="off" )
	heatmap_ax.tick_params(axis='y', labelleft='off',left='off',right='off', labelsize=1)

	# Column bars
	for c, (col_name, *kind) in enumerate(col_attrs):
		if kind == []:
			if len(np.unique(cols_annot.ix[col_name].values)) > 2:
				kind = ("categorical",)
			else:
				kind = ("binary",)
		columnbar_bbox = [left , bottom + height + c*h_col_bar , width, h_col_bar]
		column_bar = fig.add_axes(columnbar_bbox, sharex=heatmap_ax)
		values, generated_cmap = generate_pcolor_args(cols_annot.ix[col_name].values, kind=kind[0])
		column_bar.pcolorfast(values[None,:], cmap=generated_cmap)
		column_bar.tick_params(axis='y', left='off', right='off', labelleft='off', labelright='off' )
		if col_name == "Clusters":
			column_bar.tick_params(axis='x', bottom='off', top='off', labelbottom='on', labeltop='off' )
			column_bar.tick_params(direction='out', pad=-9, colors='w') 
			bpos = np.where(np.diff(cols_annot.ix["Clusters"].values))[0]
			cpos = (np.r_[0, bpos[:-1]] + bpos) / 2.
			for b in bpos:
				heatmap_ax.axvline(b, linewidth=0.5, c="darkred", alpha=0.6)
			uq, ix = np.unique(cols_annot.ix["Clusters"].values, return_index=True)
			order_pos = uq[np.argsort(ix)]
			plt.xticks(cpos, order_pos,fontsize=7, ha="center", va="center")
			for t in column_bar.xaxis.get_major_ticks():
				t.label1.set_fontweight('bold')
		else:
			column_bar.tick_params(axis='x', bottom='off', top='off', labelbottom='off', labeltop='off' )
		plt.text(left-0.1*e, bottom + height + c*h_col_bar + 0.5*h_col_bar, col_name,
			ha='right', va='center', fontsize=7,transform = fig.transFigure)  

	# Row bars
	for r, (row_name, *kind) in enumerate(row_attrs):
		if kind == []:
			if len(np.unique(rows_annot.ix[row_name].values)) > 2:
				kind = ("categorical",)
			else:
				kind = ("binary",)
		rowbar_bbox = [left-w_row_bar, bottom, w_row_bar, height]
		row_bar = fig.add_axes(rowbar_bbox, sharey=heatmap_ax)
		values, generated_cmap = generate_pcolor_args(rows_annot.ix[row_name].values, kind=kind[0])
		row_bar.pcolorfast(values[:,None], cmap=generated_cmap)
		row_bar.tick_params(axis='both', bottom='off', top='off',
							right='on',left='off',labelright="off",labelleft="on",labelbottom='off',
							direction='in', labelsize=9, colors='k')

		names = rows_annot.ix["Gene", :].values.astype(str)

		class Y_Locator(ticker.MaxNLocator):
			def tick_values(self, vmin: float, vmax: float) -> np.ndarray:
				if vmin < 0:
					vmin = 0
				if vmax - vmin > 90:
					return []
				else:
					return np.arange(int(vmin), int(vmax + 1)) + 0.5

		def my_formatter(x: Any, pos: float = None) -> str:
			return names[int(x)]

		row_bar.yaxis.set_major_formatter(ticker.FuncFormatter(my_formatter))
		row_bar.yaxis.set_major_locator(Y_Locator())
		
	fig.canvas.draw()