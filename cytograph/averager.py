from math import exp, lgamma, log
import logging
import os
from typing import *
import numpy as np
import loompy
import numpy_groupies as npg


def aggregate_loom(ds: loompy.LoomConnection, out_file: str, select: np.ndarray, group_by: str, aggr_by: str, aggr_ca_by: Dict[str, str]) -> None:
	"""
	Aggregate a Loom file by applying aggregation functions to the main matrix as well as to the column attributes

	Args:
		ds			The Loom file
		out_file	The name of the output Loom file (will be appended to if it exists)
		select		Bool array giving the columns to include (or None, to include all)
		group_by	The column attribute to group by
		aggr_by 	The aggregation function for the main matrix
		aggr_ca_by	The aggregation functions for the column attributes

	Remarks:
		aggr_by gives the aggregation function for the main matrix
		aggr_ca_by is a dictionary with column attributes as keys and aggregation functionas as values
		
		Aggregation functions can be any valid aggregation function from here: https://github.com/ml31415/numpy-groupies

		In addition, you can specify:
			"tally" to count the number of occurences of each value of a categorical attribute
			"geom" to calculate the geometric mean
	"""
	ca = {}  # type: Dict[str, np.ndarray]
	if select is not None:
		cols = np.where(select)[0]
	else:
		cols = np.fromiter(range(ds.shape[1]))
	labels = ds.col_attrs[group_by][cols]
	n_groups = len(set(labels))
	for key in ds.col_attrs.keys():
		if key not in aggr_ca_by:
			continue
		func = aggr_ca_by[key]
		if func == "tally":
			for val in set(ds.col_attrs[key]):
				ca[key + "_" + val] = npg.aggregate_numba.aggregate(labels, ds.col_attrs[key][cols] == val, func="sum")
		elif func == "geom":
			ca[key] = np.exp(npg.aggregate_numba.aggregate(labels, np.log(ds.col_attrs[key][cols]), func="mean"))
		else:
			ca[key] = npg.aggregate_numba.aggregate(labels, ds.col_attrs[key][cols], func=func)
	m = np.empty((ds.shape[0], n_groups))
	for (ix, selection, vals) in ds.batch_scan(cells=cols, genes=None, axis=0):
		if aggr_by == "geom":
			vals_aggr = np.exp(npg.aggregate_numba.aggregate(labels, np.log(vals), func="mean", axis=1))
		else:
			vals_aggr = npg.aggregate_numba.aggregate(labels, vals, func=aggr_by, axis=1)
		m[selection, :] = vals_aggr
	if os.path.exists(out_file):
		dsout = loompy.connect(out_file)
		dsout.add_columns(m, ca)
		dsout.close()
	else:
		loompy.create(out_file, m, ds.row_attrs, ca)


class Averager:
	def __init__(self, func: str = "mean") -> None:
		self.func = func

	def calculate_and_save(self, ds: loompy.LoomConnection, output_file: str, category_summary: Tuple = ("Age", "SampleID")) -> None:
		cells = np.where(ds.col_attrs["Clusters"] >= 0)[0]
		labels = ds.col_attrs["Clusters"][cells]
		Nclust = np.max(labels) + 1
		ca = {"Cluster": np.arange(Nclust), "OriginalFile": np.array([output_file] * Nclust)}
		ra = {"Accession": ds.row_attrs["Accession"], "Gene": ds.row_attrs["Gene"]}
		for category_class in category_summary:
			for unique_element in set(ds.col_attrs[category_class]):
				ca["%s_%s" % (category_class, unique_element)] = npg.aggregate_numba.aggregate(labels, ds.col_attrs[category_class][cells] == unique_element, func="sum")
		m = np.empty((ds.shape[0], Nclust))
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
			vals_avg = npg.aggregate_numba.aggregate(labels, vals, func=self.func, axis=1)
			m[selection, :] = vals_avg
		dsout = loompy.create(output_file, m, ra, ca)