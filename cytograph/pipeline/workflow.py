import os
from typing import *
import numpy as np
import loompy
import luigi
from pathlib import Path
import logging
import math
import random
import sys
import pandas as pd
import click
import cytograph.plotting as cgplot
from cytograph.annotation import AutoAutoAnnotator, AutoAnnotator, CellCycleAnnotator
from cytograph.clustering import ClusterValidator
from cytograph.species import Species
from cytograph.preprocessing import Scrublet, doublet_finder
from .config import config
from .cytograph import Cytograph
from .punchcards import Punchcard, PunchcardSubset, PunchcardDeck
from .aggregator import Aggregator
from .utils import Tempname

#
# Overview of the cytograph2 pipeline
#
# sample1 ----\                /--> First_First.loom --------------------------------------|
# sample2 -----> First.loom --<                                                            |
# sample3 ----/                \--> First_Second.loom -------------------------------------|
#                                                                                          |
#                                                                                           >---> Pool.loom
#                                                                                          |
# sample4 ----\                 /--> Second_First.loom ------------------------------------|
# sample5 -----> Second.loom --<                                                           |
# sample6 ----/                 \                          /--> Second_Second_First.loom --|
#                               \--> Second_Second.loom --<                                |
#                                                          \--> Second_Second_Second.loom -|
#


def pcw(age: str) -> float:
	"""
	Parse age strings in several formats
		"8w 5d" -> 8 weeks 5 days -> 8.71 weeks
		"8.5w" -> 8 weeks 5 days -> 8.71 weeks
		"CRL44" -> 9.16 weeks

	CRL formula
	Postconception weeks = (CRL x 1.037)^0.5 x 8.052 + 23.73
	"""
	age = str(age).lower()
	age = age.strip()
	if age.startswith("crl"):
		crl = float(age[3:])
		return (math.sqrt((crl * 1.037)) * 8.052 + 21.73) / 7
	elif " " in age:
		w, d = age.split(" ")
		if not w.endswith("w"):
			raise ValueError("Invalid age string: " + age)
		if not d.endswith("d"):
			raise ValueError("Invalid age string: " + age)
		return int(w[:-1]) + int(d[:-1]) / 7
	else:
		if not age.endswith("w"):
			raise ValueError("Invalid age string: " + age)
		age = age[:-1]
		if "." in age:
			w, d = age.split(".")
		else:
			w = age
			d = "0"
		return int(w) + int(d) / 7


# TODO: turn this into a proper class and make the API nicer
def get_metadata_for(sample: str) -> Dict:
	if config.paths.metadata is None or not os.path.exists(config.paths.metadata):
		return {}
	sid = "SampleID"
	metadata_file = config.paths.metadata
	if os.path.exists(metadata_file):
		# Special handling of typed column names for our database
		with open(metadata_file) as f:
			line = f.readline()
			if "SampleID:string" in line:
				sid = "SampleID:string"
		try:
			metadata = pd.read_csv(metadata_file, delimiter=";", index_col=sid, engine="python")
			attrs = metadata.loc[sample]
			if sid == "SampleID:string":
				return {k.split(":")[0]: v for k, v in metadata.loc[sample].items()}
			else:
				return {k: v for k, v in metadata.loc[sample].items()}
		except Exception as e:
			logging.info(f"Failed to load metadata because: {e}")
			raise e
	else:
		return {}


def compute_subsets(card: Punchcard) -> None:
	logging.info(f"Computing subset assignments for {card.name}")
	with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".loom"), mode="r+") as ds:
		subset_per_cell = np.zeros(ds.shape[1], dtype=object)
		taken = np.zeros(ds.shape[1], dtype=bool)
		with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".agg.loom"), mode="r") as dsagg:
			for subset in card.subsets.values():
				selected = np.zeros(ds.shape[1], dtype=bool)
				if len(subset.include) > 0:
					# Include clusters that have any of the given auto-annotations
					for aa in subset.include:
						for ix in range(dsagg.shape[1]):
							if aa in dsagg.ca.AutoAnnotation[ix].split(" "):
								selected = selected | (ds.ca.Clusters == ix)
				else:
					selected = ~taken
				# Exclude cells that don't match the onlyif expression
				if subset.onlyif != "" and subset.onlyif is not None:
					selected = selected & eval(subset.onlyif, globals(), ds.ca)
				# Don't include cells that were already taken
				selected = selected & ~taken
				subset_per_cell[selected] = subset.name
		ds.ca.Subset = subset_per_cell


def aggregate_and_export(loom_file: str, agg_file: str, export_dir: str) -> None:
	# STEP 2: aggregate and create the .agg.loom file
	if os.path.exists(agg_file):
		logging.info(f"Skipping '{agg_file}' because it was already complete.")
	else:
		with loompy.connect(loom_file) as dsout:
			Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, agg_file=agg_file)

	# STEP 3: export plots
	if os.path.exists(export_dir):
		logging.info(f"Skipping '{export_dir}' because it was already complete.")
	else:
		pool = os.path.split(export_dir)[1]
		logging.info(f"Exporting plots for {pool}")
		with Tempname(export_dir) as out_dir:
			os.mkdir(out_dir)
			with loompy.connect(loom_file) as ds:
				with loompy.connect(agg_file) as dsagg:
					cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.aa.png"), list(dsagg.ca.AutoAnnotation))
					cgplot.manifold(ds, os.path.join(out_dir, pool + "_TSNE_manifold.aaa.png"), list(dsagg.ca.MarkerGenes))
					cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.aaa.png"), list(dsagg.ca.MarkerGenes), embedding="UMAP")
					cgplot.markerheatmap(ds, dsagg, n_markers_per_cluster=10, out_file=os.path.join(out_dir, pool + "_heatmap.pdf"))
					cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
					cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
					cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
					cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
					cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
					cgplot.embedded_velocity(ds, out_file=os.path.join(out_dir, f"{pool}_velocity.png"))
					cgplot.TFs(ds, dsagg, out_file_root=os.path.join(out_dir, pool))
					if "cluster-validation" in config.steps:
						ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))


def process_root(deck: PunchcardDeck, subset: PunchcardSubset) -> None:
	# Collect directly from samples, optionally with doublet removal and min_umis etc.
	# Specification is a nested list giving batches and replicates
	# include: [[sample1, sample2], [sample3, sample4]]

	# STEP 1: build the .loom file and perform manifold learning (Cytograph)
	# Maybe we're already done?
	loom_file = os.path.join(config.paths.build, "data", subset.longname() + ".loom")
	if os.path.exists(loom_file):
		logging.info(f"Skipping '{subset.longname()}.loom' because it was already complete.")
	else:
		# Make sure all the sample files exist
		err = False
		for batch in subset.include:
			for sample_id in batch:
				full_path = os.path.join(config.paths.samples, sample_id + ".loom")
				if not os.path.exists(full_path):
					logging.error(f"Sample file '{full_path}' not found")
					err = True
		if err:
			sys.exit(1)

		with Tempname(loom_file) as out_file:
			logging.info(f"Collecting cells for {subset.longname()}")
			logging.debug(out_file)
			with loompy.new(out_file) as dsout:
				batch_id = 0
				for batch in subset.include:
					replicate_id = 0
					for sample_id in batch:
						full_path = os.path.join(config.paths.samples, sample_id + ".loom")
						logging.info(f"Adding {sample_id}.loom")
						with loompy.connect(full_path) as ds:
							species = Species.detect(ds).name
							col_attrs = dict(ds.ca)
							metadata = get_metadata_for(sample_id)
							for key, val in metadata.items():
								col_attrs[key] = np.array([val] * ds.shape[1])
							col_attrs["SampleID"] = np.array([sample_id] * ds.shape[1])
							col_attrs["Batch"] = np.array([batch_id] * ds.shape[1])
							col_attrs["Replicate"] = np.array([replicate_id] * ds.shape[1])
							if "Age" in metadata and species == "Homo sapiens":
								try:
									col_attrs["PCW"] = np.array([pcw(metadata["Age"])] * ds.shape[1])
								except:
									pass
							logging.info("Scoring doublets using Scrublet")
							data = ds[:, :].T
							doublet_scores, predicted_doublets = Scrublet(data, expected_doublet_rate=0.05).scrub_doublets()
							col_attrs["ScrubletScore"] = doublet_scores
							col_attrs["ScrubletFlag"] = predicted_doublets.astype("int")
							logging.info("Scoring doublets using DoubletFinder")
							col_attrs["DoubletFinderScore"] = doublet_finder(ds)
							logging.info(f"Computing total UMIs")
							(totals, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
							col_attrs["TotalUMI"] = totals
							col_attrs["NGenes"] = genes
							good_cells = (totals >= config.params.min_umis)
							if config.params.doublets_action == "remove":
								logging.info(f"Removing {predicted_doublets.sum()} doublets and {(~good_cells).sum()} cells with <{config.params.min_umis} UMIs")
								good_cells = good_cells & (~predicted_doublets)
							logging.info(f"Collecting {good_cells.sum()} of {data.shape[0]} cells")
							dsout.add_columns(ds.layers[:, good_cells], {att: vals[good_cells] for att, vals in col_attrs.items()}, row_attrs=ds.row_attrs)
						replicate_id += 1
					batch_id += 1
				Cytograph(steps=config.steps).fit(dsout)
	agg_file = os.path.join(config.paths.build, "data", subset.longname() + ".agg.loom")
	export_dir = os.path.join(config.paths.build, "exported", subset.longname())
	aggregate_and_export(loom_file, agg_file, export_dir)
	# If there's a punchcard for this subset, go ahead and compute the subsets for that card
	card_for_subset = deck.get_card(subset.longname())
	if card_for_subset is not None:
		compute_subsets(card_for_subset)
	logging.info("Done.")


def process_subset(deck: PunchcardDeck, subset: PunchcardSubset) -> None:
	# STEP 1: cytograph
	# Maybe we're already done?
	loom_file = os.path.join(config.paths.build, "data", subset.longname() + ".loom")
	if os.path.exists(loom_file):
		logging.info(f"Skipping {subset.longname()}.loom because it was already complete.")
	else:
		# Verify that the previous punchard subset exists
		parent = os.path.join(config.paths.build, "data", subset.card.name + ".loom")
		if not os.path.exists(parent):
			logging.error(f"Punchcard file '{parent}' was missing.")
			sys.exit(1)

		# Verify that there are some cells in the subset
		with loompy.connect(parent, mode="r") as ds:
			if (ds.ca.Subset == subset.name).sum() == 0:
				logging.info(f"Skipping {subset.longname()} because the subset was empty")
				sys.exit(0)

		with Tempname(loom_file) as out_file:
			logging.info(f"Collecting cells for {subset.longname()}")
			with loompy.new(out_file) as dsout:
				# Collect from a previous punchard subset
				with loompy.connect(parent, mode="r") as ds:
					for (ix, selection, view) in ds.scan(items=(ds.ca.Subset == subset.name), axis=1, key="Accession"):
						dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
				logging.info(f"Collected {ds.shape[1]} cells")
				Cytograph(steps=config.steps).fit(dsout)
	agg_file = os.path.join(config.paths.build, "data", subset.longname() + ".agg.loom")
	export_dir = os.path.join(config.paths.build, "exported", subset.longname())
	aggregate_and_export(loom_file, agg_file, export_dir)
	# If there's a punchcard for this subset, go ahead and compute the subsets for that card
	card_for_subset = deck.get_card(subset.longname())
	if card_for_subset is not None:
		compute_subsets(card_for_subset)
	logging.info("Done.")


def pool_leaves(deck: PunchcardDeck) -> None:
	loom_file = os.path.join(config.paths.build, "data", "Pool.loom")
	# Maybe we're already done?
	if os.path.exists(loom_file):
		logging.info(f"Skipping 'Pool.loom' because it was already done.")
	else:
		with Tempname(os.path.join(config.paths.build, "data", "Pool.loom")) as out_file:
			logging.info(f"Collecting cells for 'Pool.loom'")
			punchcards: List[str] = []
			clusters: List[int] = []
			punchcard_clusters: List[int] = []
			next_cluster = 0

			# Check that all the inputs exist
			err = False
			for subset in deck.get_leaves():
				if not os.path.exists(os.path.join(config.paths.build, "data", subset.longname() + ".loom")):
					logging.error(f"Punchcard file 'data/{subset.longname()}.loom' is missing")
					err = True
			if err:
				sys.exit(1)

			with loompy.new(out_file) as dsout:
				for subset in deck.get_leaves():
					with loompy.connect(os.path.join(config.paths.build, "data", subset.longname() + ".loom"), mode="r") as ds:
						punchcards = punchcards + [subset.longname()] * ds.shape[1]
						punchcard_clusters = punchcard_clusters + list(ds.ca.Clusters)
						clusters = clusters + list(ds.ca.Clusters + next_cluster)
						next_cluster = max(clusters) + 1
						for (ix, selection, view) in ds.scan(axis=1, key="Accession"):
							dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
				ds.ca.Punchcard = punchcards
				ds.ca.PunchcardClusters = punchcard_clusters
				ds.ca.Clusters = clusters
				Cytograph(steps=["nn", "embeddings", "aggregate", "export"]).fit(dsout)
	agg_file = os.path.join(config.paths.build, "data", "Pool.agg.loom")
	export_dir = os.path.join(config.paths.build, "exported", "Pool")
	aggregate_and_export(loom_file, agg_file, export_dir)


def create_build_folders(path: str) -> None:
	Path(os.path.join(path, "data")).mkdir(exist_ok=True)
	Path(os.path.join(path, "exported")).mkdir(exist_ok=True)
