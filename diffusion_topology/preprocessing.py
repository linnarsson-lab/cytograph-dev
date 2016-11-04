
import os
import logging
import numpy as np
import loompy

def make_doublets(ds, cell_id_template):
	n = int(ds.shape[1]*0.05)
	cells = np.fromiter(range(ds.shape[1]), dtype='int')

	# Create doublets
	doublets = []
	ids = []
	for ix in range(n):
		pair = np.random.choice(cells, 2, replace=False)
		dbl = ds[:, pair[0]] + ds[:, pair[1]]
		doublets.append(dbl)
		ids.append(cell_id_template + "_doublet_" + str(ix))
	doublets = np.column_stack(doublets)

	# Create dummy column attributes
	dbl_attrs = {}
	for a in ds.col_attrs.keys():
		if a == "CellID":
			dbl_attrs[a] = ids
		else:
			dtype = ds.schema["col_attrs"][a]
			if dtype == "float64":
				dbl_attrs[a] = np.zeros(n, dtype='float64')
			if dtype == "string":
				dbl_attrs[a] = np.zeros(n, dtype='unicode')
			if dtype == "int":
				dbl_attrs[a] = np.zeros(n, dtype='int')
	ds.add_columns(doublets, dbl_attrs)
	dblts = np.zeros(ds.shape[1])
	dblts[-n:] = 1
	ds.set_attr("_FakeDoublet", dblts, dtype='int',axis=1)

def validate_cells(ds):
	"""
	few more things,
	before sending the data to the clustering i usually filter out cells like that:

	validcells = (tot_mol>600 & (tot_mol./tot_genes) > 1.2 & tot_mol < 20000 & tot_genes > 500);
	then i remove doublets based on markers for neurons, oligo,endo,microglia, astro ect.
	less relevant maybe to development but can be good to check it.

	for genes, first filter
	in = find(sum(data>0,2)>20 & sum(data>0,2)<length(data(1,:))*0.6);

	then using cv vs mean select 10,000 genes (maybe should be higher)
	remove sex genes
	shuffle the columns
	normalize the total number of molecules to 10,000. i found that this give better results.

	amit
	"""
	(mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
	valid = np.logical_and(np.logical_and(mols > 600, (mols/genes) > 1.2), np.logical_and(mols < 20000, genes > 500)).astype('int')
	ds.set_attr("_Valid", valid, dtype='int', axis=1)

def validate_genes(ds):
	nnz = ds.map(np.count_nonzero, axis=0)
	valid = np.logical_and(nnz > 20, nnz < ds.shape[1]*0.6)
	ds.set_attr("_Valid", valid, dtype='int', axis=0)

def preprocess(loom_folder, sample_ids, out_file, make_doublets=False):
	for sample_id in sample_ids:
		logging.info("Preprocessing " + sample_id)
		ds = loompy.connect(os.path.join(loom_folder, sample_id + ".loom"))
		if make_doublets and not "_FakeDoublet" in ds.col_attrs:
			logging.info("Making fake doublets")
			make_doublets(ds, sample_id)
		logging.info("Marking invalid cells")
		validate_cells(ds)
		ds.close()
	logging.info("Creating joint loom file")
	loompy.combine([os.path.join(loom_folder, s + ".loom") for s in sample_ids], out_file)
	logging.info("Marking invalid genes")
	ds = loompy.connect(out_file)
	validate_genes(ds)
	ds.close()
