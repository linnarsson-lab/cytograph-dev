import loompy
import numpy as np
import os
import io
import pandas as pd
from scipy.io import mmread
import logging
import sys
import gzip


def read_vcf(path, compression: str = None):
    if compression is None:
        with open(path, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]
    if compression == 'gzip':
        with gzip.open(path, 'r') as f:
            lines = [l.decode('UTF-8') for l in f if not l.decode('UTF-8').startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})


def genotype_sample(ds: loompy.LoomConnection, config_path, threshold) -> None:

    # sample and donor names
    sample = ds.ca.CellID[0].split(':')[0]
    donor = ds.ca.Donor[0]

    # open donor_vcf
    donor_vcf = read_vcf(
        os.path.join(config_path, donor + '.final.vcf.gz'),
        compression='gzip'
        )
    donor_vcf['HBAgenomics'] = donor_vcf['HBAgenomics'].str.split(':').str[0]

    logging.info(f"Genotyping {sample} cells against {donor}")

    # open and preprocess files
    cellsnp_out = os.path.join(config_path, "cellSNP", donor, sample)

    if not os.path.exists(cellsnp_out):
        logging.error(f"cellsnp output does not exist for {sample}")
        sys.exit(1)

    cells = np.loadtxt(cellsnp_out + '/cellSNP.samples.tsv', dtype='str')
    ad = mmread(cellsnp_out + '/cellSNP.tag.AD.mtx')
    dp = mmread(cellsnp_out + '/cellSNP.tag.DP.mtx')

    # filter for homozygous positions
    counted = read_vcf(cellsnp_out + '/cellSNP.base.vcf')
    counted_genome = counted.merge(donor_vcf,  how='inner', left_on=["CHROM", "POS"], right_on = ["CHROM", "POS"])
    hom = counted_genome['HBAgenomics'] == '1/1'
    logging.info(f"{hom.sum()} homozygous alternate positions")

    ad = ad.tocsr()[hom].A
    dp = dp.tocsr()[hom].A
    f = ad / dp

    ## TO DO: CHECK IF THIS KEEPS DUPLICATE POSITIONS

    # calc homozygous alt depth and % ref
    # find corresponding cell ID in cellSNP output
    cell_attr = np.array([x.split(':')[1].replace('x', '-1') for x in ds.ca.CellID])
    ix = np.array([np.where(x == cells)[0][0] for x in cell_attr])
    # count genotype positions
    depth = np.count_nonzero(dp[:, ix], axis=0)
    # count fewer than threshold of reads is alt
    f = ad[:, ix] / dp[:, ix]
    ref = np.count_nonzero(f <= threshold, axis=0)

    ds.ca[f'Depth'] = depth
    ds.ca[f'Ref'] = ref

    return
