import matplotlib.pyplot as plt
import loompy


def genotype_hist(ds: loompy.LoomConnection, out_file: str, threshold: 0) -> None:

	# plot only cells with a threshold of homozygous alt positions
	cells = ds.ca.Depth > threshold

	# histogram: genotyped alt positions
	plt.figure(figsize=(8, 4))
	plt.subplot(121)
	plt.hist(ds.ca.Depth[cells], bins=100, alpha=0.5)
	plt.ylabel("Number of cells")
	plt.xlabel("Number of homozygous alt positions / cell")

	# histogram: fraction of alt positions that are not alt
	# "not alt" can mean different things based on genotyping procedure
	plt.subplot(122)
	plt.hist(ds.ca.Ref / ds.ca.Depth, bins=100, alpha=0.5)
	plt.ylabel("Number of cells")
	plt.xlabel("Fraction of alt positions that are not alt / cell")

	plt.tight_layout()
	plt.savefig(out_file, dpi=144, bbox_inches='tight')