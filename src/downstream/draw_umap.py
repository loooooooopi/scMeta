import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label

sc.set_figure_params(vector_friendly=True, dpi_save=300)

def main():
	# ==========================================
	# 1. FILE PATHS
	# ==========================================
	data_path = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
	clusters_csv = '../gsea_local_sub_v2/leiden_clusters.csv' # Path to the clusters we saved earlier
	output_plot = '../gsea_local_sub_v2/umap_clean_with_EMT_highlight.pdf'

	# ==========================================
	# 2. LOAD DATA (LAZY LOADING)
	# ==========================================
	print("Opening AnnData in backed mode (lazy loading)...")
	# 'backed='r'' prevents loading the massive feature matrix into RAM
	ad_full = sc.read_h5ad(data_path, backed='r')
	recompute_metastasis_label(ad_full)

	print("Filtering for Malignant Primary and Local Mets cells (matches the population run_gsea.py clustered)...")
	valid_mask = ((ad_full.obs["Final_cell_type"] == "Malignant") &
	              ((ad_full.obs["metastasis_label"] == "No_Mets") | (ad_full.obs["metastasis_label"] == "Regional_Mets")))

	print("Extracting only the necessary cells into memory...")
	# .to_memory() pulls ONLY the filtered subset from the disk into RAM
	ad = ad_full[valid_mask].to_memory()

	# Safely close the background file reader
	ad_full.file.close()
	import gc
	gc.collect()

	# Load the Leiden clusters from the CSV we saved. Index-aligned (not
	# positional) so a mismatch in cell count/order raises missing values
	# instead of silently mislabeling cells.
	print("Loading Leiden clusters...")
	clusters_df = pd.read_csv(clusters_csv, index_col=0)
	ad.obs['leiden'] = pd.Series(index=ad.obs_names, dtype='object')
	ad.obs.loc[clusters_df.index, 'leiden'] = clusters_df['leiden'].astype(str)
	n_missing = ad.obs['leiden'].isna().sum()
	if n_missing:
		print(f"WARNING: {n_missing}/{ad.n_obs} cells have no matching leiden cluster (index mismatch); dropping them.")
		ad = ad[ad.obs['leiden'].notna()].copy()
	ad.obs['leiden'] = ad.obs['leiden'].astype('category')

	# ==========================================
	# 3. CREATE EMT HIGHLIGHT COLUMN
	# ==========================================
	print("Creating EMT highlight annotations...")
	# Initialize all as gray background
	ad.obs['EMT_Status'] = 'Other Clusters'

	# Under the revision3 pipeline (unified labels + scMeta-graphloss + real
	# inductive saliency), cluster 56 is the top EMT-enriched cluster
	# (NES=1.356, FDR=0.172 -- EMT ranks #2 overall in that cluster, not #1;
	# see run_gsea_local_subpop_v2.log). There is no longer a comparably
	# strong second cluster the way the old pipeline had clusters 59 and 4,
	# so only one cluster is highlighted here.
	ad.obs.loc[ad.obs['leiden'] == '56', 'EMT_Status'] = 'Cluster 56 (EMT High)'

	# Convert to category so Scanpy plots it properly
	ad.obs['EMT_Status'] = ad.obs['EMT_Status'].astype('category')

	# Define custom colors (Gray for others, bright distinct color for the EMT cluster)
	emt_palette = {
		'Other Clusters': '#E0E0E0',        # Light Gray
		'Cluster 56 (EMT High)': '#E41A1C', # Bright Red
	}

	# ==========================================
	# 4. PLOTTING
	# ==========================================
	print("Generating plot...")
	# Create a very wide figure to give legends plenty of room to breathe
	fig, axes = plt.subplots(1, 3, figsize=(24, 6))

	# Plot 1: Leiden Clusters
	# We use legend_loc='on data' to put the numbers directly on the clusters, 
	# or 'right margin' if you want a list. Since there are 68 clusters, a list is huge.
	sc.pl.umap(
		ad, 
		color='leiden', 
		ax=axes[0], 
		show=False, 
		legend_loc='right margin', # Move legend outside
		title='Leiden Clusters'
	)

	# Plot 2: Cancer Type
	sc.pl.umap(
		ad, 
		color='Final_cancer_type', 
		ax=axes[1], 
		show=False, 
		legend_loc='right margin',
		title='Primary Cancer Type'
	)

	# Plot 3: EMT Highlight
	# In Scanpy, to ensure the gray points are plotted first (in the background) 
	# and the red/blue points are plotted on top, we sort the order
	sc.pl.umap(
		ad, 
		color='EMT_Status', 
		palette=emt_palette,
		ax=axes[2], 
		show=False, 
		legend_loc='right margin',
		title='Transient EMT Subpopulations',
		sort_order=True # Crucial: plots gray first, highlights on top
	)

	# Adjust spacing between plots so legends don't overlap the next axis
	plt.subplots_adjust(wspace=0.6)

	# Save the figure
	plt.savefig(output_plot, bbox_inches='tight', dpi=300)
	print(f"Plot successfully saved to: {output_plot}")


if __name__ == "__main__":
	main()

