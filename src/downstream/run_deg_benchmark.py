import os
import scanpy as sc
import pandas as pd
import gseapy as gp
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
CLUSTERS_CSV = '../gsea_local_sub_v2/leiden_clusters.csv'
GMT_PATH = './data/h.all.v2024.1.Hs.symbols.gmt'
OUTPUT_DIR = '../gsea_local_sub_v2/benchmark_deg/'

# Under the revision3 pipeline (unified labels + scMeta-graphloss + real
# inductive saliency), cluster 56 is the top EMT-enriched cluster -- not 59
# (see run_gsea_local_subpop_v2.log). Update this if run_gsea.py is rerun
# and the ranking changes again.
TARGET_CLUSTER = '56'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading AnnData (Backed mode)...")
    ad_full = sc.read_h5ad(DATA_PATH, backed='r')
    recompute_metastasis_label(ad_full)

    # Extract only Malignant Primary and Local Mets cells (matches the
    # population run_gsea.py clustered)
    valid_mask = ((ad_full.obs["Final_cell_type"] == "Malignant") &
                  ((ad_full.obs["metastasis_label"] == "No_Mets") | (ad_full.obs["metastasis_label"] == "Regional_Mets")))
    ad = ad_full[valid_mask].to_memory()
    ad_full.file.close()

    print("Loading SCMETA Leiden clusters...")
    clusters_df = pd.read_csv(CLUSTERS_CSV, index_col=0)
    ad.obs['leiden'] = pd.Series(index=ad.obs_names, dtype='object')
    ad.obs.loc[clusters_df.index, 'leiden'] = clusters_df['leiden'].astype(str)

    # Create a new column for the Wilcoxon comparison: target cluster vs Primary
    # Primary cells are "No_Mets". Target cluster cells are the active EMT local mets.
    ad.obs['benchmark_group'] = 'Other'
    ad.obs.loc[ad.obs['metastasis_label'] == 'No_Mets', 'benchmark_group'] = 'Primary'
    ad.obs.loc[(ad.obs['metastasis_label'] == 'Regional_Mets') & (ad.obs['leiden'] == TARGET_CLUSTER), 'benchmark_group'] = 'Cluster_target'

    # Filter strictly to these two groups
    ad_test = ad[ad.obs['benchmark_group'].isin(['Primary', 'Cluster_target'])].copy()

    print(f"Running Wilcoxon DEG: Cluster {TARGET_CLUSTER} (n={(ad_test.obs['benchmark_group'] == 'Cluster_target').sum()}) vs Primary (n={(ad_test.obs['benchmark_group'] == 'Primary').sum()})...")

    # Run traditional DEG
    sc.tl.rank_genes_groups(ad_test, groupby='benchmark_group', groups=['Cluster_target'], reference='Primary', method='wilcoxon')

    # Extract the DEG scores (test statistics) to rank genes
    deg_results = sc.get.rank_genes_groups_df(ad_test, group='Cluster_target')

    # Format for GSEApy (Gene, Score)
    rnk_df = deg_results[['names', 'scores']].rename(columns={'names': 'Gene', 'scores': 'Score'})
    rnk_df = rnk_df.dropna().sort_values(by='Score', ascending=False)

    print("Running GSEA on Wilcoxon rankings...")
    out_path = os.path.join(OUTPUT_DIR, f'deg_cluster{TARGET_CLUSTER}_vs_primary')
    gp.prerank(rnk=rnk_df, gene_sets=GMT_PATH, outdir=out_path, min_size=5, max_size=1000, seed=42)

    res = pd.read_csv(os.path.join(out_path, 'gseapy.gene_set.prerank.report.csv'))

    print("\n" + "="*50)
    print(f"TRADITIONAL DEG BENCHMARK RESULTS (CLUSTER {TARGET_CLUSTER})")
    print("="*50)
    print("TOP 5 PATHWAYS (WILCOXON):")
    print(res[['Term', 'NES', 'FDR q-val']].head(5))
    
    print("\nEMT PATHWAY RESULT (WILCOXON):")
    emt_row = res[res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False, na=False)]
    if not emt_row.empty:
        print(emt_row[['Term', 'NES', 'FDR q-val']])
    else:
        print("EMT not found in significant results.")


if __name__ == "__main__":
    main()

