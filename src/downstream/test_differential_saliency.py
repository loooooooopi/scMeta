"""
One clean test of a specific, pre-registered hypothesis: main.tex's
Methods (Sec. feature_pro, Eq. delta_g = g_bar_class2 - g_bar_class1)
documents a DIFFERENTIAL gradient signal, but the actual saliency code used
throughout this revision (inductive_saliency.compute_inductive_saliency /
_multi_threshold) only computes one side of that (mean |grad| for
confidently-predicted target-class cells alone). A gene with uniformly
large gradient magnitude regardless of class (e.g. a housekeeping gene)
inflates the one-sided ranking but cancels out in the documented
differential version.

This script implements the documented differential signal
(compute_differential_saliency, added to inductive_saliency.py) and
re-tests EMT enrichment on the two clusters where the one-sided ranking
failed to reach significance (Cluster 56, Cluster 58; see
Master_EMT_Benchmark_Summary.csv), using the exact same Cluster vs.
1:1-downsampled-Primary comparison groups as that benchmark. This is a
single, principled implementation fix decided BEFORE looking at the
outcome -- not a search over methods/thresholds until something reaches
significance. Reports the true outcome either way.
"""
import os
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import gseapy as gp
import gc
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label
from scMeta import scMeta
from inductive_saliency import build_malignant_graph, compute_differential_saliency

DATA_PATH = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
GMT_PATH = './data/h.all.v2024.1.Hs.symbols.gmt'
CLUSTERS_CSV = '../gsea_local_sub_v2/leiden_clusters.csv'
MODEL_DIR = './v2b_scMeta_models/'
OUTPUT_DIR = '../differential_saliency_v2/'
TARGET_CLUSTERS = ['56', '58']
HIDDEN_DIM = 256
NUM_CLASSES = 3
CONV_TYPE = 'TransformerConv'
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading atlas...")
    ad_full = sc.read_h5ad(DATA_PATH, backed='r')
    recompute_metastasis_label(ad_full)
    malignant_mask = (ad_full.obs["Final_cell_type"] == "Malignant").values
    ad = ad_full[malignant_mask].to_memory()
    ad_full.file.close()
    gc.collect()
    print(f"Malignant population (graph context): {ad.n_obs} cells")

    target_genes = []
    with open(GMT_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))
    valid_genes = [g for g in target_genes if g in ad.var_names]
    print(f"Feature space: {len(valid_genes)} genes")

    print("Building malignant-cell graph...")
    graph_data = build_malignant_graph(ad, valid_genes)

    clusters_df = pd.read_csv(CLUSTERS_CSV, index_col=0)
    ad.obs['leiden'] = pd.Series(index=ad.obs_names, dtype='object')
    ad.obs.loc[clusters_df.index, 'leiden'] = clusters_df['leiden'].astype(str)

    primary_idx_all = np.where((ad.obs["metastasis_label"] == "No_Mets").values)[0]
    rng = np.random.default_rng(SEED)

    for cluster_id in TARGET_CLUSTERS:
        cluster_mask = ((ad.obs["metastasis_label"] == "Regional_Mets").values &
                         (ad.obs["leiden"] == cluster_id).values)
        class2_idx = np.where(cluster_mask)[0]  # target: Cluster {cluster_id}
        n = len(class2_idx)
        print(f"\n{'='*60}\nCluster {cluster_id}: {n} cells\n{'='*60}")
        if n == 0:
            print("  no cells, skipping")
            continue
        class1_idx = rng.choice(primary_idx_all, size=n, replace=False)  # 1:1 downsampled Primary

        delta_g_sum = np.zeros(len(valid_genes), dtype=np.float64)
        for fold in range(1, 6):
            model_path = os.path.join(MODEL_DIR, f"5foldCV_fold{fold}_scMeta.pt")
            model = scMeta(input_dim=len(valid_genes), hidden_dim=HIDDEN_DIM,
                            num_classes=NUM_CLASSES, conv_type=CONV_TYPE).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()

            delta_g = compute_differential_saliency(
                model, graph_data, class1_idx, class2_idx, target_class=1,
                device=device, batch_size=512)
            delta_g_sum += delta_g
            print(f"  fold {fold} done")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        delta_g_avg = delta_g_sum / 5.0
        rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': delta_g_avg}).sort_values(
            by='Score', ascending=False)
        rnk_path = os.path.join(OUTPUT_DIR, f'cluster{cluster_id}_differential.rnk')
        rnk_df.to_csv(rnk_path, sep='\t', index=False, header=False)

        out_path = os.path.join(OUTPUT_DIR, f'cluster{cluster_id}_differential_GSEA')
        gp.prerank(rnk=rnk_df, gene_sets=GMT_PATH, outdir=out_path, min_size=5,
                   max_size=1000, permutation_num=1000, seed=42, threads=4)
        res = pd.read_csv(os.path.join(out_path, 'gseapy.gene_set.prerank.report.csv'))
        emt = res[res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False, na=False)]
        print(f"\nCluster {cluster_id} DIFFERENTIAL gradient signal -- EMT result:")
        if not emt.empty:
            print(emt[['Term', 'NES', 'FDR q-val']].to_string(index=False))
        else:
            print("  EMT not in tested set")
        print("Top 10 pathways by differential gradient signal:")
        print(res[['Term', 'NES', 'FDR q-val']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
