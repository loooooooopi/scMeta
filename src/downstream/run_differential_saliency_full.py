"""
Extends test_differential_saliency.py's single-cluster check (Cluster 56/58)
to all 84 Regional_Mets Leiden subpopulations, using the same group
construction (Target cluster vs. 1:1 downsampled Primary, seed=42) as
run_sub_benchmark.py's process_cluster(), so results are directly
comparable to Master_EMT_Benchmark_Summary.csv's Wilcoxon_DEG /
scMeta_Conf_* rows -- just adding a scMeta_Differential row per cluster.

Per-cluster results are checkpointed to disk (skip if already present), so
this is safe to interrupt and resume. Runs sequentially (no
ProcessPoolExecutor): same CUDA+fork crash reasoning as run_sub_benchmark.py.
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
PER_CLUSTER_DIR = os.path.join(OUTPUT_DIR, 'per_cluster')
HIDDEN_DIM = 256
NUM_CLASSES = 3
CONV_TYPE = 'TransformerConv'
SEED = 42
BATCH_SIZE = 512

os.makedirs(PER_CLUSTER_DIR, exist_ok=True)


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

    all_clusters = sorted([c for c in ad.obs['leiden'].dropna().unique()],
                           key=lambda x: int(x) if str(x).isdigit() else str(x))
    print(f"Total clusters to process: {len(all_clusters)}")

    # Pre-load the 5 fold models once (reused across all clusters).
    models = []
    for fold in range(1, 6):
        model_path = os.path.join(MODEL_DIR, f"5foldCV_fold{fold}_scMeta.pt")
        m = scMeta(input_dim=len(valid_genes), hidden_dim=HIDDEN_DIM,
                   num_classes=NUM_CLASSES, conv_type=CONV_TYPE).to(device)
        m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    summary_rows = []

    for cluster_id in all_clusters:
        result_path = os.path.join(PER_CLUSTER_DIR, f'cluster{cluster_id}_differential_summary.csv')
        if os.path.exists(result_path):
            print(f"[{cluster_id}] cached, loading")
            summary_rows.append(pd.read_csv(result_path).iloc[0].to_dict())
            continue

        cluster_mask = ((ad.obs["metastasis_label"] == "Regional_Mets").values &
                         (ad.obs["leiden"] == cluster_id).values)
        class2_idx = np.where(cluster_mask)[0]
        n = len(class2_idx)
        if n < 10:
            print(f"[{cluster_id}] only {n} cells, skipping")
            continue
        print(f"[{cluster_id}] {n} cells")

        rng = np.random.default_rng(SEED)
        class1_idx = rng.choice(primary_idx_all, size=n, replace=False)

        delta_g_sum = np.zeros(len(valid_genes), dtype=np.float64)
        for fold_i, model in enumerate(models, 1):
            delta_g = compute_differential_saliency(
                model, graph_data, class1_idx, class2_idx, target_class=1,
                device=device, batch_size=BATCH_SIZE)
            delta_g_sum += delta_g
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        delta_g_avg = delta_g_sum / len(models)

        rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': delta_g_avg}).sort_values(
            by='Score', ascending=False)

        gsea_out = os.path.join(PER_CLUSTER_DIR, f'cluster{cluster_id}_gsea')
        try:
            gp.prerank(rnk=rnk_df, gene_sets=GMT_PATH, outdir=gsea_out, min_size=5,
                       max_size=1000, permutation_num=1000, seed=42, threads=2)
            res = pd.read_csv(os.path.join(gsea_out, 'gseapy.gene_set.prerank.report.csv'))
            emt = res[res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False, na=False)]
            nes = float(emt['NES'].values[0]) if not emt.empty else None
            fdr = float(emt['FDR q-val'].values[0]) if not emt.empty else None
        except Exception as e:
            print(f"[{cluster_id}] GSEA FAILED: {e}")
            nes, fdr = None, None

        row = {'Cluster': cluster_id, 'Method': 'scMeta_Differential',
               'Confident_Cells': n, 'GSEA_NES': nes, 'GSEA_FDR': fdr}
        pd.DataFrame([row]).to_csv(result_path, index=False)
        summary_rows.append(row)

        sig = "SIGNIFICANT" if (fdr is not None and fdr < 0.05) else ""
        print(f"[{cluster_id}] EMT NES={nes} FDR={fdr} {sig}")
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(OUTPUT_DIR, 'Differential_Saliency_Full_Summary.csv')
    summary_df.to_csv(out_path, index=False)
    n_sig = (summary_df['GSEA_FDR'] < 0.05).sum()
    print(f"\n\nDone. {len(summary_df)} clusters processed, {n_sig} significant at FDR<0.05.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
