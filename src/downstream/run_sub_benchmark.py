import os
# CRITICAL: Prevent CPU thrashing in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import gseapy as gp
import gc
import concurrent.futures
from torch_geometric.data import Data

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from scMeta import scMeta
from label_rules import recompute_metastasis_label
from inductive_saliency import compute_inductive_saliency_multi_threshold

# Global dictionary to hold heavy data for copy-on-write memory sharing across workers
SHARED = {}

def process_cluster(cluster_id):
    torch.set_num_threads(1) # Force 1 PyTorch thread per worker
    
    # Unpack shared variables
    args = SHARED['args']
    ad = SHARED['ad']
    X_dense = SHARED['X_dense']
    valid_genes = SHARED['valid_genes']
    gene_to_idx = SHARED['gene_to_idx']
    num_features = SHARED['num_features']
    device = SHARED['device']
    graph_data = SHARED['graph_data']

    print(f"\n{'='*50}\nBENCHMARKING CLUSTER {cluster_id}\n{'='*50}")
    
    out_dir_cluster = os.path.join(args.out_dir, f"cluster_{cluster_id}_benchmark")
    summary_path = os.path.join(out_dir_cluster, f'Cluster_{cluster_id}_Summary.csv')
    
    # Check if results already exist to skip heavy computation
    if os.path.exists(summary_path):
        print(f"Results for cluster {cluster_id} already exist. Skipping computation.")
        try:
            # Load existing data to ensure it is included in the master summary
            existing_df = pd.read_csv(summary_path)
            return existing_df.to_dict('records')
        except Exception as e:
            print(f"Existing file corrupted, re-running cluster {cluster_id}... ({e})")
            
    os.makedirs(out_dir_cluster, exist_ok=True)

    # Define Groups & 1:1 Downsample Primary
    target_mask = (ad.obs['metastasis_label'] == 'Regional_Mets') & (ad.obs['leiden'] == cluster_id)
    primary_mask = ad.obs['metastasis_label'] == 'No_Mets'
    
    n_target = target_mask.sum()
    print(f"Cluster {cluster_id} contains {n_target} cells.")
    
    if n_target < 10:
        print(f"Skipping cluster {cluster_id} due to low cell count (< 10).")
        return []
    
    primary_indices = np.where(primary_mask)[0]
    np.random.seed(42)
    sampled_primary_idx = np.random.choice(primary_indices, size=n_target, replace=False)
    
    test_mask = np.zeros(ad.n_obs, dtype=bool)
    test_mask[target_mask] = True
    test_mask[sampled_primary_idx] = True
    
    ad_test = ad[test_mask].copy()
    ad_test.obs['benchmark_group'] = np.where(ad_test.obs['metastasis_label'] == 'No_Mets', 'Primary', 'Target')

    # ==========================================
    # METHOD 1: WILCOXON DEG
    # ==========================================
    print(f"[{cluster_id}] --- Running Method 1: Wilcoxon DEG ---")
    sc.tl.rank_genes_groups(ad_test, groupby='benchmark_group', groups=['Target'], reference='Primary', method='wilcoxon')
    deg_res = sc.get.rank_genes_groups_df(ad_test, group='Target')
    deg_rnk = deg_res[['names', 'scores']].rename(columns={'names': 'Gene', 'Score': 'scores'}).set_index('Gene')['scores']
    
    # ==========================================
    # METHOD 2: GLOBAL LR * LOCAL EXPRESSION
    # ==========================================
    print(f"[{cluster_id}] --- Running Method 2: Global Baseline LR ---")
    baseline_df = pd.read_csv(args.baseline_rnk, sep='\t', header=None, names=['Gene', 'Weight'])
    baseline_weights = np.zeros(num_features)
    for _, row in baseline_df.iterrows():
        if row['Gene'] in gene_to_idx:
            baseline_weights[gene_to_idx[row['Gene']]] = row['Weight']
            
    target_X = X_dense[target_mask]
    target_mean_expr = np.mean(target_X, axis=0)
    lr_importance = np.abs(baseline_weights * target_mean_expr)
    lr_rnk = pd.Series(lr_importance, index=valid_genes)

    # ==========================================
    # METHOD 3: SCMETA GRADIENTS
    # ==========================================
    print(f"[{cluster_id}] --- Running Method 3: scMeta Gradients (real inductive graph) ---")
    thresholds = [0.0, 0.70, 0.80, 0.90, 0.95]
    scmeta_trackers = {t: np.zeros(num_features) for t in thresholds}
    scmeta_counts = {t: 0 for t in thresholds}

    seed_idx = np.where(target_mask)[0]  # indices into the full malignant graph
    target_class = 1

    for fold in range(1, 6):
        model_path = os.path.join(args.model_dir, f"5foldCV_fold{fold}_scMeta.pt")
        model = scMeta(input_dim=num_features, hidden_dim=args.hidden_dim, num_classes=args.num_classes, conv_type=args.conv_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        fold_results = compute_inductive_saliency_multi_threshold(
            model, graph_data, seed_idx, target_class, device,
            thresholds=thresholds, batch_size=args.batch_size)
        for t in thresholds:
            sal_sum, count = fold_results[t]
            scmeta_trackers[t] += sal_sum
            scmeta_counts[t] += count

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    scmeta_rnks = {}
    for t in thresholds:
        if scmeta_counts[t] > 0:
            avg_sal = scmeta_trackers[t] / scmeta_counts[t]
            scmeta_rnks[f'scMeta_Conf_{int(t*100)}'] = pd.Series(avg_sal, index=valid_genes)

    # ==========================================
    # RUN GSEA & ENRICHR PIPELINE
    # ==========================================
    print(f"[{cluster_id}] --- Processing Enrichments ---")
    all_rnks = {'Wilcoxon_DEG': deg_rnk, 'Baseline_LR': lr_rnk}
    all_rnks.update(scmeta_rnks)
    
    cluster_summary_data = []

    for name, rnk_series in all_rnks.items():
        rnk_df = rnk_series.reset_index()
        rnk_df.columns = ['Gene', 'Score']
        rnk_df = rnk_df.sort_values(by='Score', ascending=False)
        
        gsea_out = os.path.join(out_dir_cluster, f'gsea_{name}')
        gp.prerank(rnk=rnk_df, gene_sets=args.gmt, outdir=gsea_out, min_size=5, max_size=1000, seed=42)
        gsea_res = pd.read_csv(os.path.join(gsea_out, 'gseapy.gene_set.prerank.report.csv'))
        emt_gsea = gsea_res[gsea_res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False, na=False)]
        
        top_500 = rnk_df.head(500)['Gene'].tolist()
        enr_res = gp.enrichr(gene_list=top_500, gene_sets='MSigDB_Hallmark_2020', outdir=None).res2d
        emt_enr = enr_res[enr_res['Term'].str.contains('Epithelial Mesenchymal Transition', case=False, na=False)]
        
        cluster_summary_data.append({
            'Cluster': cluster_id,
            'Method': name,
            # 'scMeta' in name -> per-threshold pooled confident-cell count from
            # scmeta_counts; otherwise (Wilcoxon_DEG, Baseline_LR) the full
            # cluster size, n_target. NOTE: previously used
            # scmeta_counts.get(..., n_target) with an integer 0 fallback key,
            # which incorrectly matched scmeta_counts' float 0.0 threshold key
            # (0 == 0.0 in Python dict lookups) for every non-scMeta method,
            # silently substituting the scMeta_Conf_0 5-fold-pooled count in
            # place of n_target. GSEA_NES/GSEA_FDR were unaffected (computed
            # from the correct rankings); only this reported count was wrong.
            'Confident_Cells': scmeta_counts[float(name.split('_')[-1])/100] if 'scMeta' in name else n_target,
            'GSEA_NES': emt_gsea['NES'].values[0] if not emt_gsea.empty else None,
            'GSEA_FDR': emt_gsea['FDR q-val'].values[0] if not emt_gsea.empty else None,
            'Enrichr_Adj_Pval': emt_enr['Adjusted P-value'].values[0] if not emt_enr.empty else None
        })
        
    summary_df = pd.DataFrame(cluster_summary_data)
    summary_df.to_csv(summary_path, index=False)
    
    # Clean up local variables
    del ad_test, target_X, all_rnks
    gc.collect()
    
    return cluster_summary_data


def parse_args():
    parser = argparse.ArgumentParser(description="scMeta vs Baseline Benchmark Tool")
    parser.add_argument("--data", type=str, required=True, help="Path to the .h5ad dataset")
    parser.add_argument("--clusters", type=str, required=True, help="Path to leiden_clusters.csv")
    parser.add_argument("--gmt", type=str, required=True, help="Path to MSigDB .gmt file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with scMeta .pt models")
    parser.add_argument("--baseline_rnk", type=str, required=True, help="Path to baseline_gene_importance.rnk")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_cluster", type=str, default="all", help="Leiden cluster ID to test, or 'all'")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    # scMeta Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--conv_type", type=str, default="TransformerConv")
    parser.add_argument("--batch_size", type=int, default=10000)
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking Target: {args.target_cluster} on device: {device} with {args.workers} workers")
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Reading target genes from GMT...")
    target_genes = []
    with open(args.gmt, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))

    print("Loading full dataset into RAM...")
    ad_full = sc.read_h5ad(args.data)
    recompute_metastasis_label(ad_full)

    print("Identifying target cells and genes...")
    # Full malignant population, not just No_Mets/Regional_Mets: message
    # passing for Method 3 (scMeta gradients) needs real graph context, same
    # as train_v2.py/run_gsea.py. Distant_Mets cells simply won't match
    # target_mask/primary_mask in process_cluster, so they're excluded from
    # the Wilcoxon/baseline-LR comparisons but still usable as graph
    # neighbors.
    valid_mask = (ad_full.obs["Final_cell_type"] == "Malignant").values
    valid_cell_idx = np.where(valid_mask)[0]

    valid_genes = [g for g in target_genes if g in ad_full.var_names]
    valid_gene_idx = np.where(ad_full.var_names.isin(valid_genes))[0]

    print("Bypassing Scanpy: Slicing sparse matrices directly at C-level...")
    X = ad_full.X
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    X_sub = X[valid_cell_idx, :]
    X_sub = X_sub.tocsc()[:, valid_gene_idx].tocsr()

    print("Rebuilding lightweight AnnData object...")
    ad = sc.AnnData(
        X=X_sub,
        obs=ad_full.obs.iloc[valid_cell_idx].copy(),
        var=ad_full.var.iloc[valid_gene_idx].copy()
    )

    print("Building malignant-cell graph (real edges, not self-loops) for Method 3...")
    adj = ad_full.obsp["connectivities"].tocsr()[valid_cell_idx, :].tocsc()[:, valid_cell_idx].tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    del ad_full, X, X_sub, adj
    gc.collect()

    print("Attaching Leiden clusters...")
    clusters_df = pd.read_csv(args.clusters, index_col=0)
    ad.obs['leiden'] = ad.obs.index.map(clusters_df['leiden'].astype(str))

    print("Generating dense matrix for fast extraction...")
    num_features = len(valid_genes)
    gene_to_idx = {gene: i for i, gene in enumerate(valid_genes)}

    X_dense = ad.X
    if sp.issparse(X_dense):
        X_dense = X_dense.toarray()
    X_dense = X_dense.astype(np.float32)

    graph_data = Data(x=torch.tensor(X_dense, dtype=torch.float32), edge_index=edge_index)

    # ==========================================
    # POPULATE SHARED CONTEXT
    # ==========================================
    SHARED['args'] = args
    SHARED['ad'] = ad
    SHARED['X_dense'] = X_dense
    SHARED['valid_genes'] = valid_genes
    SHARED['gene_to_idx'] = gene_to_idx
    SHARED['num_features'] = num_features
    SHARED['device'] = device
    SHARED['graph_data'] = graph_data

    if args.target_cluster.lower() == 'all':
        clusters_to_test = sorted([c for c in ad.obs['leiden'].unique() if str(c) != 'nan'], key=lambda x: int(x) if x.isdigit() else x)
        print(f"Target cluster set to 'all'. Iterating over {len(clusters_to_test)} clusters...")
    else:
        clusters_to_test = [args.target_cluster]

    master_summary_data = []

    # NOTE: process_cluster loads CUDA models (Method 3). A forked
    # ProcessPoolExecutor cannot safely re-initialize CUDA in worker
    # processes ("Cannot re-initialize CUDA in forked subprocess"), and
    # switching to 'spawn' would break the SHARED-dict copy-on-write pattern
    # used to share `ad`/`X_dense`/`graph_data` cheaply across workers
    # (spawned processes don't inherit that runtime state at all, only fork
    # does). So this runs sequentially in the main process instead of a
    # worker pool -- slower, but avoids silently dropping every cluster's
    # results the way the forked pool did (0/N clusters succeeded before
    # this fix, since Method 3 always raised on first CUDA use).
    print(f"\nProcessing {len(clusters_to_test)} clusters sequentially "
          f"(--workers is ignored; see note in main() about CUDA + fork)...")
    for cid in clusters_to_test:
        try:
            res = process_cluster(cid)
            if res:
                master_summary_data.extend(res)
            print(f"--- Cluster {cid} fully completed ---")
        except Exception as e:
            print(f"!!! Error processing cluster {cid}: {e} !!!")

    if len(master_summary_data) > 0:
        master_df = pd.DataFrame(master_summary_data)
        master_path = os.path.join(args.out_dir, 'Master_EMT_Benchmark_Summary.csv')
        master_df.to_csv(master_path, index=False)
        print("\n" + "="*50)
        print(f"PIPELINE COMPLETE. MASTER SUMMARY SAVED TO:\n{master_path}")
        print("="*50)
    else:
        print("\nNo data generated to save.")

if __name__ == "__main__":
    main()
    