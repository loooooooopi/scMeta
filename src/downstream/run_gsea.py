import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import gseapy as gp
import gc
from torch_geometric.data import Data

# ---------------------------------------------------------
# IMPORT YOUR MODEL HERE
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from scMeta import scMeta
from label_rules import recompute_metastasis_label
from inductive_saliency import compute_inductive_saliency
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="scMeta GSEA Extraction and Analysis Tool")
    
    # File Paths
    parser.add_argument("--data", type=str, required=True, help="Path to the .h5ad dataset")
    parser.add_argument("--gmt", type=str, nargs='+', required=True, help="Path(s) to one or more MSigDB .gmt files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the 5-fold .pt models")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save GSEA results")
    
    # Analysis Scenario
    parser.add_argument("--scenario", type=str, choices=['local_global', 'distant_global', 'local_subpop'], required=True, 
                        help="Which analysis to run: global local mets, global distant mets, or local sub-populations")
    
    # String matcher for printing specific pathway results (e.g., EMT or MET)
    parser.add_argument("--target_pathways", type=str, nargs='+', default=["EPITHELIAL_MESENCHYMAL_TRANSITION"], 
                        help="Strings to search for in GSEA results (e.g. EPITHELIAL_MESENCHYMAL_TRANSITION MESENCHYMAL_TO_EPITHELIAL_TRANSITION)")
    
    # Model Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the loaded model")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes the model was trained on")
    parser.add_argument("--conv_type", type=str, default="TransformerConv", help="GNN layer type")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for gradient extraction")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\nScenario: {args.scenario}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load the FULL malignant population (graph context needs all of it,
    #    not just the scenario subset -- matches evaluate_inductive() in
    #    train_v2.py, which always seeds over the full malignant graph).
    print(f"Loading dataset from {args.data}...")
    ad_full = sc.read_h5ad(args.data, backed='r')
    recompute_metastasis_label(ad_full)
    malignant_mask = (ad_full.obs["Final_cell_type"] == "Malignant").values
    ad = ad_full[malignant_mask].to_memory()
    ad_full.file.close()
    gc.collect()
    print(f"Malignant population (graph context): {ad.n_obs} cells")

    if args.scenario == "distant_global":
        scenario_mask = (ad.obs["metastasis_label"] == "Distant_Mets").values
        target_class = 2
    else:
        scenario_mask = ((ad.obs["metastasis_label"] == "No_Mets") |
                          (ad.obs["metastasis_label"] == "Regional_Mets")).values
        target_class = 1
    seed_idx = np.where(scenario_mask)[0]
    print(f"Scenario '{args.scenario}' seed set: {len(seed_idx)} cells (target_class={target_class})")

    # 2. Compute or Load Leiden Clusters (Only needed for subpop scenario).
    #    Clustering itself only makes sense on the scenario subset, so it
    #    runs on a throwaway copy ad_sub -- ad (and its original atlas-level
    #    obsp['connectivities'], needed for the GNN graph below) is untouched.
    ad.obs['leiden'] = pd.Series(index=ad.obs_names, dtype='object')
    if args.scenario == "local_subpop":
        cluster_csv_path = os.path.join(args.out_dir, "leiden_clusters.csv")

        if os.path.exists(cluster_csv_path):
            print(f"Found existing clustering results at {cluster_csv_path}. Loading...")
            clusters_df = pd.read_csv(cluster_csv_path, index_col=0)
            ad.obs.loc[clusters_df.index, 'leiden'] = clusters_df['leiden'].astype(str)
        else:
            ad_sub = ad[scenario_mask].copy()
            print("Computing Leiden clusters on Harmony PCA space for subpopulation analysis...")
            sc.pp.neighbors(ad_sub, use_rep='X_pca_harmony', n_neighbors=15)
            sc.tl.leiden(ad_sub, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)

            print(f"Saving clustering results to {cluster_csv_path}...")
            ad_sub.obs[['leiden']].to_csv(cluster_csv_path)

            print(f"Saving UMAP visualizations to {args.out_dir}...")
            sc.settings.figdir = args.out_dir
            if 'X_umap' not in ad_sub.obsm:
                print("UMAP embeddings not found. Computing them now (this may take a few minutes)...")
                sc.tl.umap(ad_sub)
            sc.pl.umap(ad_sub, color=['leiden', 'Final_cancer_type'], show=False, save="_local_subpopulations.pdf")

            ad.obs.loc[ad_sub.obs_names, 'leiden'] = ad_sub.obs['leiden'].astype(str)
            del ad_sub
            gc.collect()

    ad.obs['leiden'] = ad.obs['leiden'].astype('category')

    # 3. Prepare Feature Space (over the full malignant population) and build
    #    the real malignant-cell graph for inductive saliency.
    print("Aligning feature space with GMT database(s)...")
    target_genes = []
    for gmt_path in args.gmt:
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))
    valid_genes = [g for g in target_genes if g in ad.var_names]

    num_features = len(valid_genes)
    new_X = np.zeros((ad.n_obs, num_features), dtype=np.float32)
    gene_to_idx = {gene: i for i, gene in enumerate(valid_genes)}

    # Vectorized column selection instead of looping per-gene (looping does
    # ~1500+ individual single-column sparse slices, which is extremely slow
    # on a CSR matrix -- one batched fancy-index selection is much faster).
    src_positions, dst_positions = [], []
    for i, gene in enumerate(ad.var_names):
        if gene in gene_to_idx:
            src_positions.append(i)
            dst_positions.append(gene_to_idx[gene])
    if sp.issparse(ad.X):
        new_X[:, dst_positions] = ad.X.tocsc()[:, src_positions].toarray()
    else:
        new_X[:, dst_positions] = ad.X[:, src_positions]

    print("Building malignant-cell graph (real edges, not self-loops) for inductive saliency...")
    adj = ad.obsp["connectivities"].tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
    graph_data = Data(x=torch.tensor(new_X, dtype=torch.float32), edge_index=edge_index)

    # 4. Initialize Trackers
    if args.scenario == "local_subpop":
        unique_cancers = ad.obs["Final_cancer_type"].unique()
        unique_clusters = [c for c in ad.obs["leiden"].unique() if pd.notna(c)]
        sal_cancer = {c: np.zeros(num_features) for c in unique_cancers}
        count_cancer = {c: 0 for c in unique_cancers}
        sal_leiden = {l: np.zeros(num_features) for l in unique_clusters}
        count_leiden = {l: 0 for l in unique_clusters}
    else:
        global_saliency = np.zeros(num_features)
        global_count = 0

    # 5. Extract Gradients via real inductive inference (NeighborLoader,
    #    same 2-hop sampling as train_v2.py's evaluation), not self-loops.
    print(f"Extracting feature importance across {len(seed_idx)} seed cells (Target Class: {target_class})...")
    cancer_type_arr = ad.obs["Final_cancer_type"].values
    leiden_arr = ad.obs["leiden"].values

    for fold in range(1, 6):
        model_path = os.path.join(args.model_dir, f"5foldCV_fold{fold}_scMeta.pt")
        model = scMeta(input_dim=num_features, hidden_dim=args.hidden_dim, num_classes=args.num_classes, conv_type=args.conv_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        confident_mask, confident_saliency = compute_inductive_saliency(
            model, graph_data, seed_idx, target_class, device,
            confidence_threshold=0.70, batch_size=args.batch_size)
        confident_seed_idx = seed_idx[confident_mask]

        if args.scenario == "local_subpop":
            batch_cancers = cancer_type_arr[confident_seed_idx]
            batch_leiden = leiden_arr[confident_seed_idx]
            for i in range(len(confident_seed_idx)):
                sal_cancer[batch_cancers[i]] += confident_saliency[i]
                count_cancer[batch_cancers[i]] += 1
                if pd.notna(batch_leiden[i]):
                    sal_leiden[batch_leiden[i]] += confident_saliency[i]
                    count_leiden[batch_leiden[i]] += 1
        else:
            global_saliency += confident_saliency.sum(axis=0)
            global_count += len(confident_seed_idx)

        del model, confident_mask, confident_saliency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"Fold {fold} gradients extracted ({len(confident_seed_idx)} confident cells).")

    # 6. Run GSEA Processing
    def run_gseapy_and_report(saliency_array, count, output_prefix, title):
        if count == 0: return
        avg_sal = saliency_array / count if args.scenario == "local_subpop" else saliency_array
        rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': avg_sal}).sort_values(by='Score', ascending=False)
        out_path = os.path.join(args.out_dir, output_prefix)
        
        gp.prerank(rnk=rnk_df, gene_sets=args.gmt, outdir=out_path, min_size=5, max_size=1000, seed=42)
        res = pd.read_csv(os.path.join(out_path, 'gseapy.gene_set.prerank.report.csv'))
        
        print(f"\n{'='*50}\n{title} (n={count} confident cells)\n{'='*50}")
        print("TOP 5 PATHWAYS:")
        print(res[['Term', 'NES', 'FDR q-val']].head(5))
        
        print("\nTARGET PATHWAYS:")
        for target in args.target_pathways:
            target_row = res[res['Term'].str.contains(target, case=False, na=False)]
            if not target_row.empty:
                print(f"--- Matches for '{target}' ---")
                print(target_row[['Term', 'NES', 'FDR q-val']])
            else:
                print(f"--- '{target}' not found ---")

    if args.scenario in ["local_global", "distant_global"]:
        run_gseapy_and_report(global_saliency, global_count, 'global_results', f"GLOBAL {args.scenario.upper()} RESULTS")
    else:
        for ctype in unique_cancers:
            if count_cancer[ctype] > 50:
                run_gseapy_and_report(sal_cancer[ctype], count_cancer[ctype], f'cancer_{ctype.replace(" ", "_")}', f"CANCER TYPE: {ctype}")
        for cluster in unique_clusters:
            if count_leiden[cluster] > 50:
                run_gseapy_and_report(sal_leiden[cluster], count_leiden[cluster], f'leiden_cluster_{cluster}', f"LEIDEN CLUSTER: {cluster}")


if __name__ == "__main__":
    main()

