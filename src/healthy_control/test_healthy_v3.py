"""
Revision3 (R2) expansion of the healthy-cell negative control.

Reviewer 2 asked for a larger donor set than the 7 donors of the
tissue-matched control. This script re-runs the EXACT methodology of
test_healthy_v2.py (bridge kNN edges into the malignant graph via
pynndescent, NeighborLoader 2-hop inductive inference, plain-argmax FPR,
threshold-free ROC/PR per fold, v2b_scMeta_models/ scMeta-graphloss
checkpoints, 5 folds) under two scopes:

  Scope A  matched_tissue : the original 6-tissue filter (43,522 cells / 7 donors)
  Scope B  all_epithelium : the full Tabula Sapiens epithelium (228,032 cells / 22 donors)

Inference only. No retraining. test_healthy_v2.py is untouched.

Outputs (this directory):
  healthy_negative_control_v3_results.csv
  healthy_donor_summary_v3.csv
  healthy_tissue_summary_v3.csv
  healthy_cell_predictions_v3.csv
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from pynndescent import NNDescent
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from train_v2 import prepare_data, make_model, NUM_NEIGHBORS_EVAL, EVAL_BATCH_SIZE

CELLXGENE_PATH = '/home/wang4887/scMetas/data/czi/tabula_sapiens_epithelium.h5ad'
MODEL_DIR = './v2b_scMeta_models/'
N_BRIDGE_NEIGHBORS = 25  # matches NUM_NEIGHBORS_EVAL[0]
N_FOLDS = 5

RELEVANT_TISSUES = ['lung', 'ovary', 'mammary gland', 'breast', 'large intestine', 'colon']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_healthy_data_full(gene_names):
    """Load the whole epithelium atlas once, normalize, and project onto the
    training gene space. Scope A is then a row subset of this (normalize_total
    + log1p are per-cell, so row-subsetting after is identical to filtering
    before, as test_healthy_v2.py did)."""
    print(f"Loading healthy reference from {CELLXGENE_PATH}...", flush=True)
    adata = sc.read_h5ad(CELLXGENE_PATH)
    print(f"Full healthy epithelial atlas: {adata.n_obs} cells", flush=True)

    adata.var_names = adata.var['feature_name'].astype(str)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    num_features = len(gene_names)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    src_cols, dst_cols = [], []
    for i, gene in enumerate(adata.var_names):
        if gene in gene_to_idx:
            src_cols.append(i)
            dst_cols.append(gene_to_idx[gene])
    print(f"Matched {len(src_cols)}/{num_features} training genes in healthy reference "
          f"(unmatched genes zero-filled).", flush=True)

    new_X = np.zeros((adata.n_obs, num_features), dtype=np.float32)
    src_cols = np.asarray(src_cols)
    dst_cols = np.asarray(dst_cols)
    if sp.issparse(adata.X):
        sub = adata.X[:, src_cols]
        new_X[:, dst_cols] = np.asarray(sub.todense(), dtype=np.float32)
    else:
        new_X[:, dst_cols] = np.asarray(adata.X[:, src_cols], dtype=np.float32)

    obs = pd.DataFrame({
        'donor_id': adata.obs['donor_id'].astype(str).values,
        'tissue': adata.obs['tissue'].astype(str).values,
        'cell_type': adata.obs['cell_type'].astype(str).values,
    })
    del adata
    return obs, new_X


def build_bridge_edges(malignant_x, healthy_x, n_malignant, index=None, k=N_BRIDGE_NEIGHBORS):
    print(f"Building approximate kNN bridge (k={k}) from healthy cells into the malignant graph...",
          flush=True)
    if index is None:
        index = NNDescent(malignant_x, n_neighbors=k, metric="euclidean", random_state=0)
        index.prepare()
    neighbor_idx, _ = index.query(healthy_x, k=k)  # [n_healthy, k], indices into malignant_x

    n_healthy = healthy_x.shape[0]
    healthy_node_ids = np.arange(n_malignant, n_malignant + n_healthy)
    src = neighbor_idx.reshape(-1)          # malignant neighbor indices
    dst = np.repeat(healthy_node_ids, k)    # corresponding healthy target node
    bridge_edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return bridge_edge_index, index


def get_healthy_predictions(model, combined_data, healthy_node_idx, device):
    model.eval()
    healthy_idx_t = torch.as_tensor(healthy_node_idx, dtype=torch.long)
    loader = NeighborLoader(combined_data, num_neighbors=NUM_NEIGHBORS_EVAL, input_nodes=healthy_idx_t,
                            batch_size=EVAL_BATCH_SIZE, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            all_logits.append(out[:batch.batch_size].cpu())
    logits = torch.cat(all_logits, dim=0)
    return torch.softmax(logits, dim=1).numpy()  # [n_healthy, 3] No_Mets, Regional_Mets, Distant_Mets


def get_malignant_val_prob_mets(model, data, val_idx, device):
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long)
    loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS_EVAL, input_nodes=val_idx_t,
                            batch_size=EVAL_BATCH_SIZE, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            all_logits.append(out[:batch.batch_size].cpu())
    logits = torch.cat(all_logits, dim=0)
    prob = torch.softmax(logits, dim=1)
    return (prob[:, 1] + prob[:, 2]).numpy()


def run_scope(scope_name, data, obs, healthy_x_np, nnd_index):
    n_malignant = data.num_nodes
    n_healthy = healthy_x_np.shape[0]
    print(f"\n########## SCOPE {scope_name}: {n_healthy} cells, "
          f"{obs['donor_id'].nunique()} donors, {obs['tissue'].nunique()} tissues ##########",
          flush=True)

    bridge_edge_index, nnd_index = build_bridge_edges(
        data.x.numpy(), healthy_x_np, n_malignant, index=nnd_index)
    combined_edge_index = torch.cat([data.edge_index, bridge_edge_index], dim=1)
    combined_x = torch.cat([data.x, torch.tensor(healthy_x_np, dtype=torch.float32)], dim=0)
    combined_data = Data(x=combined_x, edge_index=combined_edge_index)
    healthy_node_idx = np.arange(n_malignant, n_malignant + n_healthy)

    all_probs = []
    fold_summaries = []
    roc_rows = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n=== [{scope_name}] Fold {fold} ===", flush=True)
        model = make_model("scMeta", combined_x.shape[1], device)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/5foldCV_fold{fold}_scMeta.pt", map_location=device))
        model.to(device).eval()

        probs = get_healthy_predictions(model, combined_data, healthy_node_idx, device)
        all_probs.append(probs)
        preds = probs.argmax(axis=1)  # plain argmax, no threshold
        fpr = float(np.mean(preds != 0))
        print(f"[{scope_name}] Fold {fold}: plain-argmax FPR = {fpr*100:.2f}% "
              f"({int((preds!=0).sum())}/{n_healthy})", flush=True)
        fold_summaries.append({
            "scope": scope_name, "fold": fold, "n_healthy_cells": n_healthy,
            "frac_No_Mets": float(np.mean(preds == 0)),
            "frac_Regional_Mets": float(np.mean(preds == 1)),
            "frac_Distant_Mets": float(np.mean(preds == 2)),
            "false_positive_rate": fpr,
        })

        val_idx = data.kfold_splits[fold - 1]["val_idx"]
        y_true_val = data.y_binary[torch.as_tensor(val_idx, dtype=torch.long)].numpy()
        prob_mets_val = get_malignant_val_prob_mets(model, data, val_idx, device)
        prob_mets_healthy = probs[:, 1] + probs[:, 2]

        y_true_combined = np.concatenate([y_true_val, np.zeros(n_healthy, dtype=int)])
        prob_combined = np.concatenate([prob_mets_val, prob_mets_healthy])
        auc = roc_auc_score(y_true_combined, prob_combined)
        auprc = average_precision_score(y_true_combined, prob_combined)
        print(f"[{scope_name}] Fold {fold}: AUC = {auc:.4f}, AUPRC = {auprc:.4f}", flush=True)
        roc_rows.append({"scope": scope_name, "fold": fold,
                         "auc_with_healthy_negatives": auc,
                         "auprc_with_healthy_negatives": auprc,
                         "n_malignant_val": len(val_idx), "n_healthy": n_healthy})

    ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_fpr = float(np.mean(ensemble_preds != 0))
    print(f"\n=== [{scope_name}] Ensemble (5-fold probability average) ===")
    print(f"Plain-argmax FPR: {ensemble_fpr*100:.4f}% ({int((ensemble_preds!=0).sum())}/{n_healthy})",
          flush=True)

    fold_df = pd.DataFrame(fold_summaries)
    roc_df = pd.DataFrame(roc_rows)
    summary_df = fold_df.merge(roc_df, on=["scope", "fold"])
    ensemble_row = pd.DataFrame([{
        "scope": scope_name, "fold": "ensemble", "n_healthy_cells": n_healthy,
        "frac_No_Mets": float(np.mean(ensemble_preds == 0)),
        "frac_Regional_Mets": float(np.mean(ensemble_preds == 1)),
        "frac_Distant_Mets": float(np.mean(ensemble_preds == 2)),
        "false_positive_rate": ensemble_fpr,
        "auc_with_healthy_negatives": roc_df["auc_with_healthy_negatives"].mean(),
        "auprc_with_healthy_negatives": roc_df["auprc_with_healthy_negatives"].mean(),
        "n_malignant_val": np.nan, "n_healthy": n_healthy,
    }])
    summary_out = pd.concat([summary_df, ensemble_row], ignore_index=True)

    cell_df = pd.DataFrame({
        "scope": scope_name,
        "donor_id": obs['donor_id'].values, "tissue": obs['tissue'].values,
        "cell_type": obs['cell_type'].values,
        "pred_class": ensemble_preds, "prob_No_Mets": ensemble_probs[:, 0],
        "prob_Regional_Mets": ensemble_probs[:, 1], "prob_Distant_Mets": ensemble_probs[:, 2],
    })

    donor_summary = cell_df.groupby(["scope", "donor_id"]).agg(
        n_cells=("pred_class", "size"),
        n_tissues=("tissue", "nunique"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()
    donor_tissue_summary = cell_df.groupby(["scope", "donor_id", "tissue"]).agg(
        n_cells=("pred_class", "size"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()
    tissue_summary = cell_df.groupby(["scope", "tissue"]).agg(
        n_cells=("pred_class", "size"),
        n_donors=("donor_id", "nunique"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()

    print(f"\n[{scope_name}] Per-donor FPR (ensemble):")
    print(donor_summary.to_string(index=False), flush=True)
    print(f"\n[{scope_name}] Per-tissue FPR (ensemble):")
    print(tissue_summary.to_string(index=False), flush=True)

    del combined_data, combined_x, combined_edge_index
    return summary_out, cell_df, donor_summary, donor_tissue_summary, tissue_summary, nnd_index


def main():
    data = prepare_data()
    obs_all, healthy_x_all = load_healthy_data_full(data.gene_names)

    mask_a = obs_all['tissue'].isin(RELEVANT_TISSUES).values
    obs_a = obs_all[mask_a].reset_index(drop=True)
    healthy_x_a = np.ascontiguousarray(healthy_x_all[mask_a])
    print(f"Scope A (matched tissues): {mask_a.sum()} cells, "
          f"{obs_a['donor_id'].nunique()} donors, {obs_a['tissue'].nunique()} tissues", flush=True)

    nnd_index = None
    res_a = run_scope("A_matched_tissue", data, obs_a, healthy_x_a, nnd_index)
    nnd_index = res_a[5]
    del healthy_x_a

    res_b = run_scope("B_all_epithelium", data, obs_all, healthy_x_all, nnd_index)

    summary = pd.concat([res_a[0], res_b[0]], ignore_index=True)
    cells = pd.concat([res_a[1], res_b[1]], ignore_index=True)
    donors = pd.concat([res_a[2], res_b[2]], ignore_index=True)
    donor_tissue = pd.concat([res_a[3], res_b[3]], ignore_index=True)
    tissues = pd.concat([res_a[4], res_b[4]], ignore_index=True)

    summary.to_csv("./healthy_negative_control_v3_results.csv", index=False)
    cells.to_csv("./healthy_cell_predictions_v3.csv", index=False)
    donors.merge(donor_tissue.rename(columns={
        "n_cells": "n_cells_in_tissue",
        "false_positive_rate": "false_positive_rate_in_tissue"}),
        on=["scope", "donor_id"], how="left").to_csv("./healthy_donor_summary_v3.csv", index=False)
    tissues.to_csv("./healthy_tissue_summary_v3.csv", index=False)

    print("\n=== SCOPE COMPARISON (ensemble) ===")
    print(summary[summary["fold"] == "ensemble"].to_string(index=False))
    print("\nWritten: healthy_negative_control_v3_results.csv, healthy_donor_summary_v3.csv, "
          "healthy_tissue_summary_v3.csv, healthy_cell_predictions_v3.csv")


if __name__ == "__main__":
    main()
