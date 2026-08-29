"""
Revision3 rewrite of the healthy-cell negative control (Reviewer 2 point 3).

Fixes relative to test_healthy.py:
  1. Real inductive graph inference. Healthy cells are external to the atlas
     and have no edges of their own, so we build bridge edges from each
     healthy cell to its k nearest malignant-cell neighbors in the same
     1579-hallmark-gene expression space the model was trained on (via
     pynndescent, since exact kNN against ~550k reference cells is
     expensive), then run NeighborLoader seeded at the healthy nodes over
     the malignant graph + bridge edges -- same 2-hop sampling
     (num_neighbors=[25,25]) used everywhere else in train_v2.py, instead of
     the previous self-loop-only edge_index.
  2. No confidence threshold. The headline number is the plain-argmax false
     positive rate (fraction of healthy cells classified Regional_Mets or
     Distant_Mets), computed identically to how the main results are scored,
     instead of a p>0.90 threshold that silently drops "unconfident" cells
     from the denominator and inflates the reported success rate.
  3. A threshold-free ROC/PR curve per fold, pooling that fold's malignant
     validation cells (true label) with all healthy cells (true label =
     No_Mets), scored by the same model.
  4. Uses the scMeta-graphloss checkpoints (v2b_scMeta_models/), the model
     version selected as the paper's primary model after DeLong testing.

Outputs: healthy_negative_control_results.csv (per-fold + ensemble summary),
healthy_donor_summary.csv, healthy_cell_predictions.csv (per-cell release
list: donor id, tissue, cell type, ensemble-averaged predicted class).
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_healthy_data(gene_names):
    print(f"Loading healthy reference from {CELLXGENE_PATH}...")
    adata = sc.read_h5ad(CELLXGENE_PATH)

    relevant_tissues = ['lung', 'ovary', 'mammary gland', 'breast', 'large intestine', 'colon']
    adata = adata[adata.obs['tissue'].isin(relevant_tissues)].copy()
    print(f"Healthy epithelial cells from target tissues: {adata.n_obs}")

    adata.var_names = adata.var['feature_name'].astype(str)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    num_features = len(gene_names)
    new_X = np.zeros((adata.n_obs, num_features), dtype=np.float32)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    n_matched = 0
    for i, gene in enumerate(adata.var_names):
        if gene in gene_to_idx:
            new_idx = gene_to_idx[gene]
            col = adata.X[:, i].toarray().flatten() if sp.issparse(adata.X) else adata.X[:, i]
            new_X[:, new_idx] = col
            n_matched += 1
    print(f"Matched {n_matched}/{num_features} training genes in healthy reference "
          f"(unmatched genes zero-filled).")

    return adata, new_X


def build_bridge_edges(malignant_x, healthy_x, n_malignant, k=N_BRIDGE_NEIGHBORS):
    print(f"Building approximate kNN bridge (k={k}) from healthy cells into the malignant graph...")
    index = NNDescent(malignant_x, n_neighbors=k, metric="euclidean", random_state=0)
    index.prepare()
    neighbor_idx, _ = index.query(healthy_x, k=k)  # [n_healthy, k], indices into malignant_x

    n_healthy = healthy_x.shape[0]
    healthy_node_ids = np.arange(n_malignant, n_malignant + n_healthy)
    src = neighbor_idx.reshape(-1)  # malignant neighbor indices
    dst = np.repeat(healthy_node_ids, k)  # corresponding healthy target node
    bridge_edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return bridge_edge_index


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
    return torch.softmax(logits, dim=1).numpy()  # [n_healthy, 3] -> No_Mets, Regional_Mets, Distant_Mets


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


def main():
    data = prepare_data()
    n_malignant = data.num_nodes
    adata, healthy_x_np = load_healthy_data(data.gene_names)
    n_healthy = healthy_x_np.shape[0]

    malignant_x_np = data.x.numpy()
    bridge_edge_index = build_bridge_edges(malignant_x_np, healthy_x_np, n_malignant)
    combined_edge_index = torch.cat([data.edge_index, bridge_edge_index], dim=1)
    combined_x = torch.cat([data.x, torch.tensor(healthy_x_np, dtype=torch.float32)], dim=0)
    combined_data = Data(x=combined_x, edge_index=combined_edge_index)
    healthy_node_idx = np.arange(n_malignant, n_malignant + n_healthy)

    donor_ids = adata.obs['donor_id'].astype(str).values
    tissues = adata.obs['tissue'].astype(str).values
    cell_types = adata.obs['cell_type'].astype(str).values

    all_probs = []  # one [n_healthy, 3] array per fold
    fold_summaries = []
    roc_rows = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n=== Fold {fold} ===")
        model = make_model("scMeta", combined_x.shape[1], device)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/5foldCV_fold{fold}_scMeta.pt", map_location=device))
        model.to(device).eval()

        probs = get_healthy_predictions(model, combined_data, healthy_node_idx, device)
        all_probs.append(probs)
        preds = probs.argmax(axis=1)  # 0=No_Mets, 1=Regional_Mets, 2=Distant_Mets -- plain argmax, no threshold
        fpr = float(np.mean(preds != 0))
        print(f"Fold {fold}: plain-argmax false positive rate (healthy cells called Regional/Distant Mets) "
              f"= {fpr*100:.2f}%  ({int((preds!=0).sum())}/{n_healthy})")
        fold_summaries.append({
            "fold": fold, "n_healthy_cells": n_healthy,
            "frac_No_Mets": float(np.mean(preds == 0)),
            "frac_Regional_Mets": float(np.mean(preds == 1)),
            "frac_Distant_Mets": float(np.mean(preds == 2)),
            "false_positive_rate": fpr,
        })

        # threshold-free ROC/PR: this fold's malignant val cells + all healthy cells (label 0)
        val_idx = data.kfold_splits[fold - 1]["val_idx"]
        y_true_val = data.y_binary[torch.as_tensor(val_idx, dtype=torch.long)].numpy()
        prob_mets_val = get_malignant_val_prob_mets(model, data, val_idx, device)
        prob_mets_healthy = probs[:, 1] + probs[:, 2]

        y_true_combined = np.concatenate([y_true_val, np.zeros(n_healthy, dtype=int)])
        prob_combined = np.concatenate([prob_mets_val, prob_mets_healthy])
        auc = roc_auc_score(y_true_combined, prob_combined)
        auprc = average_precision_score(y_true_combined, prob_combined)
        print(f"Fold {fold}: threshold-free AUC (malignant val + healthy) = {auc:.4f}, AUPRC = {auprc:.4f}")
        roc_rows.append({"fold": fold, "auc_with_healthy_negatives": auc, "auprc_with_healthy_negatives": auprc,
                          "n_malignant_val": len(val_idx), "n_healthy": n_healthy})

    # --- Ensemble (probability-averaged across folds) ---
    ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_fpr = float(np.mean(ensemble_preds != 0))
    print(f"\n=== Ensemble (5-fold probability average) ===")
    print(f"Plain-argmax false positive rate: {ensemble_fpr*100:.2f}% "
          f"({int((ensemble_preds!=0).sum())}/{n_healthy})")

    fold_df = pd.DataFrame(fold_summaries)
    roc_df = pd.DataFrame(roc_rows)
    summary_df = fold_df.merge(roc_df, on="fold")
    summary_df.to_csv("./healthy_negative_control_results.csv", index=False)

    ensemble_row = pd.DataFrame([{
        "fold": "ensemble", "n_healthy_cells": n_healthy,
        "frac_No_Mets": float(np.mean(ensemble_preds == 0)),
        "frac_Regional_Mets": float(np.mean(ensemble_preds == 1)),
        "frac_Distant_Mets": float(np.mean(ensemble_preds == 2)),
        "false_positive_rate": ensemble_fpr,
        "auc_with_healthy_negatives": roc_df["auc_with_healthy_negatives"].mean(),
        "auprc_with_healthy_negatives": roc_df["auprc_with_healthy_negatives"].mean(),
        "n_malignant_val": np.nan,
    }])
    pd.concat([summary_df, ensemble_row], ignore_index=True).to_csv(
        "./healthy_negative_control_results.csv", index=False)

    # --- per-donor / per-tissue breakdown (ensemble) ---
    cell_df = pd.DataFrame({
        "donor_id": donor_ids, "tissue": tissues, "cell_type": cell_types,
        "pred_class": ensemble_preds, "prob_No_Mets": ensemble_probs[:, 0],
        "prob_Regional_Mets": ensemble_probs[:, 1], "prob_Distant_Mets": ensemble_probs[:, 2],
    })
    cell_df.to_csv("./healthy_cell_predictions.csv", index=False)

    donor_summary = cell_df.groupby(["donor_id", "tissue"]).agg(
        n_cells=("pred_class", "size"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()
    donor_summary.to_csv("./healthy_donor_summary.csv", index=False)

    tissue_summary = cell_df.groupby("tissue").agg(
        n_cells=("pred_class", "size"),
        n_donors=("donor_id", "nunique"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()
    print("\nPer-tissue false positive rate (ensemble):")
    print(tissue_summary.to_string(index=False))
    tissue_summary.to_csv("./healthy_tissue_summary.csv", index=False)

    print("\nWritten: healthy_negative_control_results.csv, healthy_donor_summary.csv, "
          "healthy_cell_predictions.csv, healthy_tissue_summary.csv")


if __name__ == "__main__":
    main()
