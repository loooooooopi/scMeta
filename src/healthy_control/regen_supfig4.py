"""
Regenerate Supplementary Figure 4 inputs: per-fold + ensemble FPR and per-fold
ROC AUC / AUPRC for the CORRECTED matched-tissue healthy negative control
(52,509 cells / 9 donors), fixing the tissue-name-matching gap in the original
RELEVANT_TISSUES filter (test_healthy_v3.py) that missed several colorectal
and ovarian donors (e.g. 'ascending colon', 'sigmoid colon', 'left ovary',
'right ovary' were not matched by the old list).

Inference only. No retraining. Reuses the exact methodology/functions from
test_healthy_v3.py (which itself reuses test_healthy_v2.py's methodology):
bridge kNN edges via pynndescent, NeighborLoader 2-hop inductive inference,
v2b_scMeta_models/ fold checkpoints, plain-argmax decision rule,
threshold-free ROC/PR construction against malignant validation negatives.

test_healthy_v3.py is NOT modified; its functions are imported and reused.
This script additionally saves PER-FOLD (not just ensemble-averaged)
probabilities for the corrected cell subset, which test_healthy_v3.py did not
persist to disk.

Outputs (this directory):
  regen_supfig4_perfold_cell_probs.npz   (per-fold + ensemble probs, obs)
  regen_supfig4_fold_summary.csv         (per-fold + ensemble FPR/AUC/AUPRC)
  regen_supfig4_organ_summary.csv        (ensemble FPR per organ, corrected mapping)
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from train_v2 import prepare_data, make_model, NUM_NEIGHBORS_EVAL, EVAL_BATCH_SIZE
from test_healthy_v3 import (
    load_healthy_data_full, build_bridge_edges,
    get_healthy_predictions, get_malignant_val_prob_mets,
    MODEL_DIR, N_FOLDS,
)

# Corrected organ mapping (fixes the tissue-name-matching gap):
ORGAN_MAP = {
    'lung': 'Lung',
    'mammary gland': 'Breast', 'breast': 'Breast',
    'large intestine': 'Colorectal', 'colon': 'Colorectal',
    'ascending colon': 'Colorectal', 'sigmoid colon': 'Colorectal',
    'ovary': 'Ovary', 'left ovary': 'Ovary', 'right ovary': 'Ovary',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    data = prepare_data()
    obs_all, healthy_x_all = load_healthy_data_full(data.gene_names)

    mask = obs_all['tissue'].isin(ORGAN_MAP.keys()).values
    obs = obs_all[mask].reset_index(drop=True)
    obs['organ'] = obs['tissue'].map(ORGAN_MAP)
    healthy_x = np.ascontiguousarray(healthy_x_all[mask])
    n_healthy = healthy_x.shape[0]
    print(f"Corrected matched-tissue set: {n_healthy} cells, "
          f"{obs['donor_id'].nunique()} donors, organs={sorted(obs['organ'].unique())}",
          flush=True)

    n_malignant = data.num_nodes
    bridge_edge_index, _ = build_bridge_edges(data.x.numpy(), healthy_x, n_malignant)
    combined_edge_index = torch.cat([data.edge_index, bridge_edge_index], dim=1)
    combined_x = torch.cat([data.x, torch.tensor(healthy_x, dtype=torch.float32)], dim=0)
    combined_data = Data(x=combined_x, edge_index=combined_edge_index)
    healthy_node_idx = np.arange(n_malignant, n_malignant + n_healthy)

    all_probs = []
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        print(f"\n=== Fold {fold} ===", flush=True)
        model = make_model("scMeta", combined_x.shape[1], device)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/5foldCV_fold{fold}_scMeta.pt", map_location=device))
        model.to(device).eval()

        probs = get_healthy_predictions(model, combined_data, healthy_node_idx, device)
        all_probs.append(probs)
        preds = probs.argmax(axis=1)
        fpr = float(np.mean(preds != 0))
        print(f"Fold {fold}: plain-argmax FPR = {fpr*100:.2f}% ({int((preds!=0).sum())}/{n_healthy})",
              flush=True)

        val_idx = data.kfold_splits[fold - 1]["val_idx"]
        y_true_val = data.y_binary[torch.as_tensor(val_idx, dtype=torch.long)].numpy()
        prob_mets_val = get_malignant_val_prob_mets(model, data, val_idx, device)
        prob_mets_healthy = probs[:, 1] + probs[:, 2]
        y_true_combined = np.concatenate([y_true_val, np.zeros(n_healthy, dtype=int)])
        prob_combined = np.concatenate([prob_mets_val, prob_mets_healthy])
        auc = roc_auc_score(y_true_combined, prob_combined)
        auprc = average_precision_score(y_true_combined, prob_combined)
        print(f"Fold {fold}: AUC = {auc:.4f}, AUPRC = {auprc:.4f}", flush=True)

        fold_rows.append({"fold": fold, "n_cells": n_healthy, "false_positive_rate": fpr,
                           "roc_auc": auc, "auprc": auprc})
        del model
        torch.cuda.empty_cache()

    ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_fpr = float(np.mean(ensemble_preds != 0))
    ensemble_auc = np.mean([r["roc_auc"] for r in fold_rows])
    ensemble_auprc = np.mean([r["auprc"] for r in fold_rows])
    print(f"\n=== Ensemble ===\nFPR = {ensemble_fpr*100:.4f}% "
          f"({int((ensemble_preds!=0).sum())}/{n_healthy})", flush=True)
    fold_rows.append({"fold": "ensemble", "n_cells": n_healthy, "false_positive_rate": ensemble_fpr,
                       "roc_auc": ensemble_auc, "auprc": ensemble_auprc})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv("./regen_supfig4_fold_summary.csv", index=False)
    print(fold_df.to_string(index=False))

    organ_df = pd.DataFrame({
        "organ": obs['organ'].values, "donor_id": obs['donor_id'].values,
        "pred_class": ensemble_preds,
    })
    organ_summary = organ_df.groupby("organ").agg(
        n_cells=("pred_class", "size"),
        n_donors=("donor_id", "nunique"),
        false_positive_rate=("pred_class", lambda s: float(np.mean(s != 0))),
    ).reset_index()
    organ_summary.to_csv("./regen_supfig4_organ_summary.csv", index=False)
    print(organ_summary.to_string(index=False))

    np.savez("./regen_supfig4_perfold_cell_probs.npz",
             fold1=all_probs[0], fold2=all_probs[1], fold3=all_probs[2],
             fold4=all_probs[3], fold5=all_probs[4], ensemble=ensemble_probs,
             organ=obs['organ'].values.astype(str), donor_id=obs['donor_id'].values.astype(str))
    print("\nWritten: regen_supfig4_fold_summary.csv, regen_supfig4_organ_summary.csv, "
          "regen_supfig4_perfold_cell_probs.npz")


if __name__ == "__main__":
    main()
