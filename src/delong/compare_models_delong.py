"""
Statistically compares scMeta-graphloss against MLP and against the original
scMeta (topology-blind NT_Xent) using the DeLong test, which operates on
paired per-cell predictions from the SAME validation cells within each split
-- much higher powered than a paired t-test across only 5 CV folds (see
v2_results.csv / v2b_results.csv, where fold-level AUC differences don't
reach significance with n=5).

Reuses the model checkpoints already saved by train_v2.py / train_v2b.py, so
this only reruns the (fast) forward pass, not training.
"""
import os
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from train_v2 import prepare_data, make_model, NUM_NEIGHBORS_EVAL, EVAL_BATCH_SIZE
from torch_geometric.loader import NeighborLoader
from delong import delong_roc_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT = {
    "MLP": "./v2_scMeta_models/{tag}_MLP.pt",
    "scMeta": "./v2_scMeta_models/{tag}_scMeta.pt",
    "scMeta-graphloss": "./v2b_scMeta_models/{tag}_scMeta.pt",
}


def get_probs(model_type, tag, data, val_idx, input_dim):
    path = CKPT[model_type].format(tag=tag)
    model = make_model("MLP" if model_type == "MLP" else "scMeta", input_dim, device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long)
    if model_type == "MLP":
        with torch.no_grad():
            logits = model(data.x[val_idx_t].to(device))
    else:
        loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS_EVAL, input_nodes=val_idx_t,
                                 batch_size=EVAL_BATCH_SIZE, shuffle=False)
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                all_logits.append(out[:batch.batch_size].cpu())
        logits = torch.cat(all_logits, dim=0).to(device)

    prob = torch.softmax(logits, dim=1)
    prob_mets = (prob[:, 1] + prob[:, 2]).detach().cpu().numpy()
    return prob_mets


def main():
    data = prepare_data()
    input_dim = data.num_node_features_
    rows = []

    splits = []
    for i, s in enumerate(data.kfold_splits):
        splits.append(("5foldCV", f"5foldCV_fold{i+1}", s["val_idx"]))
    for s in data.loco_splits:
        safe = s["held_out_cancer_type"].replace(" ", "_")
        splits.append(("LOCO", f"LOCO_{safe}", s["val_idx"]))

    for scheme, tag, val_idx in splits:
        y_true = data.y_binary[torch.as_tensor(val_idx, dtype=torch.long)].numpy()
        if len(np.unique(y_true)) < 2:
            print(f"{tag}: single-class val set, DeLong AUC undefined, skipping")
            continue

        prob_graphloss = get_probs("scMeta-graphloss", tag, data, val_idx, input_dim)
        prob_mlp = get_probs("MLP", tag, data, val_idx, input_dim)
        prob_scmeta = get_probs("scMeta", tag, data, val_idx, input_dim)

        auc_g, auc_m, z1, p1 = delong_roc_test(y_true, prob_graphloss, prob_mlp)
        auc_g2, auc_s, z2, p2 = delong_roc_test(y_true, prob_graphloss, prob_scmeta)

        print(f"{tag}: graphloss={auc_g:.4f} MLP={auc_m:.4f} scMeta={auc_s:.4f} | "
              f"graphloss-vs-MLP p={p1:.2e} | graphloss-vs-scMeta p={p2:.2e}")

        rows.append({
            "scheme": scheme, "split": tag, "n_val_cells": len(val_idx),
            "auc_graphloss": auc_g, "auc_mlp": auc_m, "auc_scmeta": auc_s,
            "z_graphloss_vs_mlp": z1, "p_graphloss_vs_mlp": p1,
            "z_graphloss_vs_scmeta": z2, "p_graphloss_vs_scmeta": p2,
        })

    df = pd.DataFrame(rows)
    df.to_csv("./delong_results.csv", index=False)
    print("\nWritten to delong_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
