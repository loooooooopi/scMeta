"""
Follow-up to train_v2.py: same data/splits/eval, but replaces the topology-
blind NT_Xent contrastive term (which pairs each embedding with a randomly
permuted embedding, never touching edge_index) with graph_nt_xent, which uses
real graph edges as positive pairs. Goal: test whether making the contrastive
objective graph-aware lets scMeta actually outperform the plain MLP baseline,
since the MLP results are unaffected by this change (MLP has no edges/
contrastive term at all) we reuse its numbers from v2_results.csv and only
retrain scMeta here.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

from scMeta import graph_nt_xent
from train_v2 import prepare_data, make_model, evaluate_inductive, SAVE_DIR

RESULTS_PATH = './v2b_results.csv'
SAVE_DIR_B = './v2b_scMeta_models/'
os.makedirs(SAVE_DIR_B, exist_ok=True)

TAU = 0.7
GAMMA = 0.5
EPOCHS = int(os.environ.get("SCMETA_EPOCHS", 100))
PATIENCE = int(os.environ.get("SCMETA_PATIENCE", 15))
LR = 1e-4


def train_one_split_graphloss(data, train_idx, val_idx, device, save_path):
    train_idx = np.asarray(train_idx)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True

    edge_index_train, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
    train_data = Data(x=data.x[train_mask], edge_index=edge_index_train, y=data.y[train_mask])

    model = make_model("scMeta", data.num_node_features_, device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_score = -1.0
    patience_counter = 0
    best_metrics = None
    auc_is_degenerate = None

    for epoch in range(EPOCHS):
        model.train()
        loader = RandomNodeLoader(train_data, num_parts=100, shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            logits, emb = model(batch.x, batch.edge_index, return_embedding=True)
            loss_ce = F.cross_entropy(logits, batch.y)
            loss_con = graph_nt_xent(emb, batch.edge_index, tau=TAU)
            loss = loss_ce + GAMMA * loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_inductive(model, data, val_idx, device, "scMeta")

        if auc_is_degenerate is None:
            auc_is_degenerate = np.isnan(metrics["auc"])
            if auc_is_degenerate:
                print(f"    [scMeta-graphloss] WARNING: val set has a single class present (AUC undefined); "
                      f"selecting checkpoints by accuracy instead.")
        score = metrics["accuracy"] if auc_is_degenerate else metrics["auc"]

        if epoch % 5 == 0:
            print(f"    [scMeta-graphloss] epoch {epoch}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
                  f"Acc={metrics['accuracy']:.4f}")

        if best_metrics is None or score > best_score:
            best_score = score
            best_metrics = metrics
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    [scMeta-graphloss] early stopping at epoch {epoch + 1}")
                break

    return best_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data()

    if os.path.exists(RESULTS_PATH):
        all_results = pd.read_csv(RESULTS_PATH).to_dict("records")
        print(f"Resuming: found {len(all_results)} completed runs in {RESULTS_PATH}")
    else:
        all_results = []

    def already_done(scheme, fold=None, held_out=None):
        for r in all_results:
            if r["scheme"] == scheme:
                if scheme == "5foldCV" and r.get("fold") == fold:
                    return True
                if scheme == "LOCO" and r.get("held_out") == held_out:
                    return True
        return False

    for fold_idx, split in enumerate(data.kfold_splits):
        if already_done("5foldCV", fold=fold_idx + 1):
            print(f"Skipping 5foldCV fold {fold_idx + 1} (already done)")
            continue
        print(f"\n=== 5foldCV fold {fold_idx + 1}/5, model=scMeta-graphloss ===")
        save_path = os.path.join(SAVE_DIR_B, f"5foldCV_fold{fold_idx + 1}_scMeta.pt")
        metrics = train_one_split_graphloss(data, split["train_idx"], split["val_idx"], device, save_path)
        metrics.update({"scheme": "5foldCV", "fold": fold_idx + 1, "held_out": None, "model": "scMeta-graphloss"})
        all_results.append(metrics)
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    for split in data.loco_splits:
        held_out = split["held_out_cancer_type"]
        if already_done("LOCO", held_out=held_out):
            print(f"Skipping LOCO held_out={held_out} (already done)")
            continue
        print(f"\n=== LOCO held_out={held_out}, model=scMeta-graphloss ===")
        safe_name = held_out.replace(" ", "_")
        save_path = os.path.join(SAVE_DIR_B, f"LOCO_{safe_name}_scMeta.pt")
        metrics = train_one_split_graphloss(data, split["train_idx"], split["val_idx"], device, save_path)
        metrics.update({"scheme": "LOCO", "fold": None, "held_out": held_out, "model": "scMeta-graphloss"})
        all_results.append(metrics)
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    print(f"\nDone. Results written to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
