"""
Fixes the same self-loop-only evaluation bug (Reviewer 2 point 2) in
train_no_distant.py's robustness experiment (5-fold CV strictly on
Primary vs. Regional/Local Metastasis, excluding Distant Metastasis
cells -- cited in main.tex as "AUC 0.772, Supplementary Table S10").
train_no_distant.py already uses the unified metastasis_label rule, but
evaluate_cells() still evaluates with
edge_index_val = arange(N).repeat(2, 1) (self-loops only), inconsistent
with training and with the real-graph evaluation used everywhere else in
this revision.

Reuses prepare_data() from train_no_distant.py unchanged, and
evaluate_inductive() / NUM_NEIGHBORS_EVAL / EVAL_BATCH_SIZE from
train_v2.py (identical hyperparameters, already-tested real inductive
NeighborLoader evaluation) instead of duplicating that logic.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import subgraph

from scMeta import scMeta, NT_Xent
from train_no_distant import prepare_data, CONV_TYPE, HIDDEN_DIM, TAU, GAMMA, EPOCHS, PATIENCE, LR
from train_v2 import evaluate_inductive, NUM_NEIGHBORS_EVAL, EVAL_BATCH_SIZE

SAVE_DIR = './robustness_v2_scMeta_models/'
os.makedirs(SAVE_DIR, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data()
    fold_results = []

    for fold_idx, fold in enumerate(data.cv_folds):
        print(f"\n{'='*40}\nTraining Fold {fold_idx + 1}/5\n{'='*40}")

        train_idx = np.asarray(fold["train_idx"])
        val_idx = np.asarray(fold["val_idx"])
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        edge_index_train, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
        train_data = Data(x=data.x[train_mask], edge_index=edge_index_train, y=data.y[train_mask])

        model = scMeta(input_dim=data.num_node_features, hidden_dim=HIDDEN_DIM,
                        num_classes=3, conv_type=CONV_TYPE).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_auc = -1.0
        patience_counter = 0
        best_model_path = os.path.join(SAVE_DIR, f"fold{fold_idx + 1}_best_model.pt")
        best_metrics = None

        for epoch in range(EPOCHS):
            model.train()
            loader = RandomNodeLoader(train_data, num_parts=100, shuffle=True)
            for batch in loader:
                batch = batch.to(device)
                logits, emb = model(batch.x, batch.edge_index, return_embedding=True)
                loss_ce = F.cross_entropy(logits, batch.y)
                loss_con = NT_Xent(emb, tau=TAU)
                loss = loss_ce + GAMMA * loss_con
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = evaluate_inductive(model, data, val_idx, device, model_type="scMeta")

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Cell-level AUC = {metrics['auc']:.4f}")

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_metrics = metrics
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Finished Fold {fold_idx + 1}. Best AUC: {best_auc:.4f}")
        fold_results.append({"fold": fold_idx + 1, **best_metrics})

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv("no_distant_v2_results.csv", index=False)
    print("\n\nFinal results (real inductive evaluation):")
    print(results_df)
    print(f"\nMean AUC across folds: {results_df['auc'].mean():.4f}")


if __name__ == '__main__':
    train()
