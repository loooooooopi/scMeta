"""
Revision3 training/eval script. Addresses Reviewer 2 points 1 and 2 together
(they share the same data prep, so it's cleaner to fix them in one script
rather than patch train.py piecemeal):

  1. Labels come from label_rules.py (unified biopsy-site rule) instead of
     the unreproducible pre-baked metastasis_label column.
  2. Evaluation uses a real inductive graph (NeighborLoader over the full
     malignant-cell graph, seeded at held-out cells) instead of self-loops,
     so the reported metrics actually use the graph. A plain MLP baseline
     (scMetaMLP, no message passing) is trained/evaluated identically for
     comparison, on the same features and folds.
  3. Adds leave-one-cancer-type-out (LOCO) CV, in addition to the existing
     patient-stratified 5-fold CV, to test whether the model generalizes
     within a cancer type it never saw during training (Reviewer 2 point 1's
     ask to demonstrate the model is not simply separating cancer types).

Outputs one combined results CSV with rows for {scheme, fold, model} so
5-fold / LOCO and scMeta / MLP can be compared side by side.
"""
import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader, NeighborLoader
from torch_geometric.utils import subgraph
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score

from scMeta import scMeta, scMetaMLP, NT_Xent
from label_rules import recompute_metastasis_label

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = '/home/wang4887/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
HALLMARK_GENES_PATH = '/home/wang4887/scMetas/revision3/Github/data/h.all.v2024.1.Hs.symbols.gmt'
SAVE_DIR = './v2_scMeta_models/'
RESULTS_PATH = './v2_results.csv'
os.makedirs(SAVE_DIR, exist_ok=True)

CONV_TYPE = 'TransformerConv'
HIDDEN_DIM = 256
TAU = 0.7
GAMMA = 0.5
EPOCHS = int(os.environ.get("SCMETA_EPOCHS", 100))
PATIENCE = int(os.environ.get("SCMETA_PATIENCE", 15))
LR = 1e-4
NUM_NEIGHBORS_EVAL = [25, 25]
EVAL_BATCH_SIZE = 8192


def prepare_data():
    print(f"Loading data from {DATA_PATH}...")
    ad = sc.read_h5ad(DATA_PATH)

    print("Recomputing metastasis_label under the unified biopsy-site rule (label_rules.py)...")
    recompute_metastasis_label(ad)

    print(f"Original cell count: {ad.n_obs}")
    ad = ad[ad.obs['Final_cell_type'] == 'Malignant'].copy()
    print(f"Malignant cell count: {ad.n_obs}")

    print(f"Extracting hallmark genes from {HALLMARK_GENES_PATH}...")
    hallmark_genes = set()
    with open(HALLMARK_GENES_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            hallmark_genes.update(parts[2:])
    valid_genes = sorted([g for g in hallmark_genes if g in ad.var_names])
    print(f"Subsetting dataset to {len(valid_genes)} matching hallmark features...")
    ad = ad[:, valid_genes].copy()

    primary_mask = (ad.obs["metastasis_label"] == "No_Mets").values
    local_mask = (ad.obs["metastasis_label"] == "Regional_Mets").values
    distant_mask = (ad.obs["metastasis_label"] == "Distant_Mets").values

    valid_mask = primary_mask | local_mask | distant_mask
    if (~valid_mask).sum():
        print(f"Filtering out {(~valid_mask).sum()} cells with undefined metastasis labels...")
        ad = ad[valid_mask].copy()
        primary_mask = (ad.obs["metastasis_label"] == "No_Mets").values
        local_mask = (ad.obs["metastasis_label"] == "Regional_Mets").values
        distant_mask = (ad.obs["metastasis_label"] == "Distant_Mets").values

    if not isinstance(ad.X, np.ndarray):
        X = torch.tensor(ad.X.toarray(), dtype=torch.float32)
    else:
        X = torch.tensor(ad.X, dtype=torch.float32)

    adj = ad.obsp["connectivities"].tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

    y_3class = np.full(ad.n_obs, -1, dtype=int)
    y_3class[primary_mask] = 0
    y_3class[local_mask] = 1
    y_3class[distant_mask] = 2

    y_binary = np.full(ad.n_obs, -1, dtype=int)
    y_binary[primary_mask] = 0
    y_binary[local_mask] = 1
    y_binary[distant_mask] = 1

    data = Data(x=X, edge_index=edge_index, y=torch.tensor(y_3class, dtype=torch.long))
    data.y_binary = torch.tensor(y_binary, dtype=torch.long)
    data.patient_ids = ad.obs["Final_sample_id"].values
    data.cancer_type = ad.obs["Final_cancer_type"].values
    data.num_node_features_ = X.shape[1]
    data.gene_names = valid_genes

    # --- Patient-stratified 5-fold CV splits (primary/local patients only get held out) ---
    patient_df = pd.DataFrame({
        'patient_id': data.patient_ids,
        'cell_idx': np.arange(ad.n_obs),
        'y_class': y_3class,
    })
    patient_primary_local = patient_df[patient_df['y_class'].isin([0, 1])].copy()
    patient_summary = patient_primary_local.groupby('patient_id').agg({'y_class': list}).reset_index()
    patient_summary['strat_label'] = patient_summary['y_class'].apply(
        lambda x: 0 if 0 in x else (1 if 1 in x else -1))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kfold_splits = []
    patient_ids_list = patient_summary['patient_id'].values
    strat_labels = patient_summary['strat_label'].values
    for train_patient_idx, val_patient_idx in skf.split(patient_ids_list, strat_labels):
        train_patients = patient_ids_list[train_patient_idx]
        val_patients = patient_ids_list[val_patient_idx]
        train_all_cells = patient_df[patient_df['patient_id'].isin(train_patients)]['cell_idx'].values
        val_primary_local_cells = patient_df[
            (patient_df['patient_id'].isin(val_patients)) & (patient_df['y_class'].isin([0, 1]))
        ]['cell_idx'].values
        kfold_splits.append({"train_idx": train_all_cells, "val_idx": val_primary_local_cells})
    data.kfold_splits = kfold_splits

    # --- Leave-one-cancer-type-out splits ---
    loco_splits = []
    for held_out in sorted(set(data.cancer_type)):
        val_idx = np.where((data.cancer_type == held_out) & (y_3class != 2) & (y_3class != -1))[0]
        train_idx = np.where(data.cancer_type != held_out)[0]
        loco_splits.append({"held_out_cancer_type": held_out, "train_idx": train_idx, "val_idx": val_idx})
    data.loco_splits = loco_splits

    return data


def make_model(model_type, input_dim, device):
    if model_type == "scMeta":
        return scMeta(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=3, conv_type=CONV_TYPE).to(device)
    elif model_type == "MLP":
        return scMetaMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=3).to(device)
    else:
        raise ValueError(model_type)


def evaluate_inductive(model, data, val_idx, device, model_type):
    """Real inductive evaluation: for scMeta, use NeighborLoader over the full
    graph seeded at val_idx (message passing uses real neighbours, which may
    include training cells, but no gradient flows here -- eval only). For the
    MLP baseline, edges are irrelevant so this is just a direct forward pass.
    """
    model.eval()
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long)

    if model_type == "MLP":
        with torch.no_grad():
            x_val = data.x[val_idx_t].to(device)
            logits = model(x_val)
    else:
        loader = NeighborLoader(
            data,
            num_neighbors=NUM_NEIGHBORS_EVAL,
            input_nodes=val_idx_t,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
        )
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                all_logits.append(out[:batch.batch_size].cpu())
        logits = torch.cat(all_logits, dim=0).to(device)

    y_prob = torch.softmax(logits, dim=1)
    y_true = data.y_binary[val_idx_t].cpu().numpy()
    y_prob_mets = (y_prob[:, 1] + y_prob[:, 2]).detach().cpu().numpy()
    y_pred_binary = (y_prob_mets >= 0.5).astype(int)

    metrics = {}
    metrics["auc"] = roc_auc_score(y_true, y_prob_mets) if len(np.unique(y_true)) > 1 else np.nan
    metrics["auprc"] = average_precision_score(y_true, y_prob_mets) if len(np.unique(y_true)) > 1 else np.nan
    metrics["f1"] = f1_score(y_true, y_pred_binary)
    metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
    metrics["n_val_cells"] = len(val_idx)
    return metrics


def train_one_split(data, train_idx, val_idx, model_type, device, save_path):
    train_idx = np.asarray(train_idx)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True

    if model_type == "MLP":
        train_data = Data(x=data.x[train_mask], edge_index=torch.empty((2, 0), dtype=torch.long),
                           y=data.y[train_mask])
    else:
        edge_index_train, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
        train_data = Data(x=data.x[train_mask], edge_index=edge_index_train, y=data.y[train_mask])

    model = make_model(model_type, data.num_node_features_, device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_score = -1.0
    patience_counter = 0
    best_metrics = None
    auc_is_degenerate = None  # set on first epoch: True if val set has only one class

    for epoch in range(EPOCHS):
        model.train()
        loader = RandomNodeLoader(train_data, num_parts=100, shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            if model_type == "MLP":
                logits = model(batch.x)
                loss = F.cross_entropy(logits, batch.y)
            else:
                logits, emb = model(batch.x, batch.edge_index, return_embedding=True)
                loss_ce = F.cross_entropy(logits, batch.y)
                loss_con = NT_Xent(emb, tau=TAU)
                loss = loss_ce + GAMMA * loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = evaluate_inductive(model, data, val_idx, device, model_type)

        if auc_is_degenerate is None:
            auc_is_degenerate = np.isnan(metrics["auc"])
            if auc_is_degenerate:
                print(f"    [{model_type}] WARNING: val set has a single class present (AUC undefined); "
                      f"selecting checkpoints by accuracy instead.")
        # fall back to accuracy for model selection when the val set is single-class
        # (e.g. LOCO for a cancer type with no Regional_Mets tissue in this atlas)
        score = metrics["accuracy"] if auc_is_degenerate else metrics["auc"]

        if epoch % 5 == 0:
            print(f"    [{model_type}] epoch {epoch}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
                  f"Acc={metrics['accuracy']:.4f}")

        if best_metrics is None or score > best_score:
            best_score = score
            best_metrics = metrics
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    [{model_type}] early stopping at epoch {epoch + 1}")
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

    def already_done(scheme, model_type, fold=None, held_out=None):
        for r in all_results:
            if r["scheme"] == scheme and r["model"] == model_type:
                if scheme == "5foldCV" and r.get("fold") == fold:
                    return True
                if scheme == "LOCO" and r.get("held_out") == held_out:
                    return True
        return False

    # --- Scheme 1: patient-stratified 5-fold CV ---
    for fold_idx, split in enumerate(data.kfold_splits):
        for model_type in ["scMeta", "MLP"]:
            if already_done("5foldCV", model_type, fold=fold_idx + 1):
                print(f"Skipping 5foldCV fold {fold_idx + 1}, model={model_type} (already done)")
                continue
            print(f"\n=== 5foldCV fold {fold_idx + 1}/5, model={model_type} ===")
            save_path = os.path.join(SAVE_DIR, f"5foldCV_fold{fold_idx + 1}_{model_type}.pt")
            metrics = train_one_split(data, split["train_idx"], split["val_idx"], model_type, device, save_path)
            metrics.update({"scheme": "5foldCV", "fold": fold_idx + 1, "held_out": None, "model": model_type})
            all_results.append(metrics)
            pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    # --- Scheme 2: leave-one-cancer-type-out CV ---
    for split in data.loco_splits:
        for model_type in ["scMeta", "MLP"]:
            held_out = split["held_out_cancer_type"]
            if already_done("LOCO", model_type, held_out=held_out):
                print(f"Skipping LOCO held_out={held_out}, model={model_type} (already done)")
                continue
            print(f"\n=== LOCO held_out={held_out}, model={model_type} ===")
            safe_name = held_out.replace(" ", "_")
            save_path = os.path.join(SAVE_DIR, f"LOCO_{safe_name}_{model_type}.pt")
            metrics = train_one_split(data, split["train_idx"], split["val_idx"], model_type, device, save_path)
            metrics.update({"scheme": "LOCO", "fold": None, "held_out": held_out, "model": model_type})
            all_results.append(metrics)
            pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)

    print(f"\nDone. Results written to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
