"""scMeta-GAT / scMeta-SAGE / scMeta-Tran rows of the corrected Table 1.

Architecture + training loop reuse scMeta.py and train_v2.py verbatim
(scMeta model, NT_Xent, RandomNodeLoader(num_parts=100), Adam lr=1e-4,
CE + gamma * NT_Xent).

PRE-SPECIFIED config (identical for all three conv types, no tuning here):
  hidden_dim=256, tau=0.7, gamma=0.5, heads=4, dropout=0.3, lr=1e-4,
  epochs=100, patience=15  -- i.e. train_v2.py's canonical R3 configuration,
  with (gamma, tau, hidden_dim) = the values the original notebook grid search
  selected and that train_v2.py adopted.

EVALUATION FIX: held-out cells are scored with a real inductive forward pass
through NeighborLoader over the real graph (num_neighbors=[25, 25]), exactly
as train_v2.evaluate_inductive does -- NOT the self-loop-only evaluation the
stale Table 1 used. Cell predictions are then aggregated to patient calls by
the same majority-vote rule used for every other model.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader, NeighborLoader
from torch_geometric.utils import subgraph

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from scMeta import scMeta, NT_Xent
from table1_r3_prep import load_cache
from table1_r3_common import patient_level_metrics

OUT_DIR = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/runs'
CKPT_DIR = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/ckpt'

HIDDEN_DIM = 256
TAU = 0.7
GAMMA = 0.5
LR = 1e-4
EPOCHS = 100
PATIENCE = 15
NUM_NEIGHBORS_EVAL = [25, 25]
EVAL_BATCH_SIZE = 8192

CONV = {'scMeta-GAT': 'GATConv', 'scMeta-SAGE': 'SAGEConv', 'scMeta-Tran': 'TransformerConv'}


def build_data(d):
    X = torch.from_numpy(np.asarray(d['X'], dtype=np.float32))
    data = Data(x=X,
                edge_index=torch.from_numpy(d['edge_index']),
                y=torch.from_numpy(d['y3']))
    data.y_binary = torch.from_numpy(d['yb'])
    return data


@torch.no_grad()
def inductive_predict(model, data, val_idx, device):
    """train_v2.evaluate_inductive's machinery, returning per-cell 3-class probs."""
    model.eval()
    vi = torch.as_tensor(np.asarray(val_idx), dtype=torch.long)
    loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS_EVAL,
                            input_nodes=vi, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    chunks = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        chunks.append(out[:batch.batch_size].cpu())
    logits = torch.cat(chunks, dim=0)
    return torch.softmax(logits, dim=1).numpy()


def run_task(model_name, scheme, split_name, train_idx, val_idx, data, meta, yb, device):
    tag = f'gnn__{model_name}__{scheme}__{split_name}'
    out = os.path.join(OUT_DIR, tag + '.csv')
    if os.path.exists(out):
        return
    # atomic claim so several concurrent workers never duplicate a task
    lock = os.path.join(OUT_DIR, tag + '.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f'{model_name}__{scheme}__{split_name}.pt')

    train_idx = np.asarray(train_idx)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    ei_tr, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
    train_data = Data(x=data.x[train_mask], edge_index=ei_tr, y=data.y[train_mask])

    model = scMeta(input_dim=data.x.shape[1], hidden_dim=HIDDEN_DIM, num_classes=3,
                   conv_type=CONV[model_name]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    best_score, best_state, patience_ctr = -1.0, None, 0
    degenerate = None
    pid_val = meta['patient_id'].values[val_idx]
    yb_val = yb[val_idx]

    for epoch in range(EPOCHS):
        model.train()
        for batch in RandomNodeLoader(train_data, num_parts=100, shuffle=True):
            batch = batch.to(device)
            logits, emb = model(batch.x, batch.edge_index, return_embedding=True)
            loss = F.cross_entropy(logits, batch.y) + GAMMA * NT_Xent(emb, tau=TAU)
            opt.zero_grad(); loss.backward(); opt.step()

        proba = inductive_predict(model, data, val_idx, device)
        m, _ = patient_level_metrics(proba.argmax(axis=1), yb_val, pid_val)
        if degenerate is None:
            degenerate = np.isnan(m['auroc'])
            if degenerate:
                print(f'  [{model_name}/{split_name}] single-class held-out set; '
                      f'selecting checkpoints by patient accuracy', flush=True)
        score = m['accuracy'] if degenerate else m['auroc']
        if epoch % 5 == 0:
            print(f'  [{model_name}/{scheme}/{split_name}] ep{epoch} '
                  f'AUROC={m["auroc"]:.4f} Acc={m["accuracy"]:.4f} F1={m["f1"]:.4f}', flush=True)
        if best_state is None or score > best_score:
            best_score, best_metrics, patience_ctr = score, m, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f'  early stop at epoch {epoch+1}', flush=True)
                break

    torch.save(best_state, ckpt)
    r = dict(best_metrics)
    r.update(model=model_name, scheme=scheme, split=split_name, n_val_cells=len(val_idx),
             hyperparams=f'conv={CONV[model_name]},hidden_dim={HIDDEN_DIM},heads=4,dropout=0.3,'
                         f'tau={TAU},gamma={GAMMA},lr={LR},epochs={EPOCHS},patience={PATIENCE},'
                         f'eval=NeighborLoader[25,25]')
    pd.DataFrame([r]).to_csv(out, index=False)
    print(f'done {out}: {r}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, flush=True)

    d = load_cache()
    data = build_data(d)
    meta, yb = d['meta'], d['yb']

    tasks = []
    for mname in ['scMeta-Tran', 'scMeta-GAT', 'scMeta-SAGE']:
        for i, (tr, va) in enumerate(d['folds']):
            tasks.append((mname, '5foldCV', f'fold{i+1}', tr, va))
        for proj, tr, va in d['loocv']:
            tasks.append((mname, 'LOOCV', proj, tr, va))
    # rotate the shared task list so concurrent workers start in different places;
    # claiming is atomic (lockfile), so any number of workers can co-operate.
    k = (args.shard * len(tasks)) // max(args.nshards, 1)
    mine = tasks[k:] + tasks[:k]
    print(f'worker {args.shard}/{args.nshards}: {len(tasks)} tasks, starting at {k}', flush=True)
    for t in mine:
        run_task(*t, data, meta, yb, device)


if __name__ == '__main__':
    main()
