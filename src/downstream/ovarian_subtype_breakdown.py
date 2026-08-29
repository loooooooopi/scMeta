"""
Follow-up to the Ovarian Cancer LOCO near-chance result: Regional_Mets for
ovarian cancer lumps together several anatomically distinct biopsy sites
(Omentum, Peritoneum, Ascites, Bowel, Upper Quadrant). This checks whether
the scMeta-graphloss model's LOCO failure is uniform across these sites or
concentrated in a subset, by breaking down predictions per Final_tissue_backup
value within the Ovarian Cancer held-out validation set. Reuses the already-
trained checkpoint; no retraining.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label
from train_v2 import DATA_PATH, HALLMARK_GENES_PATH, make_model, NUM_NEIGHBORS_EVAL, EVAL_BATCH_SIZE
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading atlas...")
ad = sc.read_h5ad(DATA_PATH)
recompute_metastasis_label(ad)
ad = ad[ad.obs['Final_cell_type'] == 'Malignant'].copy()

hallmark_genes = set()
with open(HALLMARK_GENES_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        hallmark_genes.update(parts[2:])
valid_genes = sorted([g for g in hallmark_genes if g in ad.var_names])
ad = ad[:, valid_genes].copy()

primary_mask = (ad.obs["metastasis_label"] == "No_Mets").values
local_mask = (ad.obs["metastasis_label"] == "Regional_Mets").values
distant_mask = (ad.obs["metastasis_label"] == "Distant_Mets").values
valid_mask = primary_mask | local_mask | distant_mask
ad = ad[valid_mask].copy()
primary_mask = (ad.obs["metastasis_label"] == "No_Mets").values
local_mask = (ad.obs["metastasis_label"] == "Regional_Mets").values
distant_mask = (ad.obs["metastasis_label"] == "Distant_Mets").values

X = torch.tensor(ad.X.toarray() if not isinstance(ad.X, np.ndarray) else ad.X, dtype=torch.float32)
adj = ad.obsp["connectivities"].tocoo()
edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

y_binary = np.full(ad.n_obs, -1, dtype=int)
y_binary[primary_mask] = 0
y_binary[local_mask] = 1
y_binary[distant_mask] = 1

data = Data(x=X, edge_index=edge_index)
data.y_binary = torch.tensor(y_binary, dtype=torch.long)
cancer_type = ad.obs["Final_cancer_type"].values
tissue_backup = ad.obs["Final_tissue_backup"].astype(str).values

held_out = "Ovarian Cancer"
val_idx = np.where((cancer_type == held_out) & (y_binary != -1))[0]
print(f"Ovarian LOCO val cells: {len(val_idx)}")

model = make_model("scMeta", X.shape[1], device)
model.load_state_dict(torch.load("./v2b_scMeta_models/LOCO_Ovarian_Cancer_scMeta.pt", map_location=device))
model.eval()

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
prob_mets = torch.softmax(logits, dim=1)[:, 1:].sum(dim=1).numpy()
pred_binary = (prob_mets >= 0.5).astype(int)

df = pd.DataFrame({
    "tissue": tissue_backup[val_idx],
    "y_true": y_binary[val_idx],
    "prob_mets": prob_mets,
    "pred": pred_binary,
})

print("\nPer-tissue-subtype breakdown (Ovarian Cancer LOCO, scMeta-graphloss):")
summary = df.groupby("tissue").agg(
    n_cells=("y_true", "size"),
    true_label=("y_true", "mean"),  # 0 = No_Mets, 1 = Mets(regional here)
    mean_pred_prob=("prob_mets", "mean"),
    frac_pred_mets=("pred", "mean"),
).round(4)
print(summary.to_string())

# AUC per tissue subtype where both classes present (there won't be, since
# label is a deterministic function of tissue -- but this shows how
# concentrated the miscalibration is per site)
from sklearn.metrics import roc_auc_score
print("\nMean predicted P(mets) by tissue (No_Mets sites should trend low, Regional sites high):")
for t, g in df.groupby("tissue"):
    print(f"  {t}: n={len(g)}, true_label={g['y_true'].iloc[0]}, mean_prob_mets={g['prob_mets'].mean():.4f}, "
          f"frac_predicted_mets={g['pred'].mean():.4f}")

summary.to_csv("./ovarian_subtype_breakdown.csv")
print("\nWritten to ovarian_subtype_breakdown.csv")
