"""
Extract scMeta-graphloss learned embeddings for every malignant cell in the
atlas, for the manuscript's Figure 3(a) UMAP (latent embedding space, colored
by cancer type / metastatic label / biopsy site).

Memory-conscious version of train_v2.prepare_data(): this interactive
session is capped at 10GB RAM (cgroup limit), and prepare_data() OOMs
because it does a full (non-backed) sc.read_h5ad() of the ~3M-cell whole
atlas before subsetting to Malignant cells. Here we subset to Malignant +
valid-label cells in backed mode first (same final cell set and graph
prepare_data() would produce, since obsp/connectivities slicing after
Malignant-filter + valid-label-filter is order-independent), so only the
~460K-cell x 3,981-hallmark-gene subset is ever materialized densely.

Uses the fold-1 5-fold-CV scMeta-graphloss checkpoint
(v2b_scMeta_models/5foldCV_fold1_scMeta.pt) to embed ALL cells via a
NeighborLoader-batched full-graph forward pass (same pattern as
evaluate_inductive in train_v2.py). This is a visualization of the learned
representation space, not an evaluation metric, so no train/val split is
needed -- every cell is embedded once.
"""
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from scMeta import scMeta
from label_rules import recompute_metastasis_label

NUM_NEIGHBORS_EMBED = [25, 25]
EMBED_BATCH_SIZE = 4096
CHECKPOINT = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/v2b_scMeta_models/5foldCV_fold1_scMeta.pt'
DATA_PATH = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
HALLMARK_GENES_PATH = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/data/h.all.v2024.1.Hs.symbols.gmt'
HIDDEN_DIM = 256
CONV_TYPE = 'TransformerConv'

OUT_EMB = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings.npy'
OUT_META = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings_meta.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

print("Opening atlas (backed)...")
ad_full = sc.read_h5ad(DATA_PATH, backed='r')
recompute_metastasis_label(ad_full)

valid_mask = (
    (ad_full.obs['Final_cell_type'] == 'Malignant') &
    (ad_full.obs['metastasis_label'].isin(['No_Mets', 'Regional_Mets', 'Distant_Mets']))
)
print(f"Malignant cells with a defined metastasis label: {valid_mask.sum()} / {ad_full.n_obs}")

print("Reading hallmark gene list...")
hallmark_genes = set()
with open(HALLMARK_GENES_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        hallmark_genes.update(parts[2:])
valid_genes = sorted([g for g in hallmark_genes if g in ad_full.var_names])
print(f"{len(valid_genes)} hallmark genes present in atlas.")

print("Materializing malignant+hallmark subset into memory (this is the only dense load)...")
ad = ad_full[valid_mask, valid_genes].to_memory()
ad_full.file.close()
del ad_full
gc.collect()
print("Subset shape:", ad.shape)

if not isinstance(ad.X, np.ndarray):
    X = torch.tensor(ad.X.toarray(), dtype=torch.float32)
else:
    X = torch.tensor(ad.X, dtype=torch.float32)

adj = ad.obsp['connectivities'].tocoo()
edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

cancer_type = ad.obs['Final_cancer_type'].values
metastasis_label = ad.obs['metastasis_label'].values
biopsy_site = ad.obs['Final_tissue_backup'].values
obs_names = ad.obs_names.values

data = Data(x=X, edge_index=edge_index)
data.num_node_features_ = X.shape[1]
print("Graph: ", data.num_nodes, "nodes,", edge_index.shape[1], "edges,", data.num_node_features_, "features")

del ad
gc.collect()

print("Loading scMeta-graphloss checkpoint (fold 1)...")
model = scMeta(input_dim=data.num_node_features_, hidden_dim=HIDDEN_DIM, num_classes=3, conv_type=CONV_TYPE).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

all_idx = torch.arange(data.num_nodes, dtype=torch.long)
loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS_EMBED,
    input_nodes=all_idx,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
)

embeddings = torch.zeros(data.num_nodes, HIDDEN_DIM, dtype=torch.float32)
n_done = 0
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        _, emb = model(batch.x, batch.edge_index, return_embedding=True)
        emb = emb[:batch.batch_size].cpu()
        seed_idx = batch.input_id.cpu()
        embeddings[seed_idx] = emb
        n_done += batch.batch_size
        if n_done % 40960 < EMBED_BATCH_SIZE or n_done >= data.num_nodes:
            print(f"  embedded {n_done}/{data.num_nodes}")

print("Saving embeddings + metadata...")
np.save(OUT_EMB, embeddings.numpy())
meta = pd.DataFrame({
    'cell_id': obs_names,
    'cancer_type': cancer_type,
    'metastasis_label': metastasis_label,
    'biopsy_site': biopsy_site,
})
meta.to_csv(OUT_META, index=False)
print("Done. Embeddings shape:", embeddings.shape)
