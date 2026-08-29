# Load libraries


```python
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import anndata as ann
import random, os
from scipy.stats import pearsonr as pr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import silhouette_score
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import gc, psutil
import joblib
import hashlib
```

    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/scanpy/_utils/__init__.py:27: FutureWarning: `__version__` is deprecated, use `importlib.metadata.version('anndata')` instead.
      from anndata import __version__ as anndata_version
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_csv from `anndata` is deprecated. Import anndata.io.read_csv instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_excel from `anndata` is deprecated. Import anndata.io.read_excel instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_hdf from `anndata` is deprecated. Import anndata.io.read_hdf instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_loom from `anndata` is deprecated. Import anndata.io.read_loom instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_mtx from `anndata` is deprecated. Import anndata.io.read_mtx instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_text from `anndata` is deprecated. Import anndata.io.read_text instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/anndata/__init__.py:70: FutureWarning: Importing read_umi_tools from `anndata` is deprecated. Import anndata.io.read_umi_tools instead.
      return module_get_attr_redirect(attr_name, deprecated_mapping=_DEPRECATED)
    /home/wang4887/scMetas/revision2/scmetas/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
import torch
import torch_geometric
from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv
from sklearn.model_selection import StratifiedKFold
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import NeighborLoader
import json
import os
from pathlib import Path
from datetime import datetime
```


```python
sc.set_figure_params(dpi=200)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 sc.set_figure_params(dpi=200)


        [... skipping hidden 1 frame]


    File ~/scMetas/revision2/scmetas/lib/python3.11/site-packages/scanpy/_settings.py:488, in ScanpyConfig.set_figure_params(self, scanpy, dpi, dpi_save, frameon, vector_friendly, fontsize, figsize, color_map, format, facecolor, transparent, ipython_format)
        486     if isinstance(ipython_format, str):
        487         ipython_format = [ipython_format]
    --> 488     IPython.display.set_matplotlib_formats(*ipython_format)
        490 from matplotlib import rcParams
        492 self._vector_friendly = vector_friendly


    AttributeError: module 'IPython.display' has no attribute 'set_matplotlib_formats'



```python
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import KFold

# Set random seeds for reproducibility
import random
import numpy as np
import torch

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)
# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

```


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from torch.nn import Linear, Dropout
from torch_geometric.nn import TransformerConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

```


```python
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
```


```python
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import subgraph
```


```python
import gc
import os, sys
def memory_usgae():
    gc.collect()
    torch.cuda.empty_cache()
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024**3  # in GB

    print(f"Current memory usage: {memory_gb:.2f} GB")
```

# Load data


```python
# load the data
ad = sc.read_h5ad('../../Data/Cancer_cell_data_reprocessed/All_integrated.hallmark.harmony.h5ad')
ad
```




    AnnData object with n_obs × n_vars = 461835 × 3981
        obs: 'Project_ID', 'Primary_or_Metastatic', 'Final_cancer_type', 'Final_histological_subtype', 'Final_molecular_subtype', 'Final_tissue', 'Final_sample_id', 'Final_patient_age', 'Final_patient_stage', 'Final_patient_treatment', 'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt', 'Final_histological_subtype_backup', 'Final_patient_age_backup', 'Final_tissue_backup', 'Final_patient_treatment_backup', 'Final_patient_stage_backup', 'Classifier_label'
        var: 'mt', 'n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts'
        uns: 'Final_cancer_type_colors', 'Final_histological_subtype_colors', 'Final_molecular_subtype_colors', 'Final_patient_stage_colors', 'Final_patient_treatment_colors', 'Final_tissue_colors', 'Primary_or_Metastatic_colors', 'Project_ID_colors', 'log1p', 'neighbors', 'pca', 'umap'
        obsm: 'X_pca', 'X_pca_harmony', 'X_pca_harmony_Project_ID', 'X_pca_harmony_project_id', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'



Label unification:

Primary: Non-metastatic Local

Local: Metastatic Local

Distant: Metastatic distant


```python
# Map cancer type → its "primary tissue"
primary_tissue_map = {
    "Breast Cancer": "Breast",
    "Lung Cancer": "Lung",
    "Ovarian Cancer": "Ovary",
    "Colorectal Cancer": "Colon"
}

# Initialize as primary
ad.obs["source"] = "Primary"


# Loop through rows
for idx, row in ad.obs.iterrows():
    cancer_type = row["Final_cancer_type"]
    tissue = row["Final_tissue"]

    primary_site = primary_tissue_map.get(cancer_type, None)

    if row["Primary_or_Metastatic"] == "Primary":
        ad.obs.at[idx, "source"] = "Primary"

    else:  # Metastatic
        if tissue == primary_site:
            ad.obs.at[idx, "source"] = "local"
        else:
            ad.obs.at[idx, "source"] = "distant"
```

# Model


```python
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import TransformerConv, GATConv, SAGEConv


class scMeta(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, conv_type='TransformerConv', heads=4, dropout=0.3):
        """
        Flexible scMeta model with multiple GNN layer options.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden layer dimension
        num_classes : int
            Number of output classes
        conv_type : str
            Type of graph convolution layer. Options: 'TransformerConv', 'GATConv', 'SAGEConv'
            Default: 'TransformerConv'
        heads : int
            Number of attention heads (for TransformerConv and GATConv only)
            Default: 4
        dropout : float
            Dropout rate
            Default: 0.3
        """
        super(scMeta, self).__init__()
        
        self.conv_type = conv_type
        
        # Select GNN layer based on conv_type
        if conv_type == 'TransformerConv':
            self.conv1 = TransformerConv(
                in_channels=input_dim, 
                out_channels=hidden_dim, 
                heads=heads, 
                dropout=dropout
            )
            self.conv2 = TransformerConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim, 
                heads=1, 
                dropout=dropout
            )
            
        elif conv_type == 'GATConv':
            self.conv1 = GATConv(
                in_channels=input_dim, 
                out_channels=hidden_dim, 
                heads=heads, 
                dropout=dropout
            )
            self.conv2 = GATConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim, 
                heads=1, 
                dropout=dropout
            )
            
        elif conv_type == 'SAGEConv':
            # SAGEConv doesn't use heads parameter
            self.conv1 = SAGEConv(
                in_channels=input_dim, 
                out_channels=hidden_dim * heads  # Scale output to match other models
            )
            self.conv2 = SAGEConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim
            )
            self.dropout = Dropout(dropout)
            
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Choose from 'TransformerConv', 'GATConv', 'SAGEConv'")
        
        # Classifier head (same for all models)
        self.classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, return_embedding=False):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        return_embedding : bool
            If True, return both logits and embeddings
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits
        embeddings : torch.Tensor (optional)
            Node embeddings (only if return_embedding=True)
        """
        # First conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # SAGEConv needs manual dropout
        if self.conv_type == 'SAGEConv':
            x = self.dropout(x)
        
        # Second conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Classifier
        logits = self.classifier(x)
        
        if return_embedding:
            return logits, x  # logits, embeddings
        else:
            return logits
```


```python
def NT_Xent(embeddings, tau=0.5):
    device = embeddings.device
    z_i = F.normalize(embeddings, dim=1)
    z_j = F.normalize(embeddings[torch.randperm(z_i.size(0))], dim=1)

    logits = torch.mm(z_i, z_j.t()) / tau
    labels = torch.arange(z_i.size(0), device=device)
    loss = F.cross_entropy(logits, labels)
    return loss
```

# 5 fold CV (Per Cell)

## Data preperation


```python
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold

# AnnData copy
ad_sub = ad.copy()

# Features
if not isinstance(ad_sub.X, np.ndarray):
    X = torch.tensor(ad_sub.X.toarray(), dtype=torch.float32)
else:
    X = torch.tensor(ad_sub.X, dtype=torch.float32)

# Graph edges
adj = ad_sub.obsp["connectivities"].tocoo()
edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

# Labels
primary_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Primary").values
local_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Metastatic") & (ad_sub.obs["source"] == "local").values
distant_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Metastatic") & (ad_sub.obs["source"] == "distant").values

# 3-class labels
y_3class = np.full(ad_sub.n_obs, -1, dtype=int)
y_3class[primary_mask] = 0
y_3class[local_mask]   = 1
y_3class[distant_mask] = 2
y_3class_np = y_3class.copy()   # keep numpy for CV
y_3class = torch.tensor(y_3class, dtype=torch.long)

# Binary labels: 0 = Primary, 1 = Mets (Local + Distant)
y_binary = np.full(ad_sub.n_obs, -1, dtype=int)
y_binary[primary_mask] = 0
y_binary[local_mask]   = 1
y_binary[distant_mask] = 1
y_binary = torch.tensor(y_binary, dtype=torch.long)

# ============================================================
# PURE CELL-LEVEL 5-FOLD CV (TRAIN: 0/1/2, VAL: 0/1 only)
# ============================================================

n_cells = y_3class_np.shape[0]
all_idx = np.arange(n_cells)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = []

for fold_idx, (train_rel, val_rel) in enumerate(skf.split(all_idx, y_3class_np)):
    # Base train/val indices over ALL classes (0,1,2)
    train_idx_all = all_idx[train_rel]
    val_idx_all   = all_idx[val_rel]

    # TRAIN: use all classes (0,1,2)
    train_idx = train_idx_all

    # VAL: keep only Primary + Local (0,1), drop Distant (2)
    val_mask_pl = np.isin(y_3class_np[val_idx_all], [0, 1])
    val_idx = val_idx_all[val_mask_pl]

    # Sanity check
    y_train_fold = y_3class_np[train_idx]
    y_val_fold   = y_3class_np[val_idx]

    print(f"\nFold {fold_idx}")
    print("  Train unique classes:", np.unique(y_train_fold, return_counts=True))  # should include 2
    print("  Val   unique classes:", np.unique(y_val_fold, return_counts=True))    # should be [0 1]

    folds.append({
        "train_idx": train_idx,
        "val_idx": val_idx
    })

data = Data(
    x=X,
    edge_index=edge_index,
    y=y_3class
)
data.cv_folds = folds
data.y_binary = y_binary

print(f"\nPrepared {data.num_nodes} nodes")
print(f"Primary: {(y_3class==0).sum().item()} | Local: {(y_3class==1).sum().item()} | Distant: {(y_3class==2).sum().item()}")
print("Cell-level stratified 5-fold CV (train: 0/1/2, val: 0/1) done.")

```

    
    Fold 0
      Train unique classes: (array([0, 1, 2]), array([146001,  51641, 171826]))
      Val   unique classes: (array([0, 1]), array([36501, 12910]))
    
    Fold 1
      Train unique classes: (array([0, 1, 2]), array([146001,  51641, 171826]))
      Val   unique classes: (array([0, 1]), array([36501, 12910]))
    
    Fold 2
      Train unique classes: (array([0, 1, 2]), array([146002,  51640, 171826]))
      Val   unique classes: (array([0, 1]), array([36500, 12911]))
    
    Fold 3
      Train unique classes: (array([0, 1, 2]), array([146002,  51641, 171825]))
      Val   unique classes: (array([0, 1]), array([36500, 12910]))
    
    Fold 4
      Train unique classes: (array([0, 1, 2]), array([146002,  51641, 171825]))
      Val   unique classes: (array([0, 1]), array([36500, 12910]))
    
    Prepared 461835 nodes
    Primary: 182502 | Local: 64551 | Distant: 214782
    Cell-level stratified 5-fold CV (train: 0/1/2, val: 0/1) done.


## Training


```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def evaluate(model, x_val, y_binary, device):
    """
    Evaluate model on PRIMARY (0) vs NON-PRIMARY/METS (1).
    All cells are included (Primary, Local, Distant).
    
    Parameters:
    -----------
    model : nn.Module
        Model to evaluate
    x_val : torch.Tensor
        Validation features
    y_binary : torch.Tensor
        Binary labels (0=Primary, 1=Mets including Local+Distant)
    device : torch.device
        Device
    
    Returns:
    --------
    acc, f1, auc, auprc : float
        Evaluation metrics
    y_true, y_pred : np.array
        True and predicted labels
    """
    model.eval()
    
    # Self-loop edges
    N = x_val.size(0)
    edge_index_val = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    
    with torch.no_grad():
        logits, _ = model(x_val, edge_index=edge_index_val, return_embedding=True)
    
    # Get probabilities and predictions
    y_prob = torch.softmax(logits, dim=1)
    y_pred = logits.argmax(dim=1)
    
    # Convert to numpy
    y_true = y_binary.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # This is better for Primary vs (Local+Distant)
    y_prob_mets_combined = (y_prob[:, 1] + y_prob[:, 2]).cpu().numpy()
    
    # For binary evaluation, map predictions: 0→0, {1,2}→1
    y_pred_binary = (y_pred >= 1).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        auc = np.nan
        auprc = np.nan
    else:
        try:
            # Use combined probability (Local + Distant)
            auc = roc_auc_score(y_true, y_prob_mets_combined)
        except Exception as e:
            print(f"[ERROR] AUC: {e}")
            auc = np.nan
        
        try:
            auprc = average_precision_score(y_true, y_prob_mets_combined)
        except Exception as e:
            print(f"[ERROR] AUPRC: {e}")
            auprc = np.nan
    
    return acc, f1, auc, auprc, y_true, y_pred_binary
```


```python
import torch.optim as optim
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

def run_fold_cv(data, save_dir, epochs=100, seed=42, patience=10, gamma=0.1, tau=0.5, hidden_dim = 128, conv_type = 'TransformerConv'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_processed = True
    for fold, fold_dict in enumerate(data.cv_folds):
        best_model_path = os.path.join(save_dir, f"fold{fold + 1}_best_model.pt")
        if not os.path.exists(best_model_path):
            all_processed = False
    
    if all_processed:
        print("Already processed 5-fold CV with contrastive + CE loss.")
        return

    for fold, fold_dict in enumerate(data.cv_folds):
        
        memory_usgae()

        print(f"\n Fold {fold + 1}")

        train_idx = fold_dict["train_idx"]
        val_idx = fold_dict["val_idx"]

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        # Build train graph
        edge_index_train, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
        x_train = data.x[train_mask]
        y_train = data.y[train_mask]

        train_data = Data(
            x=x_train,
            edge_index=edge_index_train,
            y=y_train
        )

        x_val = data.x[val_idx].to(device)
        y_val = data.y[val_idx].to(device)
        
        y_binary_val = y_binary[val_idx]


        model = scMeta(
            input_dim=data.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=len(torch.unique(data.y)),
            conv_type=conv_type
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        best_acc = 0
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(save_dir, f"fold{fold + 1}_best_model.pt")
        
        # double check class distribution in val set
        # no distant mets should be here, no label 2
        y_val_counts = torch.bincount(data.y[val_idx])
        for label, count in enumerate(y_val_counts):
            print(f"Val label {label}: {count.item()} samples")

        # Pre-training eval
        acc, f1, auc, auprc, _, _ = evaluate(
            model, x_val, y_binary_val, device
        )
        print(f"Pre-training stats: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auc:.4f}, AUPRC={auprc:.4f}")

        for epoch in tqdm(range(epochs), desc=f"Fold {fold + 1}"):
            model.train()
            total_loss = 0

            loader = RandomNodeLoader(train_data, num_parts=100, shuffle=True)
            for batch in loader:
                batch = batch.to(device)
                logits, emb = model(batch.x, batch.edge_index, return_embedding=True)

                loss_ce = F.cross_entropy(logits, batch.y)
                loss_con = NT_Xent(emb, tau=tau)
                loss = loss_ce + gamma * loss_con

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Evaluate
            acc, f1, auc, auprc, _, _ = evaluate(
                model, x_val, y_binary_val, device
            )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auc:.4f}, AUPRC={auprc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" Early stopping at epoch {epoch + 1}")
                    break

        print(f"Best Acc for Fold {fold + 1}: {best_auc:.4f}")

    print("Finished 5-fold CV with contrastive + CE loss.")

```


```python
for gamma in  [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
    print('*'*10, gamma, '*'*10)
    for tau in [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]:
        print('-'*10, tau, '-'*10)
        for dim in [64, 128, 256, 512]:
            print('+'*10, dim, '+'*10)
            save_dir = '/scratch/gilbreth/wang3712/Metastasis_single_cell/Rerun_scMeta_5fold_CV_gamma'+str(gamma)+'_tau'+str(tau)+'_hidden'+str(dim)+'/'
            run_fold_cv(
                data,
                save_dir=save_dir,
                epochs=100,
                patience=10,
                gamma=gamma,
                tau = tau,
                hidden_dim=dim,
                conv_type='TransformerConv' # change this to GATConv or SAGEConv for other models
            )
```

## Evaluation

This is what is reported in the paper.

### TransformerConv


```python
df = pd.read_csv('/scratch/gilbreth/wang3712/Metastasis_single_cell_TransformerConv//summary_statistics.csv').sort_values(by='auc_mean')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>tau</th>
      <th>hidden_dim</th>
      <th>acc_mean</th>
      <th>acc_std</th>
      <th>acc_count</th>
      <th>f1_mean</th>
      <th>f1_std</th>
      <th>auc_mean</th>
      <th>auc_std</th>
      <th>auprc_mean</th>
      <th>auprc_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>0.2</td>
      <td>0.30</td>
      <td>256</td>
      <td>0.9590</td>
      <td>0.0234</td>
      <td>5</td>
      <td>0.9272</td>
      <td>0.0385</td>
      <td>0.9958</td>
      <td>0.0063</td>
      <td>0.9890</td>
      <td>0.0164</td>
    </tr>
    <tr>
      <th>216</th>
      <td>3.0</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9654</td>
      <td>0.0079</td>
      <td>5</td>
      <td>0.9373</td>
      <td>0.0131</td>
      <td>0.9975</td>
      <td>0.0001</td>
      <td>0.9935</td>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>264</th>
      <td>5.0</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9692</td>
      <td>0.0067</td>
      <td>5</td>
      <td>0.9437</td>
      <td>0.0114</td>
      <td>0.9978</td>
      <td>0.0001</td>
      <td>0.9945</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>281</th>
      <td>5.0</td>
      <td>0.70</td>
      <td>128</td>
      <td>0.9644</td>
      <td>0.0102</td>
      <td>5</td>
      <td>0.9359</td>
      <td>0.0172</td>
      <td>0.9979</td>
      <td>0.0017</td>
      <td>0.9949</td>
      <td>0.0039</td>
    </tr>
    <tr>
      <th>265</th>
      <td>5.0</td>
      <td>0.05</td>
      <td>128</td>
      <td>0.9658</td>
      <td>0.0052</td>
      <td>5</td>
      <td>0.9379</td>
      <td>0.0088</td>
      <td>0.9979</td>
      <td>0.0002</td>
      <td>0.9948</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>259</th>
      <td>4.0</td>
      <td>0.70</td>
      <td>512</td>
      <td>0.9691</td>
      <td>0.0038</td>
      <td>5</td>
      <td>0.9438</td>
      <td>0.0066</td>
      <td>0.9988</td>
      <td>0.0002</td>
      <td>0.9970</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>103</th>
      <td>0.1</td>
      <td>0.10</td>
      <td>512</td>
      <td>0.9723</td>
      <td>0.0058</td>
      <td>5</td>
      <td>0.9495</td>
      <td>0.0100</td>
      <td>0.9988</td>
      <td>0.0001</td>
      <td>0.9967</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>167</th>
      <td>0.5</td>
      <td>1.00</td>
      <td>512</td>
      <td>0.9722</td>
      <td>0.0036</td>
      <td>5</td>
      <td>0.9492</td>
      <td>0.0061</td>
      <td>0.9989</td>
      <td>0.0002</td>
      <td>0.9971</td>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0.2</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9706</td>
      <td>0.0113</td>
      <td>5</td>
      <td>0.9468</td>
      <td>0.0194</td>
      <td>0.9989</td>
      <td>0.0004</td>
      <td>0.9973</td>
      <td>0.0010</td>
    </tr>
    <tr>
      <th>263</th>
      <td>4.0</td>
      <td>1.00</td>
      <td>512</td>
      <td>0.9733</td>
      <td>0.0063</td>
      <td>5</td>
      <td>0.9512</td>
      <td>0.0109</td>
      <td>0.9989</td>
      <td>0.0001</td>
      <td>0.9971</td>
      <td>0.0004</td>
    </tr>
  </tbody>
</table>
<p>288 rows × 12 columns</p>
</div>




```python
df = df.sort_values("f1_mean", ascending=False)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>tau</th>
      <th>hidden_dim</th>
      <th>acc_mean</th>
      <th>acc_std</th>
      <th>acc_count</th>
      <th>f1_mean</th>
      <th>f1_std</th>
      <th>auc_mean</th>
      <th>auc_std</th>
      <th>auprc_mean</th>
      <th>auprc_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>0.01</td>
      <td>1.00</td>
      <td>256</td>
      <td>0.9752</td>
      <td>0.0028</td>
      <td>5</td>
      <td>0.9543</td>
      <td>0.0048</td>
      <td>0.9988</td>
      <td>0.0001</td>
      <td>0.9969</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.02</td>
      <td>0.30</td>
      <td>128</td>
      <td>0.9750</td>
      <td>0.0018</td>
      <td>5</td>
      <td>0.9539</td>
      <td>0.0031</td>
      <td>0.9985</td>
      <td>0.0002</td>
      <td>0.9963</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>170</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>256</td>
      <td>0.9749</td>
      <td>0.0065</td>
      <td>5</td>
      <td>0.9538</td>
      <td>0.0113</td>
      <td>0.9984</td>
      <td>0.0004</td>
      <td>0.9960</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.00</td>
      <td>0.10</td>
      <td>512</td>
      <td>0.9748</td>
      <td>0.0044</td>
      <td>5</td>
      <td>0.9536</td>
      <td>0.0077</td>
      <td>0.9986</td>
      <td>0.0004</td>
      <td>0.9964</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>169</th>
      <td>1.00</td>
      <td>0.05</td>
      <td>128</td>
      <td>0.9748</td>
      <td>0.0068</td>
      <td>5</td>
      <td>0.9536</td>
      <td>0.0117</td>
      <td>0.9983</td>
      <td>0.0002</td>
      <td>0.9957</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.01</td>
      <td>0.30</td>
      <td>256</td>
      <td>0.9631</td>
      <td>0.0087</td>
      <td>5</td>
      <td>0.9338</td>
      <td>0.0146</td>
      <td>0.9984</td>
      <td>0.0003</td>
      <td>0.9960</td>
      <td>0.0010</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1.00</td>
      <td>0.30</td>
      <td>128</td>
      <td>0.9626</td>
      <td>0.0059</td>
      <td>5</td>
      <td>0.9327</td>
      <td>0.0097</td>
      <td>0.9983</td>
      <td>0.0002</td>
      <td>0.9957</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>274</th>
      <td>5.00</td>
      <td>0.30</td>
      <td>256</td>
      <td>0.9619</td>
      <td>0.0116</td>
      <td>5</td>
      <td>0.9319</td>
      <td>0.0188</td>
      <td>0.9986</td>
      <td>0.0002</td>
      <td>0.9964</td>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>269</th>
      <td>5.00</td>
      <td>0.10</td>
      <td>128</td>
      <td>0.9612</td>
      <td>0.0071</td>
      <td>5</td>
      <td>0.9305</td>
      <td>0.0116</td>
      <td>0.9981</td>
      <td>0.0001</td>
      <td>0.9952</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0.20</td>
      <td>0.30</td>
      <td>256</td>
      <td>0.9590</td>
      <td>0.0234</td>
      <td>5</td>
      <td>0.9272</td>
      <td>0.0385</td>
      <td>0.9958</td>
      <td>0.0063</td>
      <td>0.9890</td>
      <td>0.0164</td>
    </tr>
  </tbody>
</table>
<p>288 rows × 12 columns</p>
</div>




```python
gamma, tau, dim = df.iloc[0]['gamma'], df.iloc[0]['tau'], df.iloc[0]['hidden_dim']
print(f"/scratch/gilbreth/wang3712/Metastasis_single_cell/Rerun_scMeta_5fold_CV_gamma{str(gamma)}_tau{str(tau)}_hidden{str(int(dim))}/")
```

    /scratch/gilbreth/wang3712/Metastasis_single_cell/Rerun_scMeta_5fold_CV_gamma0.01_tau1.0_hidden256/


### GATConv


```python
pd.read_csv('/scratch/gilbreth/wang3712/Metastasis_single_cell/_GATConv//summary_statistics.csv').sort_values(by='auc_mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>tau</th>
      <th>hidden_dim</th>
      <th>acc_mean</th>
      <th>acc_std</th>
      <th>acc_count</th>
      <th>f1_mean</th>
      <th>f1_std</th>
      <th>auc_mean</th>
      <th>auc_std</th>
      <th>auprc_mean</th>
      <th>auprc_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>273</th>
      <td>5.0</td>
      <td>0.30</td>
      <td>128</td>
      <td>0.9451</td>
      <td>0.0547</td>
      <td>5</td>
      <td>0.9000</td>
      <td>0.0990</td>
      <td>0.9791</td>
      <td>0.0420</td>
      <td>0.9564</td>
      <td>0.0855</td>
    </tr>
    <tr>
      <th>268</th>
      <td>5.0</td>
      <td>0.10</td>
      <td>64</td>
      <td>0.9671</td>
      <td>0.0048</td>
      <td>5</td>
      <td>0.9399</td>
      <td>0.0082</td>
      <td>0.9975</td>
      <td>0.0002</td>
      <td>0.9938</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>264</th>
      <td>5.0</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9693</td>
      <td>0.0028</td>
      <td>5</td>
      <td>0.9437</td>
      <td>0.0050</td>
      <td>0.9976</td>
      <td>0.0001</td>
      <td>0.9938</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>240</th>
      <td>4.0</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9681</td>
      <td>0.0054</td>
      <td>5</td>
      <td>0.9417</td>
      <td>0.0093</td>
      <td>0.9976</td>
      <td>0.0004</td>
      <td>0.9938</td>
      <td>0.0006</td>
    </tr>
    <tr>
      <th>244</th>
      <td>4.0</td>
      <td>0.10</td>
      <td>64</td>
      <td>0.9687</td>
      <td>0.0031</td>
      <td>5</td>
      <td>0.9427</td>
      <td>0.0053</td>
      <td>0.9976</td>
      <td>0.0001</td>
      <td>0.9940</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2.0</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9734</td>
      <td>0.0050</td>
      <td>5</td>
      <td>0.9510</td>
      <td>0.0087</td>
      <td>0.9982</td>
      <td>0.0001</td>
      <td>0.9954</td>
      <td>0.0004</td>
    </tr>
    <tr>
      <th>147</th>
      <td>0.5</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9786</td>
      <td>0.0025</td>
      <td>5</td>
      <td>0.9601</td>
      <td>0.0045</td>
      <td>0.9982</td>
      <td>0.0002</td>
      <td>0.9953</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>219</th>
      <td>3.0</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9774</td>
      <td>0.0073</td>
      <td>5</td>
      <td>0.9581</td>
      <td>0.0130</td>
      <td>0.9983</td>
      <td>0.0003</td>
      <td>0.9957</td>
      <td>0.0007</td>
    </tr>
    <tr>
      <th>267</th>
      <td>5.0</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9747</td>
      <td>0.0069</td>
      <td>5</td>
      <td>0.9534</td>
      <td>0.0121</td>
      <td>0.9983</td>
      <td>0.0001</td>
      <td>0.9958</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2.0</td>
      <td>0.50</td>
      <td>512</td>
      <td>0.9773</td>
      <td>0.0022</td>
      <td>5</td>
      <td>0.9577</td>
      <td>0.0039</td>
      <td>0.9983</td>
      <td>0.0001</td>
      <td>0.9956</td>
      <td>0.0003</td>
    </tr>
  </tbody>
</table>
<p>288 rows × 12 columns</p>
</div>



### SAGEConv


```python
pd.read_csv('/scratch/gilbreth/wang3712/Metastasis_single_cell/_SAGEConv//summary_statistics.csv').sort_values(by='auc_mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>tau</th>
      <th>hidden_dim</th>
      <th>acc_mean</th>
      <th>acc_std</th>
      <th>acc_count</th>
      <th>f1_mean</th>
      <th>f1_std</th>
      <th>auc_mean</th>
      <th>auc_std</th>
      <th>auprc_mean</th>
      <th>auprc_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>64</td>
      <td>0.9724</td>
      <td>0.0047</td>
      <td>5</td>
      <td>0.9493</td>
      <td>0.0084</td>
      <td>0.9980</td>
      <td>0.0005</td>
      <td>0.9951</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>216</th>
      <td>3.00</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9749</td>
      <td>0.0028</td>
      <td>5</td>
      <td>0.9535</td>
      <td>0.0049</td>
      <td>0.9981</td>
      <td>0.0004</td>
      <td>0.9953</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>264</th>
      <td>5.00</td>
      <td>0.05</td>
      <td>64</td>
      <td>0.9745</td>
      <td>0.0030</td>
      <td>5</td>
      <td>0.9527</td>
      <td>0.0053</td>
      <td>0.9981</td>
      <td>0.0003</td>
      <td>0.9952</td>
      <td>0.0008</td>
    </tr>
    <tr>
      <th>244</th>
      <td>4.00</td>
      <td>0.10</td>
      <td>64</td>
      <td>0.9731</td>
      <td>0.0014</td>
      <td>5</td>
      <td>0.9504</td>
      <td>0.0024</td>
      <td>0.9981</td>
      <td>0.0001</td>
      <td>0.9953</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>268</th>
      <td>5.00</td>
      <td>0.10</td>
      <td>64</td>
      <td>0.9716</td>
      <td>0.0029</td>
      <td>5</td>
      <td>0.9479</td>
      <td>0.0049</td>
      <td>0.9981</td>
      <td>0.0001</td>
      <td>0.9952</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>279</th>
      <td>5.00</td>
      <td>0.50</td>
      <td>512</td>
      <td>0.9751</td>
      <td>0.0043</td>
      <td>5</td>
      <td>0.9542</td>
      <td>0.0075</td>
      <td>0.9988</td>
      <td>0.0002</td>
      <td>0.9969</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.01</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9768</td>
      <td>0.0046</td>
      <td>5</td>
      <td>0.9571</td>
      <td>0.0081</td>
      <td>0.9989</td>
      <td>0.0001</td>
      <td>0.9971</td>
      <td>0.0002</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.10</td>
      <td>0.05</td>
      <td>512</td>
      <td>0.9760</td>
      <td>0.0053</td>
      <td>5</td>
      <td>0.9558</td>
      <td>0.0094</td>
      <td>0.9989</td>
      <td>0.0002</td>
      <td>0.9972</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>271</th>
      <td>5.00</td>
      <td>0.10</td>
      <td>512</td>
      <td>0.9770</td>
      <td>0.0030</td>
      <td>5</td>
      <td>0.9575</td>
      <td>0.0053</td>
      <td>0.9989</td>
      <td>0.0002</td>
      <td>0.9971</td>
      <td>0.0007</td>
    </tr>
    <tr>
      <th>255</th>
      <td>4.00</td>
      <td>0.50</td>
      <td>512</td>
      <td>0.9775</td>
      <td>0.0036</td>
      <td>5</td>
      <td>0.9584</td>
      <td>0.0065</td>
      <td>0.9989</td>
      <td>0.0003</td>
      <td>0.9971</td>
      <td>0.0007</td>
    </tr>
  </tbody>
</table>
<p>288 rows × 12 columns</p>
</div>



# 5 fold CV (Per Patient)

## Data preperation


```python
from sklearn.model_selection import StratifiedKFold

# ============================================================
# PREPARE DATA
# ============================================================

# AnnData copy
ad_sub = ad.copy()

# Features
if not isinstance(ad_sub.X, np.ndarray):
    X = torch.tensor(ad_sub.X.toarray(), dtype=torch.float32)
else:
    X = torch.tensor(ad_sub.X, dtype=torch.float32)

# Graph edges
adj = ad_sub.obsp["connectivities"].tocoo()
edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)

# Cell-level labels
primary_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Primary").values
local_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Metastatic") & (ad_sub.obs["source"] == "local").values
distant_mask = (ad_sub.obs["Primary_or_Metastatic"] == "Metastatic") & (ad_sub.obs["source"] == "distant").values

# 3-class labels
y_3class = np.full(ad_sub.n_obs, -1, dtype=int)
y_3class[primary_mask] = 0
y_3class[local_mask] = 1
y_3class[distant_mask] = 2
y_3class = torch.tensor(y_3class, dtype=torch.long)

# Binary labels
y_binary = np.full(ad_sub.n_obs, -1, dtype=int)
y_binary[primary_mask] = 0
y_binary[local_mask] = 1
y_binary[distant_mask] = 1
y_binary = torch.tensor(y_binary, dtype=torch.long)

# ============================================================
# PATIENT-LEVEL STRATIFIED CV (CORRECT VERSION)
# ============================================================

# Get patient IDs
patient_ids = ad_sub.obs["Final_sample_id"].values

# Create patient-level metadata
patient_df = pd.DataFrame({
    'patient_id': patient_ids,
    'cell_idx': np.arange(len(patient_ids)),
    'y_class': y_3class.numpy()
})

# For each patient, get their PRIMARY/LOCAL cells for stratification
patient_primary_local = patient_df[patient_df['y_class'].isin([0, 1])].copy()

# Group by patient
patient_summary = patient_primary_local.groupby('patient_id').agg({
    'y_class': lambda x: list(x),
    'cell_idx': lambda x: list(x)
}).reset_index()

# Create stratification label: 0 if has primary, 1 if has local only
def get_patient_label(y_classes):
    if 0 in y_classes:
        return 0  # Has primary
    elif 1 in y_classes:
        return 1  # Has local only
    else:
        return -1

patient_summary['strat_label'] = patient_summary['y_class'].apply(get_patient_label)

print(f"\n{'='*60}")
print("PATIENT-LEVEL STATISTICS")
print(f"{'='*60}")
print(f"Total unique patients: {len(np.unique(patient_ids))}")
print(f"Patients with primary/local cells: {len(patient_summary)}")
print(f"  Patients with primary: {(patient_summary['strat_label'] == 0).sum()}")
print(f"  Patients with local only: {(patient_summary['strat_label'] == 1).sum()}")

# ============================================================
# 5-FOLD CV AT PATIENT LEVEL
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

patient_ids_list = patient_summary['patient_id'].values
strat_labels = patient_summary['strat_label'].values

folds = []

for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(
    skf.split(patient_ids_list, strat_labels)
):
    # Get patient IDs for this fold
    train_patients = patient_ids_list[train_patient_idx]
    val_patients = patient_ids_list[val_patient_idx]
    
    # ============================================================
    # CRITICAL: Get cells based on patient assignment
    # ============================================================
    
    # For TRAINING patients: get ALL their cells (Primary + Local + Distant)
    train_all_cells = patient_df[patient_df['patient_id'].isin(train_patients)]['cell_idx'].values
    
    # For VALIDATION patients: get ONLY their Primary/Local cells (NO Distant)
    val_primary_local_cells = []
    for patient_id in val_patients:
        patient_cells = patient_summary[
            patient_summary['patient_id'] == patient_id
        ]['cell_idx'].iloc[0]
        val_primary_local_cells.extend(patient_cells)
    
    val_primary_local_cells = np.array(val_primary_local_cells)
    
    # Store fold
    folds.append({
        "train_idx": train_all_cells,
        "val_idx": val_primary_local_cells,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "n_train_cells": len(train_all_cells),
        "n_val_cells": len(val_primary_local_cells)
    })
    
    # Print fold statistics
    train_y = y_3class[train_all_cells].numpy()
    val_y = y_3class[val_primary_local_cells].numpy()
    
    print(f"\n{'='*40}")
    print(f"Fold {fold_idx}")
    print(f"{'='*40}")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"\nTraining cells: {len(train_all_cells)} (ALL cell types from training patients)")
    print(f"  Primary:  {np.sum(train_y==0):6d} cells")
    print(f"  Local:    {np.sum(train_y==1):6d} cells")
    print(f"  Distant:  {np.sum(train_y==2):6d} cells")
    print(f"\nValidation cells: {len(val_primary_local_cells)} (Primary + Local ONLY)")
    print(f"  Primary:  {np.sum(val_y==0):6d} cells")
    print(f"  Local:    {np.sum(val_y==1):6d} cells")
    print(f"  Distant:  {np.sum(val_y==2):6d} cells (should be 0)")

# ============================================================
# CREATE DATA OBJECT
# ============================================================

data = Data(
    x=X,
    edge_index=edge_index,
    y=y_3class
)
data.cv_folds = folds
data.y_binary = y_binary

print(f"\n{'='*60}")
print("FINAL DATA SUMMARY")
print(f"{'='*60}")
print(f"Total nodes: {data.num_nodes}")
print(f"Primary: {(y_3class==0).sum().item()} | Local: {(y_3class==1).sum().item()} | Distant: {(y_3class==2).sum().item()}")
print("\nPatient-level stratified 5-fold CV done.")

# ============================================================
# VERIFICATION
# ============================================================

print(f"\n{'='*60}")
print("VERIFICATION")
print(f"{'='*60}")

for fold_idx, fold in enumerate(folds):
    train_patients_set = set(fold['train_patients'])
    val_patients_set = set(fold['val_patients'])
    
    # Check for patient overlap
    overlap = train_patients_set & val_patients_set
    assert len(overlap) == 0, f"Fold {fold_idx}: Patient overlap detected! {overlap}"
    
    # Verify validation set has NO distant cells
    val_y = y_3class[fold['val_idx']].numpy()
    n_distant_in_val = np.sum(val_y == 2)
    assert n_distant_in_val == 0, \
        f"Fold {fold_idx}: Validation has {n_distant_in_val} distant cells (should be 0)"
    
    # Verify all validation cells are primary or local
    assert np.all((val_y == 0) | (val_y == 1)), \
        f"Fold {fold_idx}: Validation should only have primary (0) or local (1) cells"
    
    # Check that validation patient's distant cells are NOT in training
    train_y = y_3class[fold['train_idx']].numpy()
    train_patient_ids = patient_ids[fold['train_idx']]
    
    # Ensure no validation patients appear in training data
    for val_patient in val_patients_set:
        assert val_patient not in train_patient_ids, \
            f"Fold {fold_idx}: Validation patient {val_patient} found in training!"
    
    print(f"✓ Fold {fold_idx}: Clean split - validation patients' distant cells excluded from training")

print("\n✓ All verification checks passed!")
print("✓ No patient appears in both train and val")
print("✓ Validation sets contain ONLY primary and local cells")
print("✓ Validation patients' distant cells are EXCLUDED from training")
print("✓ Training includes distant cells ONLY from training patients")


```

    
    ============================================================
    PATIENT-LEVEL STATISTICS
    ============================================================
    Total unique patients: 370
    Patients with primary/local cells: 370
      Patients with primary: 258
      Patients with local only: 50
    
    ========================================
    Fold 0
    ========================================
    Training patients: 296
    Validation patients: 74
    
    Training cells: 343379 (ALL cell types from training patients)
      Primary:  138058 cells
      Local:     44076 cells
      Distant:  161245 cells
    
    Validation cells: 64919 (Primary + Local ONLY)
      Primary:   44444 cells
      Local:     20475 cells
      Distant:       0 cells (should be 0)
    
    ========================================
    Fold 1
    ========================================
    Training patients: 296
    Validation patients: 74
    
    Training cells: 389089 (ALL cell types from training patients)
      Primary:  151805 cells
      Local:     54656 cells
      Distant:  182628 cells
    
    Validation cells: 40592 (Primary + Local ONLY)
      Primary:   30697 cells
      Local:      9895 cells
      Distant:       0 cells (should be 0)
    
    ========================================
    Fold 2
    ========================================
    Training patients: 296
    Validation patients: 74
    
    Training cells: 372781 (ALL cell types from training patients)
      Primary:  146052 cells
      Local:     53895 cells
      Distant:  172834 cells
    
    Validation cells: 47106 (Primary + Local ONLY)
      Primary:   36450 cells
      Local:     10656 cells
      Distant:       0 cells (should be 0)
    
    ========================================
    Fold 3
    ========================================
    Training patients: 296
    Validation patients: 74
    
    Training cells: 370036 (ALL cell types from training patients)
      Primary:  141638 cells
      Local:     53723 cells
      Distant:  174675 cells
    
    Validation cells: 51692 (Primary + Local ONLY)
      Primary:   40864 cells
      Local:     10828 cells
      Distant:       0 cells (should be 0)
    
    ========================================
    Fold 4
    ========================================
    Training patients: 296
    Validation patients: 74
    
    Training cells: 372055 (ALL cell types from training patients)
      Primary:  152455 cells
      Local:     51854 cells
      Distant:  167746 cells
    
    Validation cells: 42744 (Primary + Local ONLY)
      Primary:   30047 cells
      Local:     12697 cells
      Distant:       0 cells (should be 0)
    
    ============================================================
    FINAL DATA SUMMARY
    ============================================================
    Total nodes: 461835
    Primary: 182502 | Local: 64551 | Distant: 214782
    
    Patient-level stratified 5-fold CV done.
    
    ============================================================
    VERIFICATION
    ============================================================
    ✓ Fold 0: Clean split - validation patients' distant cells excluded from training
    ✓ Fold 1: Clean split - validation patients' distant cells excluded from training
    ✓ Fold 2: Clean split - validation patients' distant cells excluded from training
    ✓ Fold 3: Clean split - validation patients' distant cells excluded from training
    ✓ Fold 4: Clean split - validation patients' distant cells excluded from training
    
    ✓ All verification checks passed!
    ✓ No patient appears in both train and val
    ✓ Validation sets contain ONLY primary and local cells
    ✓ Validation patients' distant cells are EXCLUDED from training
    ✓ Training includes distant cells ONLY from training patients



```python
def verify_validation_patient_homogeneity(data, folds, patient_ids, verbose=True):
    """
    Verify that all validation patients have homogeneous cell labels.
    
    Parameters:
    -----------
    data : Data
        PyG Data object with y and y_binary
    folds : list
        List of fold dictionaries
    patient_ids : np.array
        Patient IDs for all cells
    verbose : bool
        If True, print detailed information
    
    Returns:
    --------
    all_homogeneous : bool
        True if all validation patients are homogeneous
    report : dict
        Detailed report for each fold
    """
    
    print(f"\n{'='*60}")
    print("VALIDATION PATIENT HOMOGENEITY VERIFICATION")
    print(f"{'='*60}")
    
    all_homogeneous = True
    report = {}
    
    for fold_idx, fold in enumerate(folds):
        val_idx = fold['val_idx']
        val_patients = patient_ids[val_idx]
        val_y_3class = data.y[val_idx].numpy()
        val_y_binary = data.y_binary[val_idx].numpy()
        
        unique_val_patients = np.unique(val_patients)
        
        fold_report = {
            'n_patients': len(unique_val_patients),
            'homogeneous_patients': [],
            'heterogeneous_patients': [],
            'patient_details': []
        }
        
        print(f"\n{'='*40}")
        print(f"Fold {fold_idx}")
        print(f"{'='*40}")
        print(f"Validation patients: {len(unique_val_patients)}")
        
        # Check each patient
        heterogeneous_count = 0
        
        for patient_id in unique_val_patients:
            patient_mask = val_patients == patient_id
            patient_labels_3class = val_y_3class[patient_mask]
            patient_labels_binary = val_y_binary[patient_mask]
            
            # Get unique labels
            unique_3class = np.unique(patient_labels_3class)
            unique_binary = np.unique(patient_labels_binary)
            
            # Check homogeneity
            is_homogeneous_3class = len(unique_3class) == 1
            is_homogeneous_binary = len(unique_binary) == 1
            
            patient_info = {
                'patient_id': patient_id,
                'n_cells': np.sum(patient_mask),
                '3class_labels': unique_3class.tolist(),
                'binary_labels': unique_binary.tolist(),
                'is_homogeneous': is_homogeneous_3class and is_homogeneous_binary
            }
            
            fold_report['patient_details'].append(patient_info)
            
            if patient_info['is_homogeneous']:
                fold_report['homogeneous_patients'].append(patient_id)
            else:
                fold_report['heterogeneous_patients'].append(patient_id)
                heterogeneous_count += 1
                all_homogeneous = False
                
                # Print warning for heterogeneous patients
                print(f"\n⚠️  WARNING: Patient {patient_id} has MIXED labels!")
                print(f"    Cells: {patient_info['n_cells']}")
                print(f"    3-class labels: {unique_3class} {['Primary' if x==0 else 'Local' if x==1 else 'Distant' for x in unique_3class]}")
                print(f"    Binary labels: {unique_binary}")
                
                # Show detailed breakdown
                label_counts_3class = np.bincount(patient_labels_3class, minlength=3)
                print(f"    Breakdown:")
                print(f"      Primary (0): {label_counts_3class[0]} cells")
                print(f"      Local (1):   {label_counts_3class[1]} cells")
                print(f"      Distant (2): {label_counts_3class[2]} cells")
        
        # Summary for this fold
        if heterogeneous_count == 0:
            print(f"\n✓ All {len(unique_val_patients)} validation patients are HOMOGENEOUS")
        else:
            print(f"\n✗ Found {heterogeneous_count} HETEROGENEOUS patients out of {len(unique_val_patients)}")
        
        # Distribution summary
        primary_patients = sum([1 for p in fold_report['patient_details'] 
                               if p['is_homogeneous'] and 0 in p['3class_labels']])
        local_patients = sum([1 for p in fold_report['patient_details'] 
                             if p['is_homogeneous'] and 1 in p['3class_labels']])
        
        print(f"\nHomogeneous patient distribution:")
        print(f"  Primary only: {primary_patients} patients")
        print(f"  Local only:   {local_patients} patients")
        
        report[fold_idx] = fold_report
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_patients = sum([r['n_patients'] for r in report.values()])
    total_heterogeneous = sum([len(r['heterogeneous_patients']) for r in report.values()])
    
    if all_homogeneous:
        print(f"✓✓✓ SUCCESS: All {total_patients} validation patients across all folds are HOMOGENEOUS!")
        print(f"    Each patient has cells with a single label (all Primary OR all Local)")
    else:
        print(f"✗✗✗ FAILED: Found {total_heterogeneous} heterogeneous patients out of {total_patients}")
        print(f"    Some patients have mixed labels - this violates the assumption!")
    
    return all_homogeneous, report


# ============================================================
# RUN VERIFICATION
# ============================================================

all_homogeneous, detailed_report = verify_validation_patient_homogeneity(
    data=data,
    folds=folds,
    patient_ids=patient_ids,
    verbose=True
)

# ============================================================
# DETAILED BREAKDOWN (OPTIONAL)
# ============================================================

def print_detailed_patient_breakdown(report, fold_idx=0):
    """Print detailed breakdown for a specific fold."""
    
    fold_report = report[fold_idx]
    
    print(f"\n{'='*60}")
    print(f"DETAILED BREAKDOWN - FOLD {fold_idx}")
    print(f"{'='*60}")
    
    # Create summary table
    patient_data = []
    for p in fold_report['patient_details']:
        label_str = f"Primary" if 0 in p['3class_labels'] else f"Local" if 1 in p['3class_labels'] else "Mixed"
        if len(p['3class_labels']) > 1:
            label_str = f"MIXED: {p['3class_labels']}"
        
        patient_data.append({
            'Patient ID': p['patient_id'],
            'N Cells': p['n_cells'],
            'Label': label_str,
            'Homogeneous': '✓' if p['is_homogeneous'] else '✗'
        })
    
    df = pd.DataFrame(patient_data)
    print(df.to_string(index=False))
    
    # Summary stats
    print(f"\nSummary:")
    print(f"  Homogeneous: {len(fold_report['homogeneous_patients'])}")
    print(f"  Heterogeneous: {len(fold_report['heterogeneous_patients'])}")

# Example: Print detailed breakdown for fold 0
print_detailed_patient_breakdown(detailed_report, fold_idx=0)

# ============================================================
# ADDITIONAL CHECK: Cell-level verification
# ============================================================

def check_cell_level_consistency(data, folds, patient_ids):
    """
    Additional check: Verify at cell level that validation has no distant cells
    and that binary labels match expectations.
    """
    
    print(f"\n{'='*60}")
    print("CELL-LEVEL CONSISTENCY CHECK")
    print(f"{'='*60}")
    
    for fold_idx, fold in enumerate(folds):
        val_idx = fold['val_idx']
        
        val_y_3class = data.y[val_idx].numpy()
        val_y_binary = data.y_binary[val_idx].numpy()
        
        # Check 1: No distant cells
        n_distant = np.sum(val_y_3class == 2)
        
        # Check 2: Binary labels are consistent with 3-class
        # If 3-class is 0 → binary should be 0
        # If 3-class is 1 → binary should be 1
        primary_mask = val_y_3class == 0
        local_mask = val_y_3class == 1
        
        binary_consistent = (
            np.all(val_y_binary[primary_mask] == 0) and
            np.all(val_y_binary[local_mask] == 1)
        )
        
        print(f"\nFold {fold_idx}:")
        print(f"  Total validation cells: {len(val_idx)}")
        print(f"  Primary cells (0): {np.sum(primary_mask)}")
        print(f"  Local cells (1):   {np.sum(local_mask)}")
        print(f"  Distant cells (2): {n_distant} {'✓ (should be 0)' if n_distant == 0 else '✗ ERROR!'}")
        print(f"  Binary consistency: {'✓' if binary_consistent else '✗ ERROR!'}")
        
        # Assert critical conditions
        assert n_distant == 0, f"Fold {fold_idx} has {n_distant} distant cells in validation!"
        assert binary_consistent, f"Fold {fold_idx} has inconsistent binary labels!"

check_cell_level_consistency(data, folds, patient_ids)

print(f"\n{'='*60}")
print("✓ ALL VERIFICATION CHECKS COMPLETE")
print(f"{'='*60}")
```

    
    ============================================================
    VALIDATION PATIENT HOMOGENEITY VERIFICATION
    ============================================================
    
    ========================================
    Fold 0
    ========================================
    Validation patients: 62
    
    ✓ All 62 validation patients are HOMOGENEOUS
    
    Homogeneous patient distribution:
      Primary only: 52 patients
      Local only:   10 patients
    
    ========================================
    Fold 1
    ========================================
    Validation patients: 62
    
    ✓ All 62 validation patients are HOMOGENEOUS
    
    Homogeneous patient distribution:
      Primary only: 52 patients
      Local only:   10 patients
    
    ========================================
    Fold 2
    ========================================
    Validation patients: 62
    
    ✓ All 62 validation patients are HOMOGENEOUS
    
    Homogeneous patient distribution:
      Primary only: 52 patients
      Local only:   10 patients
    
    ========================================
    Fold 3
    ========================================
    Validation patients: 61
    
    ✓ All 61 validation patients are HOMOGENEOUS
    
    Homogeneous patient distribution:
      Primary only: 51 patients
      Local only:   10 patients
    
    ========================================
    Fold 4
    ========================================
    Validation patients: 61
    
    ✓ All 61 validation patients are HOMOGENEOUS
    
    Homogeneous patient distribution:
      Primary only: 51 patients
      Local only:   10 patients
    
    ============================================================
    OVERALL SUMMARY
    ============================================================
    ✓✓✓ SUCCESS: All 308 validation patients across all folds are HOMOGENEOUS!
        Each patient has cells with a single label (all Primary OR all Local)
    
    ============================================================
    DETAILED BREAKDOWN - FOLD 0
    ============================================================
                            Patient ID  N Cells   Label Homogeneous
                                38FE7L       43 Primary           ✓
                                3CCF1L     1161   Local           ✓
                                3E5CFL     1005 Primary           ✓
                                 BC_14     3297 Primary           ✓
                          BIOKEY_11-On      163 Primary           ✓
                          BIOKEY_15-On      105 Primary           ✓
                          BIOKEY_17-On      567 Primary           ✓
                         BIOKEY_19-Pre      342 Primary           ✓
                          BIOKEY_2-Pre      181 Primary           ✓
                          BIOKEY_21-On     2974 Primary           ✓
                         BIOKEY_30-Pre     2733 Primary           ✓
                          BIOKEY_36-On       34 Primary           ✓
                          BIOKEY_4-Pre      440 Primary           ✓
                          BIOKEY_5-Pre       65 Primary           ✓
                          BIOKEY_7-Pre       31 Primary           ✓
                                  C113      721 Primary           ✓
                                  C124      307 Primary           ✓
                                  C132      150 Primary           ✓
                                  C143      445 Primary           ✓
                                  C144      278 Primary           ✓
                                  C146     1148 Primary           ✓
                                  C147      842 Primary           ✓
                                  C149      245 Primary           ✓
                                  C152      465 Primary           ✓
                                  C158      120 Primary           ✓
                                  C166      136 Primary           ✓
                                  C171      376 Primary           ✓
                               CID4067     2219 Primary           ✓
                               CID4471      191 Primary           ✓
                              CID44971      569 Primary           ✓
                              CID45171      561 Primary           ✓
                               CID4523      855 Primary           ✓
                                 CRC_2     1554 Primary           ✓
      Goveia_Carmeliet_2020_patient_40       59 Primary           ✓
                        He_Fan_2021_P1      668 Primary           ✓
                        He_Fan_2021_P4      285 Primary           ✓
                    Kim_Lee_2020_P0019       66 Primary           ✓
                    Kim_Lee_2020_P0028      560 Primary           ✓
                                  LC_2      172 Primary           ✓
                                  LC_5     1658 Primary           ✓
    Lambrechts_Thienpont_2018_6149v2_4     1448 Primary           ✓
      Lambrechts_Thienpont_2018_6653_7      363 Primary           ✓
          Laughney_Massague_2020_LX679      210 Primary           ✓
                 Leader_Merad_2021_406      470 Primary           ✓
                 Leader_Merad_2021_408      408 Primary           ✓
                 Leader_Merad_2021_569       38 Primary           ✓
                                RU1108    10580 Primary           ✓
                                RU1229     3259 Primary           ✓
                                 SMC11      152 Primary           ✓
                                 SMC15      187 Primary           ✓
                       SPECTRUM-OV-002     1338   Local           ✓
                       SPECTRUM-OV-009     4081   Local           ✓
                       SPECTRUM-OV-025        1 Primary           ✓
                       SPECTRUM-OV-026     4651   Local           ✓
                       SPECTRUM-OV-051      402 Primary           ✓
                       SPECTRUM-OV-080       66   Local           ✓
                       SPECTRUM-OV-107     1373   Local           ✓
                       SPECTRUM-OV-118     5080   Local           ✓
                                    T7     1004   Local           ✓
                          UKIM-V-2_P15      312   Local           ✓
                             UKIM-V_P3      296 Primary           ✓
                                 s0107     1409   Local           ✓
    
    Summary:
      Homogeneous: 62
      Heterogeneous: 0
    
    ============================================================
    CELL-LEVEL CONSISTENCY CHECK
    ============================================================
    
    Fold 0:
      Total validation cells: 64919
      Primary cells (0): 44444
      Local cells (1):   20475
      Distant cells (2): 0 ✓ (should be 0)
      Binary consistency: ✓
    
    Fold 1:
      Total validation cells: 40592
      Primary cells (0): 30697
      Local cells (1):   9895
      Distant cells (2): 0 ✓ (should be 0)
      Binary consistency: ✓
    
    Fold 2:
      Total validation cells: 47106
      Primary cells (0): 36450
      Local cells (1):   10656
      Distant cells (2): 0 ✓ (should be 0)
      Binary consistency: ✓
    
    Fold 3:
      Total validation cells: 51692
      Primary cells (0): 40864
      Local cells (1):   10828
      Distant cells (2): 0 ✓ (should be 0)
      Binary consistency: ✓
    
    Fold 4:
      Total validation cells: 42744
      Primary cells (0): 30047
      Local cells (1):   12697
      Distant cells (2): 0 ✓ (should be 0)
      Binary consistency: ✓
    
    ============================================================
    ✓ ALL VERIFICATION CHECKS COMPLETE
    ============================================================


## Training


```python

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def evaluate(model, x_val, y_binary, device):
    """
    Evaluate model on PRIMARY (0) vs NON-PRIMARY/METS (1).
    All cells are included (Primary, Local, Distant).
    
    Parameters:
    -----------
    model : nn.Module
        Model to evaluate
    x_val : torch.Tensor
        Validation features
    y_binary : torch.Tensor
        Binary labels (0=Primary, 1=Mets including Local+Distant)
    device : torch.device
        Device
    
    Returns:
    --------
    acc, f1, auc, auprc : float
        Evaluation metrics
    y_true, y_pred : np.array
        True and predicted labels
    """
    model.eval()
    
    # Self-loop edges
    N = x_val.size(0)
    edge_index_val = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    
    with torch.no_grad():
        logits, _ = model(x_val, edge_index=edge_index_val, return_embedding=True)
    
    # Get probabilities and predictions
    y_prob = torch.softmax(logits, dim=1)
    y_pred = logits.argmax(dim=1)
    
    # Convert to numpy
    y_true = y_binary.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # This is better for Primary vs (Local+Distant)
    y_prob_mets_combined = (y_prob[:, 1] + y_prob[:, 2]).cpu().numpy()
    
    # For binary evaluation, map predictions: 0→0, {1,2}→1
    y_pred_binary = (y_pred >= 1).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        auc = np.nan
        auprc = np.nan
    else:
        try:
            # Use combined probability (Local + Distant)
            auc = roc_auc_score(y_true, y_prob_mets_combined)
        except Exception as e:
            print(f"[ERROR] AUC: {e}")
            auc = np.nan
        
        try:
            auprc = average_precision_score(y_true, y_prob_mets_combined)
        except Exception as e:
            print(f"[ERROR] AUPRC: {e}")
            auprc = np.nan
    
    return acc, f1, auc, auprc, y_true, y_pred_binary
```


```python
def get_patient_level_statistics(data, fold, patient_ids):
    """
    Get patient-level statistics for a fold.
    
    Parameters:
    -----------
    data : Data
        Full PyG Data object
    fold : dict
        Fold definition with train_idx, val_idx, train_patients, val_patients
    patient_ids : np.array
        Patient IDs for all cells
    
    Returns:
    --------
    stats : dict
        Patient-level statistics
    """
    train_idx = fold['train_idx']
    val_idx = fold['val_idx']
    
    # Get patient IDs for train and val
    train_patients = patient_ids[train_idx]
    val_patients = patient_ids[val_idx]
    
    # Get labels
    y_train = data.y[train_idx].numpy()
    y_val = data.y[val_idx].numpy()
    
    # Count patients by type in training set
    train_patient_counts = {}
    for label, label_name in [(0, 'Primary'), (1, 'Local'), (2, 'Distant')]:
        label_mask = y_train == label
        patients_with_label = np.unique(train_patients[label_mask])
        cells_with_label = np.sum(label_mask)
        train_patient_counts[label_name] = {
            'n_patients': len(patients_with_label),
            'n_cells': cells_with_label
        }
    
    # Count patients by type in validation set
    val_patient_counts = {}
    for label, label_name in [(0, 'Primary'), (1, 'Local'), (2, 'Distant')]:
        label_mask = y_val == label
        patients_with_label = np.unique(val_patients[label_mask])
        cells_with_label = np.sum(label_mask)
        val_patient_counts[label_name] = {
            'n_patients': len(patients_with_label),
            'n_cells': cells_with_label
        }
    
    return {
        'train': train_patient_counts,
        'val': val_patient_counts,
        'train_patients': train_patients,
        'val_patients': val_patients
    }

def calculate_patient_level_metrics(model, x_val, y_binary, patient_ids_val, device):
    """
    Calculate patient-level metrics using majority vote.
    
    IMPORTANT: 
    - Validation patients have homogeneous labels (all Primary OR all Local)
    - TRUE labels are binary (0=Primary, 1=Local/Mets)
    - PREDICTED labels can be 0, 1, or 2 (model outputs 3 classes)
    - Score = pct_local + pct_distant (combined Mets probability)
    
    Parameters:
    -----------
    model : nn.Module
        Model to evaluate
    x_val : torch.Tensor
        Validation features
    y_binary : torch.Tensor
        Binary labels (0=Primary, 1=Mets) - CHANGED from y_val!
    patient_ids_val : np.array
        Patient IDs for validation cells
    device : torch.device
        Device
    
    Returns:
    --------
    metrics : dict
        Patient-level metrics
    """
    model.eval()
    N = x_val.size(0)
    edge_index_val = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    
    with torch.no_grad():
        logits, _ = model(x_val, edge_index=edge_index_val, return_embedding=True)
    
    # Cell-level predictions (can be 0, 1, or 2)
    y_pred_cells = logits.argmax(dim=1).cpu().numpy()
    y_true_cells = y_binary.cpu().numpy()  # ✅ Binary labels (0 or 1)
    
    # Aggregate to patient level
    unique_patients = np.unique(patient_ids_val)
    
    patient_results = []
    
    for patient_id in unique_patients:
        patient_mask = patient_ids_val == patient_id
        
        # Get true label for patient (should be consistent for all cells)
        patient_true = y_true_cells[patient_mask][0]  # Binary: 0 or 1
        
        # Verify homogeneity (all cells should have same true label)
        assert np.all(y_true_cells[patient_mask] == patient_true), \
            f"Patient {patient_id} has heterogeneous labels! This should not happen."
        
        # Count predictions per class (predictions can be 0, 1, or 2)
        patient_preds = y_pred_cells[patient_mask]
        n_cells = len(patient_preds)
        
        # Calculate percentages for each predicted class
        pred_counts = np.bincount(patient_preds, minlength=3)
        pct_primary = (pred_counts[0] / n_cells) * 100
        pct_local = (pred_counts[1] / n_cells) * 100
        pct_distant = (pred_counts[2] / n_cells) * 100
        
        # Patient-level prediction: Primary (0) vs Mets (1)
        # If majority of cells predicted as Primary → 0, otherwise → 1
        patient_pred = 0 if (pct_primary > (pct_local+pct_distant)) else 1
        
        patient_results.append({
            'patient_id': patient_id,
            'true_label': patient_true,  # Binary: 0 or 1
            'pred_label': patient_pred,   # Binary: 0 or 1
            'pct_primary': pct_primary,
            'pct_local': pct_local,
            'pct_distant': pct_distant,
            'n_cells': n_cells
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(patient_results)
    
    # All patients should already be Primary (0) or Local (1)
    # No filtering needed!
    if len(df) == 0:
        return {
            'acc': np.nan,
            'f1': np.nan,
            'auc': np.nan,
            'auprc': np.nan,
            'n_patients': 0
        }
    
    # Binary metrics
    y_true_bin = df['true_label'].values  # Already binary: 0 or 1
    y_pred_bin = df['pred_label'].values  # Already binary: 0 or 1
    
    # Combined Mets probability (Local + Distant)
    y_scores = (df['pct_local'] + df['pct_distant']).values
    
    # Verify all labels are binary
    assert np.all(np.isin(y_true_bin, [0, 1])), "True labels should only be 0 or 1!"
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true_bin, y_scores)
    except:
        auc = np.nan
    
    try:
        auprc = average_precision_score(y_true_bin, y_scores)
    except:
        auprc = np.nan
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    return {
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'auprc': auprc,
        'n_patients': len(df),
        'patient_results': df
    }


def calculate_majority_class_prior(y_val, patient_ids_val):
    """
    Calculate prior metrics using majority class prediction.
    
    FIXED: Correct F1 calculation and AUC/AUPRC for majority baseline.
    """
    
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
    
    # Determine majority class at PATIENT level
    unique_patients = np.unique(patient_ids_val)
    patient_true_labels = []
    
    for patient_id in unique_patients:
        patient_mask = patient_ids_val == patient_id
        patient_true = y_val[patient_mask][0]
        patient_true_labels.append(patient_true)
    
    patient_true_labels = np.array(patient_true_labels)
    
    # Find majority class
    class_counts = np.bincount(patient_true_labels)
    majority_class = np.argmax(class_counts)
    
    print(f"\nClass distribution in validation:")
    print(f"  Primary (0): {np.sum(patient_true_labels == 0)} patients")
    print(f"  Local (1):   {np.sum(patient_true_labels == 1)} patients")
    print(f"  Majority class: {'Primary' if majority_class == 0 else 'Local'} ({majority_class})")
    
    # Predict everything as majority class
    y_pred_majority = np.full(len(patient_true_labels), majority_class)
    
    # For AUC/AUPRC, we need scores for the POSITIVE class (Local = 1)
    if majority_class == 0:
        # Predicting Primary (0), so score for Local (1) = 0%
        y_scores = np.full(len(patient_true_labels), 0.0)  # 0% Local
    else:
        # Predicting Local (1), so score for Local (1) = 100%
        y_scores = np.full(len(patient_true_labels), 100.0)  # 100% Local
    
    # Calculate metrics
    y_true = patient_true_labels
    y_pred = y_pred_majority
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC/AUPRC for constant predictions
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5  # Random guess baseline
    
    try:
        auprc = average_precision_score(y_true, y_scores)
    except:
        # AUPRC baseline = proportion of positive class
        auprc = np.mean(y_true == 1)
    
    return {
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'auprc': auprc,
        'n_patients': len(patient_true_labels),
        'majority_class': majority_class,
        'baseline_type': 'majority_class'
    }

def train_one_fold(
    data,
    fold,
    patient_ids,
    tau,
    gamma,
    device,
    epochs=100,
    patience=10,
    lr=1e-4,
    hidden_dim=128,
    num_parts=100,
    save_model=True,
    conv_type='TransformerConv',
    model_save_dir="./models"
):
    """
    Train model on one fold with specific hyperparameters.
    """

    
    # Get train and val indices
    train_idx = fold['train_idx']
    val_idx = fold['val_idx']
    
    # ============================================================
    # PRINT DETAILED STATISTICS
    # ============================================================
    
    print(f"\n{'='*60}")
    print("FOLD STATISTICS")
    print(f"{'='*60}")
    
    # Get patient-level statistics
    stats = get_patient_level_statistics(data, fold, patient_ids)
    
    # Print training set breakdown
    print("\n TRAINING SET:")
    print(f"  Total cells: {len(train_idx)}")
    for label_name in ['Primary', 'Local', 'Distant']:
        info = stats['train'][label_name]
        print(f"  {info['n_patients']:3d} {label_name:8s} patients ({info['n_cells']:6d} cells)")
    
    # Print validation set breakdown
    print("\n VALIDATION SET:")
    print(f"  Total cells: {len(val_idx)}")
    for label_name in ['Primary', 'Local', 'Distant']:
        info = stats['val'][label_name]
        if info['n_cells'] > 0:
            print(f"  {info['n_patients']:3d} {label_name:8s} patients ({info['n_cells']:6d} cells)")
    
    # ============================================================
    # BUILD TRAINING SUBGRAPH
    # ============================================================
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    
    edge_index_train, _ = subgraph(train_mask, data.edge_index, relabel_nodes=True)
    x_train = data.x[train_mask]
    y_train = data.y[train_mask]
    
    train_data = Data(
        x=x_train,
        edge_index=edge_index_train,
        y=y_train
    )
    
    # Validation data
    x_val = data.x[val_idx].to(device)
    y_val = data.y[val_idx].to(device)
    y_binary_val = data.y_binary[val_idx].to(device)  # Binary labels

    patient_ids_val = patient_ids[val_idx]
    
    # Only evaluate on Primary (0) vs Local (1)
    valid_val_mask = (y_val == 0) | (y_val == 1)
    
    # Verify NO Distant cells
    assert torch.all(valid_val_mask), "Validation should have NO Distant cells!"

    x_val_filtered = x_val[valid_val_mask]
    y_binary_val_filtered = y_binary_val[valid_val_mask]  # Binary for BOTH functions now
    patient_ids_val_filtered = patient_ids_val[valid_val_mask.cpu().numpy()]
    
    # ============================================================
    # CALCULATE PRIOR (MAJORITY CLASS BASELINE)
    # ============================================================
    
    print(f"\n{'='*60}")
    print("PRIOR BASELINE (Majority Class, Patient-Level)")
    print(f"{'='*60}")
    
    prior_metrics = calculate_majority_class_prior(
        y_binary_val_filtered, patient_ids_val_filtered
    )
    
    print(f"\nMajority class prediction on {prior_metrics['n_patients']} patients:")
    print(f"  Accuracy: {prior_metrics['acc']:.4f}")
    print(f"  F1 Score: {prior_metrics['f1']:.4f}")
    if not np.isnan(prior_metrics['auc']):
        print(f"  ROC-AUC:  {prior_metrics['auc']:.4f}")
    else:
        print(f"  ROC-AUC:  N/A (constant prediction)")
    if not np.isnan(prior_metrics['auprc']):
        print(f"  AUPRC:    {prior_metrics['auprc']:.4f}")
    else:
        print(f"  AUPRC:    N/A (constant prediction)")
    
    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
    
    model = scMeta(
        input_dim=data.num_node_features,
        hidden_dim=hidden_dim,
        num_classes=len(torch.unique(data.y)),
        conv_type=conv_type
    ).to(device)
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_metrics = {
        'acc': 0.0,
        'f1': 0.0,
        'auc': 0.0,
        'auprc': 0.0,
        'epoch': 0
    }
    
    best_metrics_cell = {
        'acc': 0.0,
        'f1': 0.0,
        'auc': 0.0,
        'auprc': 0.0,
        'epoch': 0
    }
    
    best_model_state = None
    patience_counter = 0
    
    
    
    print(f"\n{'='*60}")
    print(f"TRAINING: tau={tau}, gamma={gamma}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loader = RandomNodeLoader(train_data, num_parts=num_parts, shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            logits, emb = model(batch.x, batch.edge_index, return_embedding=True)
            
            loss_ce = F.cross_entropy(logits, batch.y)
            loss_con = NT_Xent(emb, tau=tau)
            loss = loss_ce + gamma * loss_con
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on Primary vs Local only
        # Cell-level metrics (fast)
        acc_cell, f1_cell, auc_cell, auprc_cell, _, _ = evaluate(
            model, x_val_filtered, y_binary_val_filtered, device
        )
        
        # Patient-level metrics
        patient_metrics = calculate_patient_level_metrics(
            model, x_val_filtered, y_binary_val_filtered, patient_ids_val_filtered, device
        )
        
        auc_patient = patient_metrics['auc']
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 0:
            def format_metric(value):
                """Format metric value, handling NaN."""
                if np.isnan(value):
                    return "N/A    "
                else:
                    return f"{value:.4f}"

            print(f"\nEpoch {epoch:3d}:")
            print(f"  [Cell-level]    Acc={format_metric(acc_cell)}, F1={format_metric(f1_cell)}, "
                  f"AUC={format_metric(auc_cell)}, AUPRC={format_metric(auprc_cell)}")
            print(f"  [Patient-level] Acc={format_metric(patient_metrics['acc'])}, F1={format_metric(patient_metrics['f1'])}, "
                  f"AUC={format_metric(patient_metrics['auc'])}, AUPRC={format_metric(patient_metrics['auprc'])}")
        
        # Track best based on cell-level accuracy 
        if auc_cell > best_metrics_cell['auc']: 

            best_metrics_cell = {
                'acc': acc_cell,
                'f1': f1_cell,
                'auc': auc_cell,
                'auprc': auprc_cell,
                'epoch': epoch
            }
            
        # Track best based on patient-level accuracy 
        if auc_patient > best_metrics['auc']:  
            print(f"NEW BEST MODEL at epoch {epoch}! AUC improved from {best_metrics['auc']:.4f} to {auc_patient:.4f}")

            best_metrics = {
                'acc': patient_metrics['acc'],
                'f1': patient_metrics['f1'],
                'auc': patient_metrics['auc'],
                'auprc': patient_metrics['auprc'],
                'epoch': epoch
            }
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break
    
    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (Best Model)")
    print(f"{'='*60}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Calculate patient-level metrics with best model
    final_patient_metrics = best_metrics
    
    # ============================================================
    # SAVE BEST MODEL
    # ============================================================
    
    model_path = None  # Initialize to avoid undefined variable
    if save_model:
        os.makedirs(model_save_dir, exist_ok=True)
        
        model_filename = f"model_tau{tau}_gamma{gamma}_fold{fold.get('fold_id', 0)}_epoch{best_metrics['epoch']}.pt"
        model_path = os.path.join(model_save_dir, model_filename)
        
        torch.save({
            'model_state_dict': best_model_state,
            'tau': tau,
            'gamma': gamma,
            'fold_id': fold.get('fold_id', 0),
            'epoch': best_metrics['epoch'],
            'metrics': final_patient_metrics,
            'model_config': {
                'input_dim': data.num_node_features,
                'hidden_dim': hidden_dim,
                'num_classes': len(torch.unique(data.y))
            }
        }, model_path)
        
        print(f"\n✓ Saved model to: {model_path}")
    
    print(f"\nBest model selected at epoch {best_metrics['epoch']}:")
    print(f"  [Patient-level] Acc={final_patient_metrics['acc']:.4f}, F1={final_patient_metrics['f1']:.4f}, "
          f"AUC={final_patient_metrics['auc']:.4f}, AUPRC={final_patient_metrics['auprc']:.4f}")
    
    # Return PATIENT-LEVEL metrics
    return {
        'acc': final_patient_metrics['acc'],
        'f1': final_patient_metrics['f1'],
        'auc': final_patient_metrics['auc'],
        'auprc': final_patient_metrics['auprc'],
        'cell_acc': best_metrics_cell['acc'],
        'cell_f1': best_metrics_cell['f1'],
        'cell_auc': best_metrics_cell['auc'],
        'cell_auprc': best_metrics_cell['auprc'],
        'epoch': best_metrics['epoch'],
        'prior_acc': prior_metrics['acc'],
        'prior_f1': prior_metrics['f1'],
        'prior_auc': prior_metrics['auc'],
        'prior_auprc': prior_metrics['auprc'],
        'model_path': model_path  
    }


def run_5fold_grid_search(
    data,
    patient_ids,
    param_grid,
    save_dir="./contrastive_gridsearch",
    epochs=100,
    patience=10,
    lr=1e-4,
    hidden_dim=128,
    num_parts=100,
    force_retrain=False,
    save_models=True 
):
    """
    Run 5-fold CV grid search for contrastive learning hyperparameters.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, "models")  #  Define model_save_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"\n{'='*60}")
    print("GRID SEARCH SETUP")
    print(f"{'='*60}")
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Number of folds: {len(data.cv_folds)}")
    print(f"Total training runs: {len(param_combinations) * len(data.cv_folds)}")
    print(f"Save models: {save_models}")  
    print(f"Parameters to search:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Store all results
    all_results = []
    
    # Loop over all hyperparameter combinations
    for param_idx, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        tau = params.get('tau', 0.5)
        gamma = params.get('gamma', 0.1)
        
        print(f"\n{'='*60}")
        print(f"Parameter Combination {param_idx + 1}/{len(param_combinations)}")
        print(f"{'='*60}")
        print(f"tau={tau}, gamma={gamma}")
        
        # Create unique hash for this configuration
        config_signature = {
            'tau': tau,
            'gamma': gamma,
            'epochs': epochs,
            'patience': patience,
            'lr': lr,
            'hidden_dim': hidden_dim
        }
        config_str = json.dumps(config_signature, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Check for cached results
        results_file = Path(save_dir) / f"results_tau{tau}_gamma{gamma}_{config_hash}.json"
        
        if results_file.exists() and not force_retrain:
            print(f"✓ Found cached results: {results_file}")
            with open(results_file, 'r') as f:
                cached_results = json.load(f)
            all_results.extend(cached_results)
            continue
        
        # Run 5-fold CV for this parameter combination
        fold_results = []
        
        for fold_idx, fold in enumerate(data.cv_folds):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/5")
            print(f"{'='*60}")
            
            # Add fold_id to fold dict
            fold['fold_id'] = fold_idx

            # Train on this fold
            metrics = train_one_fold(
                data=data,
                fold=fold,
                patient_ids=patient_ids,
                tau=tau,
                gamma=gamma,
                device=device,
                epochs=epochs,
                patience=patience,
                lr=lr,
                hidden_dim=hidden_dim,
                num_parts=num_parts,
                save_model=save_models,  #  Now defined
                model_save_dir=model_save_dir  #  Now defined
            )

            # Store results
            result = {
                'tau': tau,
                'gamma': gamma,
                'fold': fold_idx,
                'acc': metrics['acc'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'auprc': metrics['auprc'],
                'cell_acc': metrics['cell_acc'],
                'cell_f1': metrics['cell_f1'],
                'cell_auc': metrics['cell_auc'],
                'cell_auprc': metrics['cell_auprc'],
                'prior_acc': metrics['prior_acc'],
                'prior_f1': metrics['prior_f1'],
                'prior_auc': metrics['prior_auc'],
                'prior_auprc': metrics['prior_auprc'],
                'best_epoch': metrics['epoch'],
                'model_path': metrics.get('model_path', None),
                'n_train': len(fold['train_idx']),
                'n_val': len(fold['val_idx'])
            }
            fold_results.append(result)
            all_results.append(result)
        
        # Save results for this parameter combination
        with open(results_file, 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        # Print summary for this parameter combination
        fold_df = pd.DataFrame(fold_results)
        print(f"\n{'='*60}")
        print(f"Summary for tau={tau}, gamma={gamma}")
        print(f"{'='*60}")
        print(fold_df[['acc', 'f1', 'auc', 'auprc']].agg(['mean', 'std']).to_string())
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save full results
    results_df.to_csv(Path(save_dir) / 'all_results.csv', index=False)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {save_dir}/all_results.csv")
    if save_models:
        print(f"Models saved to: {model_save_dir}")
    
    return results_df

```


```python
# Define parameter grid
param_grid = {
    'tau': [0.05, 0.1, 0.3, 0.5, 0.7, 1.0],      # Temperature for NT-Xent
    'gamma': [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],             # Weight for contrastive loss
    
}
# Total combinations: 5 × 4 = 20
# Total runs: 20 × 5 folds = 100 training runs

# Run grid search
for dim in [64, 128, 256, 512]:
    print("/scratch/gilbreth/wang3712/Metastasis_single_cell/Rerun_scMeta_5fold_CV_patient_hidden"+str(dim)+"/")
    # continue
    results_df = run_5fold_grid_search(
        data=data,
        patient_ids=patient_ids,  # Pass patient IDs
        param_grid=param_grid,
        save_dir="/scratch/gilbreth/wang3712/Metastasis_single_cell/Rerun_scMeta_5fold_CV_patient_hidden"+str(dim)+"/",
        epochs=200,
        patience=30,
        lr=1e-4,
        hidden_dim=dim,
        num_parts=100,
        force_retrain=False,
        save_models = True,
        conv_type='TransformerConv' # change this to GATConv or SAGEConv for other models
    )
```


```python
'done'
```




    'done'



## Evaluation

This is what is reported in the paper.

### TransformerConv


```python

# ============================================================
# LOAD RESULTS FROM ALL 3 DIRECTORIES
# ============================================================

base_dir = '/scratch/gilbreth/wang3712/Metastasis_single_cell/'
hidden_dims = [64, 128, 256]

all_data = []

for dim in hidden_dims:
    results_file = f'{base_dir}/Rerun_scMeta_5fold_CV_patient_hidden{dim}/all_results.csv'
    
    if Path(results_file).exists():
        df = pd.read_csv(results_file)
        df['hidden_dim'] = dim  # Add hidden_dim column
        all_data.append(df)
        print(f"✓ Loaded {len(df)} results from hidden_dim={dim}")
    else:
        print(f"  Missing: {results_file}")

# Combine all results
df_all = pd.concat(all_data, ignore_index=True)

print(f"\n{'='*60}")
print("COMBINED DATASET")
print(f"{'='*60}")
print(f"Total runs: {len(df_all)}")
print(f"Hyperparameter combinations: {len(df_all.groupby(['tau', 'gamma', 'hidden_dim']))}")
print(f"Expected: {6 * 9 * 3} combinations × 5 folds = {6 * 9 * 3 * 5} runs")

# ============================================================
# COMPUTE MEAN METRICS PER CONFIGURATION
# ============================================================

summary = df_all.groupby(['tau', 'gamma', 'hidden_dim']).agg({
    'acc': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
    'cell_acc': ['mean', 'std'],
    'cell_auc': ['mean', 'std'],
    'best_epoch': 'mean'
}).round(4)

# Flatten column names
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

# ============================================================
# FIND BEST HYPERPARAMETERS FOR EACH METRIC
# ============================================================

metrics_to_optimize = {
    'AUC': 'auc_mean',
    'AUPRC': 'auprc_mean',
    'Accuracy': 'acc_mean',
    'F1': 'f1_mean'
}

print(f"\n{'='*70}")
print("BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)")
print(f"{'='*70}")

best_configs = {}

for metric_name, metric_col in metrics_to_optimize.items():
    best_row = summary.loc[summary[metric_col].idxmax()]
    best_configs[metric_name] = best_row
    
    print(f"\n Best {metric_name}:")
    print(f"  tau:        {best_row['tau']}")
    print(f"  gamma:      {best_row['gamma']}")
    print(f"  hidden_dim: {int(best_row['hidden_dim'])}")
    print(f"  {metric_name}:      {best_row[metric_col]:.4f} ± {best_row[metric_col.replace('_mean', '_std')]:.4f}")
    print(f"  AUC:        {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print(f"  AUPRC:      {best_row['auprc_mean']:.4f} ± {best_row['auprc_std']:.4f}")
    print(f"  Accuracy:   {best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}")
    print(f"  F1:         {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
    print(f"  Avg Epoch:  {best_row['best_epoch_mean']:.1f}")

# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

print(f"\n{'='*70}")
print("SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC")
print(f"{'='*70}")

best_summary = pd.DataFrame([
    {
        'Metric': metric_name,
        'tau': row['tau'],
        'gamma': row['gamma'],
        'hidden_dim': int(row['hidden_dim']),
        'Value': f"{row[metric_col]:.4f} ± {row[metric_col.replace('_mean', '_std')]:.4f}",
        'AUC': f"{row['auc_mean']:.4f}",
        'AUPRC': f"{row['auprc_mean']:.4f}",
        'ACC': f"{row['acc_mean']:.4f}",
        'F1': f"{row['f1_mean']:.4f}"
    }
    for metric_name, metric_col in metrics_to_optimize.items()
    for row in [best_configs[metric_name]]
])

print(best_summary.to_string(index=False))

# ============================================================
# TOP 10 CONFIGURATIONS BY AUC
# ============================================================

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC")
print(f"{'='*70}")

top10 = summary.nlargest(10, 'auc_mean')[
    ['tau', 'gamma', 'hidden_dim', 'auc_mean', 'auc_std', 'auprc_mean', 'acc_mean', 'f1_mean']
]
print(top10.to_string(index=False))

# ============================================================
# EFFECT OF HIDDEN DIMENSION
# ============================================================

print(f"\n{'='*70}")
print("EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)")
print(f"{'='*70}")

dim_effect = summary.groupby('hidden_dim').agg({
    'auc_mean': 'mean',
    'auprc_mean': 'mean',
    'acc_mean': 'mean',
    'f1_mean': 'mean'
}).round(4)

print(dim_effect.to_string())

# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = f'{base_dir[:-1]}_transformerconv_results/'
Path(output_dir).mkdir(exist_ok=True)

# Save full summary
summary.to_csv(f'{output_dir}/all_hyperparameters_summary.csv', index=False)
print(f"\n✓ Saved full summary to: {output_dir}/all_hyperparameters_summary.csv")

# Save best configs
best_summary.to_csv(f'{output_dir}/best_hyperparameters_by_metric.csv', index=False)
print(f"✓ Saved best configs to: {output_dir}/best_hyperparameters_by_metric.csv")

# Save top 10
top10.to_csv(f'{output_dir}/top10_by_auc.csv', index=False)
print(f"✓ Saved top 10 to: {output_dir}/top10_by_auc.csv")

```

    ✓ Loaded 360 results from hidden_dim=64
    ✓ Loaded 360 results from hidden_dim=128
    ✓ Loaded 360 results from hidden_dim=256
    
    ============================================================
    COMBINED DATASET
    ============================================================
    Total runs: 1080
    Hyperparameter combinations: 216
    Expected: 162 combinations × 5 folds = 810 runs
    
    ======================================================================
    BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)
    ======================================================================
    
     Best AUC:
      tau:        0.7
      gamma:      0.5
      hidden_dim: 256
      AUC:      0.8865 ± 0.0709
      AUC:        0.8865 ± 0.0709
      AUPRC:      0.6487 ± 0.0623
      Accuracy:   0.8572 ± 0.0384
      F1:         0.8624 ± 0.0388
      Avg Epoch:  51.8
    
     Best AUPRC:
      tau:        0.05
      gamma:      0.1
      hidden_dim: 256
      AUPRC:      0.6821 ± 0.1232
      AUC:        0.8791 ± 0.0855
      AUPRC:      0.6821 ± 0.1232
      Accuracy:   0.8702 ± 0.0394
      F1:         0.8744 ± 0.0393
      Avg Epoch:  40.4
    
     Best Accuracy:
      tau:        0.5
      gamma:      2.0
      hidden_dim: 128
      Accuracy:      0.8799 ± 0.0372
      AUC:        0.8754 ± 0.0846
      AUPRC:      0.6352 ± 0.0824
      Accuracy:   0.8799 ± 0.0372
      F1:         0.8815 ± 0.0408
      Avg Epoch:  26.0
    
     Best F1:
      tau:        0.5
      gamma:      2.0
      hidden_dim: 128
      F1:      0.8815 ± 0.0408
      AUC:        0.8754 ± 0.0846
      AUPRC:      0.6352 ± 0.0824
      Accuracy:   0.8799 ± 0.0372
      F1:         0.8815 ± 0.0408
      Avg Epoch:  26.0
    
    ======================================================================
    SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC
    ======================================================================
      Metric  tau  gamma  hidden_dim           Value    AUC  AUPRC    ACC     F1
         AUC 0.70    0.5         256 0.8865 ± 0.0709 0.8865 0.6487 0.8572 0.8624
       AUPRC 0.05    0.1         256 0.6821 ± 0.1232 0.8791 0.6821 0.8702 0.8744
    Accuracy 0.50    2.0         128 0.8799 ± 0.0372 0.8754 0.6352 0.8799 0.8815
          F1 0.50    2.0         128 0.8815 ± 0.0408 0.8754 0.6352 0.8799 0.8815
    
    ======================================================================
    TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC
    ======================================================================
     tau  gamma  hidden_dim  auc_mean  auc_std  auprc_mean  acc_mean  f1_mean
    0.70   0.50         256    0.8865   0.0709      0.6487    0.8572   0.8624
    0.70   0.20         128    0.8853   0.0771      0.6400    0.8637   0.8675
    0.05   5.00          64    0.8845   0.0750      0.6431    0.8572   0.8616
    0.50   0.05         256    0.8845   0.0817      0.6661    0.8636   0.8676
    1.00   0.10          64    0.8843   0.0797      0.6525    0.8604   0.8633
    0.30   4.00         128    0.8839   0.0787      0.6334    0.8638   0.8673
    0.05   0.05          64    0.8836   0.0770      0.6369    0.8604   0.8644
    0.05   2.00         256    0.8834   0.0787      0.6483    0.8604   0.8621
    0.05   3.00          64    0.8829   0.0753      0.6085    0.7260   0.7070
    0.10   3.00         256    0.8828   0.0801      0.6543    0.8636   0.8670
    
    ======================================================================
    EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)
    ======================================================================
                auc_mean  auprc_mean  acc_mean  f1_mean
    hidden_dim                                         
    64            0.8735      0.6271    0.8581   0.8613
    128           0.8743      0.6261    0.8577   0.8625
    256           0.8757      0.6330    0.8582   0.8635
    
    ✓ Saved full summary to: /scratch/gilbreth/wang3712/Metastasis_single_cell_transformerconv_results//all_hyperparameters_summary.csv
    ✓ Saved best configs to: /scratch/gilbreth/wang3712/Metastasis_single_cell_transformerconv_results//best_hyperparameters_by_metric.csv
    ✓ Saved top 10 to: /scratch/gilbreth/wang3712/Metastasis_single_cell_transformerconv_results//top10_by_auc.csv


### GATConv


```python

# ============================================================
# LOAD RESULTS FROM ALL 3 DIRECTORIES
# ============================================================

base_dir = '/scratch/gilbreth/wang3712/Metastasis_single_cell/'
hidden_dims = [64, 128, 256, 512]

all_data = []

for dim in hidden_dims:
    results_file = f'{base_dir}/Rerun_scMeta_GATConv_5fold_CV_patient_hidden{dim}/all_results.csv'
    
    if Path(results_file).exists():
        df = pd.read_csv(results_file)
        df['hidden_dim'] = dim  # Add hidden_dim column
        all_data.append(df)
        print(f"✓ Loaded {len(df)} results from hidden_dim={dim}")
    else:
        print(f"  Missing: {results_file}")

# Combine all results
df_all = pd.concat(all_data, ignore_index=True)

print(f"\n{'='*60}")
print("COMBINED DATASET")
print(f"{'='*60}")
print(f"Total runs: {len(df_all)}")
print(f"Hyperparameter combinations: {len(df_all.groupby(['tau', 'gamma', 'hidden_dim']))}")
print(f"Expected: {6 * 9 * 3} combinations × 5 folds = {6 * 9 * 3 * 5} runs")

# ============================================================
# COMPUTE MEAN METRICS PER CONFIGURATION
# ============================================================

summary = df_all.groupby(['tau', 'gamma', 'hidden_dim']).agg({
    'acc': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
    'cell_acc': ['mean', 'std'],
    'cell_auc': ['mean', 'std'],
    'best_epoch': 'mean'
}).round(4)

# Flatten column names
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

# ============================================================
# FIND BEST HYPERPARAMETERS FOR EACH METRIC
# ============================================================

metrics_to_optimize = {
    'AUC': 'auc_mean',
    'AUPRC': 'auprc_mean',
    'Accuracy': 'acc_mean',
    'F1': 'f1_mean'
}

print(f"\n{'='*70}")
print("BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)")
print(f"{'='*70}")

best_configs = {}

for metric_name, metric_col in metrics_to_optimize.items():
    best_row = summary.loc[summary[metric_col].idxmax()]
    best_configs[metric_name] = best_row
    
    print(f"\n Best {metric_name}:")
    print(f"  tau:        {best_row['tau']}")
    print(f"  gamma:      {best_row['gamma']}")
    print(f"  hidden_dim: {int(best_row['hidden_dim'])}")
    print(f"  {metric_name}:      {best_row[metric_col]:.4f} ± {best_row[metric_col.replace('_mean', '_std')]:.4f}")
    print(f"  AUC:        {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print(f"  AUPRC:      {best_row['auprc_mean']:.4f} ± {best_row['auprc_std']:.4f}")
    print(f"  Accuracy:   {best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}")
    print(f"  F1:         {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
    print(f"  Avg Epoch:  {best_row['best_epoch_mean']:.1f}")

# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

print(f"\n{'='*70}")
print("SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC")
print(f"{'='*70}")

best_summary = pd.DataFrame([
    {
        'Metric': metric_name,
        'tau': row['tau'],
        'gamma': row['gamma'],
        'hidden_dim': int(row['hidden_dim']),
        'Value': f"{row[metric_col]:.4f} ± {row[metric_col.replace('_mean', '_std')]:.4f}",
        'AUC': f"{row['auc_mean']:.4f}",
        'AUPRC': f"{row['auprc_mean']:.4f}",
        'ACC': f"{row['acc_mean']:.4f}",
        'F1': f"{row['f1_mean']:.4f}"
    }
    for metric_name, metric_col in metrics_to_optimize.items()
    for row in [best_configs[metric_name]]
])

print(best_summary.to_string(index=False))

# ============================================================
# TOP 10 CONFIGURATIONS BY AUC
# ============================================================

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC")
print(f"{'='*70}")

top10 = summary.nlargest(10, 'auc_mean')[
    ['tau', 'gamma', 'hidden_dim', 'auc_mean', 'auc_std', 'auprc_mean', 'acc_mean', 'f1_mean']
]
print(top10.to_string(index=False))

# ============================================================
# EFFECT OF HIDDEN DIMENSION
# ============================================================

print(f"\n{'='*70}")
print("EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)")
print(f"{'='*70}")

dim_effect = summary.groupby('hidden_dim').agg({
    'auc_mean': 'mean',
    'auprc_mean': 'mean',
    'acc_mean': 'mean',
    'f1_mean': 'mean'
}).round(4)

print(dim_effect.to_string())

# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = f'{base_dir[:-1]}_GATconv_results/'
Path(output_dir).mkdir(exist_ok=True)

# Save full summary
summary.to_csv(f'{output_dir}/all_hyperparameters_summary.csv', index=False)
print(f"\n✓ Saved full summary to: {output_dir}/all_hyperparameters_summary.csv")

# Save best configs
best_summary.to_csv(f'{output_dir}/best_hyperparameters_by_metric.csv', index=False)
print(f"✓ Saved best configs to: {output_dir}/best_hyperparameters_by_metric.csv")

# Save top 10
top10.to_csv(f'{output_dir}/top10_by_auc.csv', index=False)
print(f"✓ Saved top 10 to: {output_dir}/top10_by_auc.csv")

```

    ✓ Loaded 360 results from hidden_dim=64
    ✓ Loaded 360 results from hidden_dim=128
    ✓ Loaded 360 results from hidden_dim=256
    ✓ Loaded 360 results from hidden_dim=512
    
    ============================================================
    COMBINED DATASET
    ============================================================
    Total runs: 1440
    Hyperparameter combinations: 288
    Expected: 162 combinations × 5 folds = 810 runs
    
    ======================================================================
    BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)
    ======================================================================
    
     Best AUC:
      tau:        1.0
      gamma:      4.0
      hidden_dim: 512
      AUC:      0.8892 ± 0.0768
      AUC:        0.8892 ± 0.0768
      AUPRC:      0.6558 ± 0.1184
      Accuracy:   0.8701 ± 0.0323
      F1:         0.8693 ± 0.0353
      Avg Epoch:  45.2
    
     Best AUPRC:
      tau:        0.5
      gamma:      0.2
      hidden_dim: 64
      AUPRC:      0.6749 ± 0.1033
      AUC:        0.8890 ± 0.0723
      AUPRC:      0.6749 ± 0.1033
      Accuracy:   0.8636 ± 0.0278
      F1:         0.8630 ± 0.0304
      Avg Epoch:  40.0
    
     Best Accuracy:
      tau:        0.5
      gamma:      0.5
      hidden_dim: 64
      Accuracy:      0.8798 ± 0.0375
      AUC:        0.8783 ± 0.0794
      AUPRC:      0.6362 ± 0.0599
      Accuracy:   0.8798 ± 0.0375
      F1:         0.8805 ± 0.0391
      Avg Epoch:  47.6
    
     Best F1:
      tau:        0.5
      gamma:      0.5
      hidden_dim: 64
      F1:      0.8805 ± 0.0391
      AUC:        0.8783 ± 0.0794
      AUPRC:      0.6362 ± 0.0599
      Accuracy:   0.8798 ± 0.0375
      F1:         0.8805 ± 0.0391
      Avg Epoch:  47.6
    
    ======================================================================
    SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC
    ======================================================================
      Metric  tau  gamma  hidden_dim           Value    AUC  AUPRC    ACC     F1
         AUC  1.0    4.0         512 0.8892 ± 0.0768 0.8892 0.6558 0.8701 0.8693
       AUPRC  0.5    0.2          64 0.6749 ± 0.1033 0.8890 0.6749 0.8636 0.8630
    Accuracy  0.5    0.5          64 0.8798 ± 0.0375 0.8783 0.6362 0.8798 0.8805
          F1  0.5    0.5          64 0.8805 ± 0.0391 0.8783 0.6362 0.8798 0.8805
    
    ======================================================================
    TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC
    ======================================================================
     tau  gamma  hidden_dim  auc_mean  auc_std  auprc_mean  acc_mean  f1_mean
     1.0   4.00         512    0.8892   0.0768      0.6558    0.8701   0.8693
     0.5   0.20          64    0.8890   0.0723      0.6749    0.8636   0.8630
     1.0   0.01         512    0.8883   0.0731      0.6683    0.8733   0.8728
     1.0   0.05         512    0.8876   0.0795      0.6669    0.8733   0.8719
     0.3   0.00         256    0.8868   0.0748      0.6339    0.8701   0.8710
     0.7   5.00         128    0.8861   0.0772      0.6530    0.8571   0.8552
     0.5   1.00         512    0.8860   0.0734      0.6722    0.8765   0.8750
     0.7   0.50         256    0.8857   0.0743      0.6416    0.8765   0.8771
     0.3   0.01         128    0.8853   0.0802      0.6735    0.8603   0.8626
     0.3   0.05         256    0.8851   0.0702      0.6367    0.8669   0.8683
    
    ======================================================================
    EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)
    ======================================================================
                auc_mean  auprc_mean  acc_mean  f1_mean
    hidden_dim                                         
    64            0.8735      0.6310    0.8636   0.8652
    128           0.8745      0.6336    0.8648   0.8655
    256           0.8774      0.6310    0.8680   0.8679
    512           0.8788      0.6327    0.8681   0.8676
    
    ✓ Saved full summary to: /scratch/gilbreth/wang3712/Metastasis_single_cell_GATconv_results//all_hyperparameters_summary.csv
    ✓ Saved best configs to: /scratch/gilbreth/wang3712/Metastasis_single_cell_GATconv_results//best_hyperparameters_by_metric.csv
    ✓ Saved top 10 to: /scratch/gilbreth/wang3712/Metastasis_single_cell_GATconv_results//top10_by_auc.csv


### SAGEConv


```python

# ============================================================
# LOAD RESULTS FROM ALL 3 DIRECTORIES
# ============================================================

base_dir = '/scratch/gilbreth/wang3712/Metastasis_single_cell/'
hidden_dims = [64, 128, 256, 512]

all_data = []

for dim in hidden_dims:
    results_file = f'{base_dir}/Rerun_scMeta_SAGEConv_5fold_CV_patient_hidden{dim}/all_results.csv'
    
    if Path(results_file).exists():
        df = pd.read_csv(results_file)
        df['hidden_dim'] = dim  # Add hidden_dim column
        all_data.append(df)
        print(f"✓ Loaded {len(df)} results from hidden_dim={dim}")
    else:
        print(f"  Missing: {results_file}")

# Combine all results
df_all = pd.concat(all_data, ignore_index=True)

print(f"\n{'='*60}")
print("COMBINED DATASET")
print(f"{'='*60}")
print(f"Total runs: {len(df_all)}")
print(f"Hyperparameter combinations: {len(df_all.groupby(['tau', 'gamma', 'hidden_dim']))}")
print(f"Expected: {6 * 9 * 3} combinations × 5 folds = {6 * 9 * 3 * 5} runs")

# ============================================================
# COMPUTE MEAN METRICS PER CONFIGURATION
# ============================================================

summary = df_all.groupby(['tau', 'gamma', 'hidden_dim']).agg({
    'acc': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'auc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
    'cell_acc': ['mean', 'std'],
    'cell_auc': ['mean', 'std'],
    'best_epoch': 'mean'
}).round(4)

# Flatten column names
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

# ============================================================
# FIND BEST HYPERPARAMETERS FOR EACH METRIC
# ============================================================

metrics_to_optimize = {
    'AUC': 'auc_mean',
    'AUPRC': 'auprc_mean',
    'Accuracy': 'acc_mean',
    'F1': 'f1_mean'
}

print(f"\n{'='*70}")
print("BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)")
print(f"{'='*70}")

best_configs = {}

for metric_name, metric_col in metrics_to_optimize.items():
    best_row = summary.loc[summary[metric_col].idxmax()]
    best_configs[metric_name] = best_row
    
    print(f"\n Best {metric_name}:")
    print(f"  tau:        {best_row['tau']}")
    print(f"  gamma:      {best_row['gamma']}")
    print(f"  hidden_dim: {int(best_row['hidden_dim'])}")
    print(f"  {metric_name}:      {best_row[metric_col]:.4f} ± {best_row[metric_col.replace('_mean', '_std')]:.4f}")
    print(f"  AUC:        {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print(f"  AUPRC:      {best_row['auprc_mean']:.4f} ± {best_row['auprc_std']:.4f}")
    print(f"  Accuracy:   {best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}")
    print(f"  F1:         {best_row['f1_mean']:.4f} ± {best_row['f1_std']:.4f}")
    print(f"  Avg Epoch:  {best_row['best_epoch_mean']:.1f}")

# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

print(f"\n{'='*70}")
print("SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC")
print(f"{'='*70}")

best_summary = pd.DataFrame([
    {
        'Metric': metric_name,
        'tau': row['tau'],
        'gamma': row['gamma'],
        'hidden_dim': int(row['hidden_dim']),
        'Value': f"{row[metric_col]:.4f} ± {row[metric_col.replace('_mean', '_std')]:.4f}",
        'AUC': f"{row['auc_mean']:.4f}",
        'AUPRC': f"{row['auprc_mean']:.4f}",
        'ACC': f"{row['acc_mean']:.4f}",
        'F1': f"{row['f1_mean']:.4f}"
    }
    for metric_name, metric_col in metrics_to_optimize.items()
    for row in [best_configs[metric_name]]
])

print(best_summary.to_string(index=False))

# ============================================================
# TOP 10 CONFIGURATIONS BY AUC
# ============================================================

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC")
print(f"{'='*70}")

top10 = summary.nlargest(10, 'auc_mean')[
    ['tau', 'gamma', 'hidden_dim', 'auc_mean', 'auc_std', 'auprc_mean', 'acc_mean', 'f1_mean']
]
print(top10.to_string(index=False))

# ============================================================
# EFFECT OF HIDDEN DIMENSION
# ============================================================

print(f"\n{'='*70}")
print("EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)")
print(f"{'='*70}")

dim_effect = summary.groupby('hidden_dim').agg({
    'auc_mean': 'mean',
    'auprc_mean': 'mean',
    'acc_mean': 'mean',
    'f1_mean': 'mean'
}).round(4)

print(dim_effect.to_string())

# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = f'{base_dir[:-1]}_SAGEconv_results/'
Path(output_dir).mkdir(exist_ok=True)

# Save full summary
summary.to_csv(f'{output_dir}/all_hyperparameters_summary.csv', index=False)
print(f"\n✓ Saved full summary to: {output_dir}/all_hyperparameters_summary.csv")

# Save best configs
best_summary.to_csv(f'{output_dir}/best_hyperparameters_by_metric.csv', index=False)
print(f"✓ Saved best configs to: {output_dir}/best_hyperparameters_by_metric.csv")

# Save top 10
top10.to_csv(f'{output_dir}/top10_by_auc.csv', index=False)
print(f"✓ Saved top 10 to: {output_dir}/top10_by_auc.csv")

```

    ✓ Loaded 360 results from hidden_dim=64
    ✓ Loaded 360 results from hidden_dim=128
    ✓ Loaded 360 results from hidden_dim=256
    ✓ Loaded 360 results from hidden_dim=512
    
    ============================================================
    COMBINED DATASET
    ============================================================
    Total runs: 1440
    Hyperparameter combinations: 288
    Expected: 162 combinations × 5 folds = 810 runs
    
    ======================================================================
    BEST HYPERPARAMETERS FOR EACH METRIC (Patient-Level)
    ======================================================================
    
     Best AUC:
      tau:        1.0
      gamma:      0.01
      hidden_dim: 64
      AUC:      0.8901 ± 0.0677
      AUC:        0.8901 ± 0.0677
      AUPRC:      0.6439 ± 0.0823
      Accuracy:   0.8636 ± 0.0390
      F1:         0.8656 ± 0.0389
      Avg Epoch:  38.0
    
     Best AUPRC:
      tau:        0.3
      gamma:      4.0
      hidden_dim: 256
      AUPRC:      0.6831 ± 0.1310
      AUC:        0.8775 ± 0.0833
      AUPRC:      0.6831 ± 0.1310
      Accuracy:   0.8604 ± 0.0336
      F1:         0.8620 ± 0.0329
      Avg Epoch:  22.4
    
     Best Accuracy:
      tau:        0.05
      gamma:      0.02
      hidden_dim: 512
      Accuracy:      0.8830 ± 0.0469
      AUC:        0.8783 ± 0.0775
      AUPRC:      0.6276 ± 0.0760
      Accuracy:   0.8830 ± 0.0469
      F1:         0.8848 ± 0.0478
      Avg Epoch:  29.6
    
     Best F1:
      tau:        0.05
      gamma:      0.02
      hidden_dim: 512
      F1:      0.8848 ± 0.0478
      AUC:        0.8783 ± 0.0775
      AUPRC:      0.6276 ± 0.0760
      Accuracy:   0.8830 ± 0.0469
      F1:         0.8848 ± 0.0478
      Avg Epoch:  29.6
    
    ======================================================================
    SUMMARY TABLE: BEST PARAMETERS FOR EACH METRIC
    ======================================================================
      Metric  tau  gamma  hidden_dim           Value    AUC  AUPRC    ACC     F1
         AUC 1.00   0.01          64 0.8901 ± 0.0677 0.8901 0.6439 0.8636 0.8656
       AUPRC 0.30   4.00         256 0.6831 ± 0.1310 0.8775 0.6831 0.8604 0.8620
    Accuracy 0.05   0.02         512 0.8830 ± 0.0469 0.8783 0.6276 0.8830 0.8848
          F1 0.05   0.02         512 0.8848 ± 0.0478 0.8783 0.6276 0.8830 0.8848
    
    ======================================================================
    TOP 10 CONFIGURATIONS BY PATIENT-LEVEL AUC
    ======================================================================
     tau  gamma  hidden_dim  auc_mean  auc_std  auprc_mean  acc_mean  f1_mean
    1.00   0.01          64    0.8901   0.0677      0.6439    0.8636   0.8656
    0.70   5.00         128    0.8888   0.0732      0.6592    0.8538   0.8603
    0.05   3.00         512    0.8879   0.0809      0.6447    0.8669   0.8683
    0.30   0.10         256    0.8874   0.0778      0.6379    0.8668   0.8700
    1.00   2.00         512    0.8873   0.0800      0.6708    0.8798   0.8803
    0.10   3.00         256    0.8870   0.0779      0.6714    0.8538   0.8577
    0.30   3.00         256    0.8870   0.0799      0.6575    0.8733   0.8731
    0.10   4.00         256    0.8865   0.0730      0.6498    0.8637   0.8651
    0.05   3.00          64    0.8857   0.0735      0.6289    0.8669   0.8702
    0.70   4.00         512    0.8857   0.0726      0.6224    0.8668   0.8668
    
    ======================================================================
    EFFECT OF HIDDEN DIMENSION (Average Across All tau/gamma)
    ======================================================================
                auc_mean  auprc_mean  acc_mean  f1_mean
    hidden_dim                                         
    64            0.8754      0.6308    0.8592   0.8612
    128           0.8759      0.6328    0.8624   0.8650
    256           0.8762      0.6345    0.8620   0.8646
    512           0.8783      0.6350    0.8646   0.8670
    
    ✓ Saved full summary to: /scratch/gilbreth/wang3712/Metastasis_single_cell_SAGEconv_results//all_hyperparameters_summary.csv
    ✓ Saved best configs to: /scratch/gilbreth/wang3712/Metastasis_single_cell_SAGEconv_results//best_hyperparameters_by_metric.csv
    ✓ Saved top 10 to: /scratch/gilbreth/wang3712/Metastasis_single_cell_SAGEconv_results//top10_by_auc.csv



```python

```
