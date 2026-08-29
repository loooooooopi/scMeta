"""
Shared utility for gradient/saliency-based feature-importance extraction
(used by run_gsea.py, run_gsea_sub.py, run_sub_benchmark.py) that fixes the
same bug flagged for evaluation in train_v2.py: these scripts previously ran
the GNN on a self-loop-only edge_index (`arange(N).repeat(2,1)`), so the
biological interpretation (which genes drive a "metastatic" prediction) was
computed under conditions inconsistent with how the model was actually
trained and validated (real graph neighbors). This builds the same kind of
real inductive subgraph via NeighborLoader used in train_v2.py's
evaluate_inductive(), and computes input-gradient saliency for seed cells
w.r.t. their own features (message passing from neighbors still contributes
to the seed's logit, but we only read out gradients on the seed's own input
row, which is the standard definition of saliency for a node in a GNN).
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

NUM_NEIGHBORS_SALIENCY = [25, 25]


def build_malignant_graph(ad_malignant, valid_genes):
    """ad_malignant: AnnData already restricted to Final_cell_type == 'Malignant'
    (the full population, NOT scenario-filtered -- graph context needs all of
    it). Returns a torch_geometric Data(x, edge_index) in the given gene order.
    """
    gene_to_idx = {g: i for i, g in enumerate(valid_genes)}
    new_X = np.zeros((ad_malignant.n_obs, len(valid_genes)), dtype=np.float32)
    for i, gene in enumerate(ad_malignant.var_names):
        if gene in gene_to_idx:
            j = gene_to_idx[gene]
            col = ad_malignant.X[:, i]
            new_X[:, j] = col.toarray().flatten() if sp.issparse(col) else np.asarray(col).flatten()

    x = torch.tensor(new_X, dtype=torch.float32)
    adj = ad_malignant.obsp["connectivities"].tocoo()
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def compute_inductive_saliency(model, data, seed_idx, target_class, device,
                                confidence_threshold=0.70, batch_size=8192,
                                num_neighbors=NUM_NEIGHBORS_SALIENCY):
    """Real inductive forward+backward pass over seed_idx (order-preserving,
    since NeighborLoader with shuffle=False keeps seed nodes in input order
    and batch.batch_size seed rows come first in every batch).

    Returns:
      confident_mask: bool array, len(seed_idx), aligned to seed_idx order.
      confident_saliency: float32 array [n_confident, F], |grad * x| for each
        seed cell's own input features, in the same order as
        seed_idx[confident_mask].
    """
    model.eval()
    seed_idx_t = torch.as_tensor(seed_idx, dtype=torch.long)
    loader = NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=seed_idx_t,
                             batch_size=batch_size, shuffle=False)
    confident_flags = []
    saliency_chunks = []

    for batch in loader:
        batch = batch.to(device)
        n_seed = batch.batch_size
        x = batch.x.clone().requires_grad_(True)
        logits = model(x, batch.edge_index)
        probs = torch.softmax(logits, dim=1)
        seed_logits = logits[:n_seed]
        seed_probs = probs[:n_seed]
        confident_mask = (seed_probs[:, target_class] > confidence_threshold).detach()
        confident_flags.append(confident_mask.cpu())

        if confident_mask.sum() > 0:
            target = seed_logits[confident_mask, target_class].sum()
            grads = torch.autograd.grad(target, x, retain_graph=False)[0]
            seed_grads = grads[:n_seed]
            seed_x = x[:n_seed].detach()
            saliency = (seed_grads.detach() * seed_x).abs()
            saliency_chunks.append(saliency[confident_mask].cpu().numpy())

        del x, logits, probs, seed_logits, seed_probs, confident_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    confident_mask_full = torch.cat(confident_flags).numpy()
    confident_saliency = (np.concatenate(saliency_chunks, axis=0) if saliency_chunks
                           else np.zeros((0, data.x.shape[1]), dtype=np.float32))
    return confident_mask_full, confident_saliency


def _mean_abs_grad_of_target_logit(model, data, idx, target_class, device,
                                    batch_size, num_neighbors):
    """Mean |grad(logit_target_class) w.r.t. own input| over every cell in
    idx (no confidence filtering), via real inductive inference. Returns a
    [F] numpy vector.
    """
    model.eval()
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    loader = NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=idx_t,
                             batch_size=batch_size, shuffle=False)
    total = torch.zeros(data.x.shape[1])
    n = 0
    for batch in loader:
        batch = batch.to(device)
        n_seed = batch.batch_size
        x = batch.x.clone().requires_grad_(True)
        logits = model(x, batch.edge_index)
        seed_logits = logits[:n_seed]
        target = seed_logits[:, target_class].sum()
        grads = torch.autograd.grad(target, x, retain_graph=False)[0]
        seed_grads = grads[:n_seed]
        seed_x = x[:n_seed].detach()
        saliency = (seed_grads.detach() * seed_x).abs()
        total += saliency.sum(dim=0).cpu()
        n += n_seed
        del x, logits, seed_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (total / max(n, 1)).numpy()


def compute_differential_saliency(model, data, class1_idx, class2_idx, target_class, device,
                                   batch_size=8192, num_neighbors=NUM_NEIGHBORS_SALIENCY):
    """Implements the manuscript's documented (Methods Sec. feature_pro) but
    previously un-implemented differential gradient signal:
        g_bar_class1 = mean_i in class1 |grad(logit_target_class) w.r.t. x_i|
        g_bar_class2 = mean_i in class2 |grad(logit_target_class) w.r.t. x_i|
        delta_g = g_bar_class2 - g_bar_class1
    Genes with a uniformly large gradient magnitude regardless of class
    (e.g. housekeeping/high-expression genes) cancel out in the subtraction;
    only genes whose attribution differs between the two classes survive.
    compute_inductive_saliency / compute_inductive_saliency_multi_threshold
    above only compute one side of this (g_bar for confidently-predicted
    target-class cells alone), which does not have this cancellation
    property. No confidence filtering is applied here, matching the
    manuscript's equations exactly (every cell in each class contributes).
    Returns delta_g, a [F] numpy vector.
    """
    g_class1 = _mean_abs_grad_of_target_logit(model, data, class1_idx, target_class,
                                               device, batch_size, num_neighbors)
    g_class2 = _mean_abs_grad_of_target_logit(model, data, class2_idx, target_class,
                                               device, batch_size, num_neighbors)
    return g_class2 - g_class1


def compute_inductive_saliency_multi_threshold(model, data, seed_idx, target_class, device,
                                                 thresholds, batch_size=8192,
                                                 num_neighbors=NUM_NEIGHBORS_SALIENCY):
    """Same as compute_inductive_saliency but accumulates saliency at several
    confidence thresholds in one pass (used by run_sub_benchmark.py, which
    needs a Confident_Cells x threshold sweep per cluster).
    Returns dict[threshold] -> (saliency_sum [F], count).
    """
    model.eval()
    seed_idx_t = torch.as_tensor(seed_idx, dtype=torch.long)
    loader = NeighborLoader(data, num_neighbors=num_neighbors, input_nodes=seed_idx_t,
                             batch_size=batch_size, shuffle=False)
    num_features = data.x.shape[1]
    trackers = {t: torch.zeros(num_features) for t in thresholds}
    counts = {t: 0 for t in thresholds}

    for batch in loader:
        batch = batch.to(device)
        n_seed = batch.batch_size
        x = batch.x.clone().requires_grad_(True)
        logits = model(x, batch.edge_index)
        probs = torch.softmax(logits, dim=1)
        seed_logits = logits[:n_seed]
        seed_probs = probs[:n_seed].detach()

        for t in thresholds:
            mask = (seed_probs[:, target_class] >= t)
            if mask.sum() > 0:
                target = seed_logits[mask, target_class].sum()
                grads = torch.autograd.grad(target, x, retain_graph=True)[0]
                seed_grads = grads[:n_seed]
                seed_x = x[:n_seed].detach()
                saliency = (seed_grads.detach() * seed_x).abs()
                trackers[t] += saliency[mask].sum(dim=0).cpu()
                counts[t] += int(mask.sum().item())

        del x, logits, probs, seed_logits, seed_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {t: (trackers[t].numpy(), counts[t]) for t in thresholds}
