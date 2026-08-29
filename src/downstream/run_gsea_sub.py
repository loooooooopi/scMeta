import os
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import gseapy as gp
import gc

# Import your scMeta model architecture here
# from your_model_file import scMeta
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DATA_PATH = '/home/wang4887/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
GMT_PATH = '/home/wang4887/scMetas/revision3/Github/data/h.all.v2024.1.Hs.symbols.gmt'
MODEL_DIR = './robustness_scMeta_models/'
OUTPUT_DIR = './gsea_subpopulations/'

BATCH_SIZE = 10000
HIDDEN_DIM = 256 # Update to your best model's hidden dim
NUM_CLASSES = 3
CONV_TYPE = 'TransformerConv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading full dataset into system RAM...")
    ad_full = sc.read_h5ad(TRAIN_DATA_PATH)
    recompute_metastasis_label(ad_full)

    # 1. Extract Primary and Local Mets
    valid_mask = (ad_full.obs["metastasis_label"] == "No_Mets").values | (ad_full.obs["metastasis_label"] == "Regional_Mets").values
    ad = ad_full[valid_mask].copy()
    
    del ad_full
    gc.collect()

    # 2. Compute Unsupervised Leiden Clusters (if not present)
    if 'leiden' not in ad.obs:
        print("Computing Leiden clusters on Harmony PCA space...")
        # Assuming X_pca_harmony exists from your preprocessing
        sc.pp.neighbors(ad, use_rep='X_pca_harmony', n_neighbors=15)
        sc.tl.leiden(ad, resolution=0.5)

    unique_cancers = ad.obs["Final_cancer_type"].unique()
    unique_clusters = ad.obs["leiden"].unique()

    # 3. Prepare Feature Space
    target_genes = []
    with open(GMT_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))
    valid_genes = [g for g in target_genes if g in ad.var_names]
    
    num_features = len(valid_genes)
    new_X = np.zeros((ad.n_obs, num_features), dtype=np.float32)
    gene_to_idx = {gene: i for i, gene in enumerate(valid_genes)}
    
    for i, gene in enumerate(ad.var_names):
        if gene in gene_to_idx:
            new_idx = gene_to_idx[gene]
            if sp.issparse(ad.X):
                new_X[:, new_idx] = ad.X[:, i].toarray().flatten()
            else:
                new_X[:, new_idx] = ad.X[:, i]

    # 4. Initialize Subpopulation Saliency Trackers
    sal_cancer = {c: np.zeros(num_features) for c in unique_cancers}
    count_cancer = {c: 0 for c in unique_cancers}
    
    sal_leiden = {l: np.zeros(num_features) for l in unique_clusters}
    count_leiden = {l: 0 for l in unique_clusters}

    print(f"Extracting feature importance across {ad.n_obs} cells...")

    # 5. Extract Gradients
    for fold in range(1, 6):
        model_path = os.path.join(MODEL_DIR, f"fold{fold}_best_model.pt")
        model = scMeta(input_dim=num_features, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, conv_type=CONV_TYPE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        for start_idx in range(0, ad.n_obs, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, ad.n_obs)
            
            x_batch = torch.tensor(new_X[start_idx:end_idx], dtype=torch.float32, device=device, requires_grad=True)
            edge_batch = torch.arange(x_batch.size(0), device=device).unsqueeze(0).repeat(2, 1)
            
            logits = model(x_batch, edge_batch)
            probs = torch.softmax(logits, dim=1)
            
            # Mask for highly confident Local Mets
            confident_mets_mask = (probs[:, 1] > 0.70).detach()
            
            if confident_mets_mask.sum() > 0:
                target = logits[confident_mets_mask, 1].sum()
                target.backward()
                
                raw_saliency = (x_batch.grad.detach().cpu().numpy() * x_batch.detach().cpu().numpy())
                confident_saliency = np.abs(raw_saliency[confident_mets_mask.cpu().numpy()])
                
                # Get metadata for these specific confident cells
                batch_cancers = ad.obs["Final_cancer_type"].iloc[start_idx:end_idx].values[confident_mets_mask.cpu().numpy()]
                batch_leiden = ad.obs["leiden"].iloc[start_idx:end_idx].values[confident_mets_mask.cpu().numpy()]
                
                # Route saliency to the correct buckets
                for i in range(len(confident_saliency)):
                    c = batch_cancers[i]
                    l = batch_leiden[i]
                    
                    sal_cancer[c] += confident_saliency[i]
                    count_cancer[c] += 1
                    sal_leiden[l] += confident_saliency[i]
                    count_leiden[l] += 1
                
                del target, raw_saliency, confident_saliency
            
            del x_batch, edge_batch, logits, probs, confident_mets_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        print(f"Fold {fold} gradients extracted.")

    # 6. Run GSEA for Cancer Types
    print("\n--- Running GSEA for Cancer Types ---")
    for ctype in unique_cancers:
        if count_cancer[ctype] > 50: # Only run if we have enough confident cells
            avg_sal = sal_cancer[ctype] / count_cancer[ctype]
            rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': avg_sal}).sort_values(by='Score', ascending=False)
            
            out_path = os.path.join(OUTPUT_DIR, f'cancer_{ctype.replace(" ", "_")}')
            gp.prerank(rnk=rnk_df, gene_sets=GMT_PATH, outdir=out_path, min_size=5, max_size=1000, seed=42)
            
            res = pd.read_csv(os.path.join(out_path, 'gseapy.gene_set.prerank.report.csv'))
            emt = res[res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False)]
            print(f"{ctype} (n={count_cancer[ctype]} cells) EMT FDR: {emt['FDR q-val'].values[0] if not emt.empty else 'Not Found'}")

    # 7. Run GSEA for Leiden Clusters
    print("\n--- Running GSEA for Leiden Clusters ---")
    for cluster in unique_clusters:
        if count_leiden[cluster] > 50:
            avg_sal = sal_leiden[cluster] / count_leiden[cluster]
            rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': avg_sal}).sort_values(by='Score', ascending=False)
            
            out_path = os.path.join(OUTPUT_DIR, f'leiden_cluster_{cluster}')
            gp.prerank(rnk=rnk_df, gene_sets=GMT_PATH, outdir=out_path, min_size=5, max_size=1000, seed=42)
            
            res = pd.read_csv(os.path.join(out_path, 'gseapy.gene_set.prerank.report.csv'))
            emt = res[res['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False)]
            
            fdr = emt['FDR q-val'].values[0] if not emt.empty else 'Not Found'
            print(f"Cluster {cluster} (n={count_leiden[cluster]} cells) EMT FDR: {fdr}")
            
            # If we find a significant EMT signal, print the top 3 pathways for this cluster!
            if isinstance(fdr, float) and fdr < 0.25:
                print(f"  --> SIGNIFICANT EMT SIGNAL FOUND IN CLUSTER {cluster}!")
                print(res[['Term', 'NES', 'FDR q-val']].head(3))


if __name__ == "__main__":
    main()

