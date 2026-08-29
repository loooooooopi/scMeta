import os
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gseapy as gp
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to match your cluster environment
TRAIN_DATA_PATH = '/home/wang4887/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
GMT_PATH = '/home/wang4887/scMetas/revision3/Github/data/h.all.v2024.1.Hs.symbols.gmt'
OUTPUT_DIR = './baseline_gsea_results/'


os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading full dataset into system RAM from {TRAIN_DATA_PATH}...")
    ad_full = sc.read_h5ad(TRAIN_DATA_PATH)
    recompute_metastasis_label(ad_full)

    # 1. Restrict to Malignant cells, then filter out Distant Mets (matches
    #    run_gsea.py / train_v2.py). metastasis_label is a tissue/cancer-type
    #    based label applied to every cell in the atlas, not just malignant
    #    ones, so without this filter the baseline would be trained on a mix
    #    of malignant and non-malignant (immune/stromal) cells from labeled
    #    tissues -- not a fair comparison to scMeta, which is malignant-only.
    print("Extracting strictly Malignant Primary and Local Mets cells...")
    valid_mask = ((ad_full.obs["Final_cell_type"] == "Malignant") &
                  ((ad_full.obs["metastasis_label"] == "No_Mets") | (ad_full.obs["metastasis_label"] == "Regional_Mets"))).values
    ad_filtered = ad_full[valid_mask].copy()
    
    # Free up memory
    del ad_full
    import gc
    gc.collect()

    # 2. Prepare Features (X) and Labels (y)
    print(f"Preparing feature matrix for {ad_filtered.n_obs} cells...")
    
    # Extract only the MSigDB genes (matching the model input space)
    target_genes = []
    with open(GMT_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))
    valid_genes = [g for g in target_genes if g in ad_filtered.var_names]
    
    num_features = len(valid_genes)
    X = np.zeros((ad_filtered.n_obs, num_features), dtype=np.float32)
    gene_to_idx = {gene: i for i, gene in enumerate(valid_genes)}
    
    src_positions, dst_positions = [], []
    for i, gene in enumerate(ad_filtered.var_names):
        if gene in gene_to_idx:
            src_positions.append(i)
            dst_positions.append(gene_to_idx[gene])
    if sp.issparse(ad_filtered.X):
        X[:, dst_positions] = ad_filtered.X.tocsc()[:, src_positions].toarray()
    else:
        X[:, dst_positions] = ad_filtered.X[:, src_positions]
                
    # Binary Labels: 0 for Primary (No_Mets), 1 for Local (Regional_Mets)
    y = np.zeros(ad_filtered.n_obs, dtype=int)
    y[(ad_filtered.obs["metastasis_label"] == "Regional_Mets").values] = 1

    # 3. Train Baseline Model (Logistic Regression)
    print("Training Logistic Regression baseline model (Primary vs Local)...")
    # Using saga solver for large datasets and class_weight to handle imbalance
    model = LogisticRegression(penalty='l2', solver='saga', class_weight='balanced', max_iter=100, n_jobs=-1)
    model.fit(X, y)

    # 4. Extract Feature Importance (Absolute Coefficients)
    print("Extracting feature weights...")
    weights = model.coef_[0]
    importance_scores = np.abs(weights)

    # 5. Create Ranked Gene List
    rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': importance_scores})
    rnk_df = rnk_df.sort_values(by='Score', ascending=False)
    
    rnk_path = os.path.join(OUTPUT_DIR, 'baseline_gene_importance.rnk')
    rnk_df.to_csv(rnk_path, sep='\t', index=False, header=False)
    print(f"\nRanked baseline gene list saved to {rnk_path}")

    # 6. Run MSigDB GSEA
    print("Running GSEA against MSigDB Hallmarks...")
    pre_res = gp.prerank(
        rnk=rnk_df,
        gene_sets=GMT_PATH,
        threads=4,
        min_size=5,
        max_size=1000,
        permutation_num=1000,
        outdir=os.path.join(OUTPUT_DIR, 'gseapy_out'),
        seed=42,
        verbose=True
    )

    # 7. Display Results
    results_df = pre_res.res2d
    print("\n" + "="*50)
    print("BASELINE: TOP 10 ENRICHED PATHWAYS")
    print("="*50)
    print(results_df[['Term', 'ES', 'NES', 'FDR q-val']].head(10))
    
    emt_row = results_df[results_df['Term'].str.contains('EPITHELIAL_MESENCHYMAL_TRANSITION', case=False, na=False)]
    print("\n" + "="*50)
    print("BASELINE: EMT PATHWAY SPECIFIC RESULTS")
    print("="*50)
    if not emt_row.empty:
        print(emt_row[['Term', 'ES', 'NES', 'FDR q-val']])
    else:
        print("EMT pathway not found in baseline results.")
        

if __name__ == "__main__":
    main()
	
