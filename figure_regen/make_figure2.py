import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from label_rules import recompute_metastasis_label

sc.set_figure_params(vector_friendly=True, dpi_save=300)

DATA_PATH = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'

print("Opening atlas (backed) -- obs/obsm load eagerly, X stays on disk...")
ad_full = sc.read_h5ad(DATA_PATH, backed='r')

print("Recomputing metastasis_label under the unified biopsy-site rule...")
recompute_metastasis_label(ad_full)

valid_mask = (
    (ad_full.obs['Final_cell_type'] == 'Malignant') &
    (ad_full.obs['metastasis_label'].isin(['No_Mets', 'Regional_Mets', 'Distant_Mets']))
).values
print(f"Malignant cells with a defined metastasis label: {valid_mask.sum()} / {ad_full.n_obs}")

label_map = {'No_Mets': 'Non-metastatic Local', 'Regional_Mets': 'Metastatic Local', 'Distant_Mets': 'Metastatic Distant'}
obs_sub = pd.DataFrame({
    'Primary Cancer Type': ad_full.obs['Final_cancer_type'].values[valid_mask],
    'Metastatic State': pd.Series(ad_full.obs['metastasis_label'].values[valid_mask]).map(label_map).values,
    'Biopsy Site': ad_full.obs['Final_tissue_backup'].values[valid_mask],
})
for c in obs_sub.columns:
    obs_sub[c] = obs_sub[c].astype('category')

pca_sub = np.asarray(ad_full.obsm['X_pca_harmony'])[valid_mask]
print("X_pca_harmony subset shape:", pca_sub.shape)

ad_full.file.close()

# Build a tiny, X-free AnnData purely for plotting (avoids ever loading expression data),
# then recompute neighbors/UMAP on just the malignant-cell subset (not the whole-atlas
# embedding, which mixes in all other cell types and washes out the malignant-cell structure).
ad = adnn.AnnData(X=np.zeros((valid_mask.sum(), 1), dtype=np.float32), obs=obs_sub)
ad.obsm['X_pca_harmony'] = pca_sub
print("Computing neighbors on malignant-cell-only PCA-harmony subset...")
sc.pp.neighbors(ad, use_rep='X_pca_harmony')
print("Computing UMAP...")
sc.tl.umap(ad)

print(ad.obs['Metastatic State'].value_counts())
print(ad.obs['Primary Cancer Type'].value_counts())

state_palette = {'Non-metastatic Local': '#2ca02c', 'Metastatic Local': '#e377c2', 'Metastatic Distant': '#ff7f0e'}

fig, axes = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 0.85, 1.25]})
sc.pl.umap(ad, color='Primary Cancer Type', ax=axes[0], show=False, legend_loc='right margin', title='Primary Cancer Type')
sc.pl.umap(ad, color='Metastatic State', ax=axes[1], show=False, legend_loc='right margin', title='Metastatic Label', palette=state_palette)
sc.pl.umap(ad, color='Biopsy Site', ax=axes[2], show=False, legend_loc='right margin', title='Biopsy Site')
# Panel 2's legend (only 3 short category names) was crowding panel 3's UMAP under a
# uniform wspace; widen specifically the panel2-panel3 gap while giving panel 3 extra
# width for its own long 12-category legend, instead of shrinking the whole figure.
fig.subplots_adjust(wspace=0.65)
pos1, pos2, pos3 = axes[0].get_position(), axes[1].get_position(), axes[2].get_position()
extra_gap = 0.03
axes[2].set_position([pos3.x0 + extra_gap, pos3.y0, pos3.width, pos3.height])
outpath = '/home/wang4887/scMetas/revision3/manuscript/Figures/Figure2_v2.png'
plt.savefig(outpath, bbox_inches='tight', dpi=300)
print("saved:", outpath)
print("n_obs used:", ad.n_obs)
