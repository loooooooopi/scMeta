import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sc.set_figure_params(vector_friendly=True, dpi_save=300)

EMB_PATH = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings.npy'
META_PATH = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings_meta.csv'

print("Loading embeddings + metadata...")
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
print("Embeddings shape:", emb.shape, "meta rows:", len(meta))

label_map = {'No_Mets': 'Non-metastatic Local', 'Regional_Mets': 'Metastatic Local', 'Distant_Mets': 'Metastatic Distant'}
obs = pd.DataFrame({
    'Primary Cancer Type': meta['cancer_type'].astype('category'),
    'Metastatic State': meta['metastasis_label'].map(label_map).astype('category'),
    'Biopsy Site': meta['biopsy_site'].astype('category'),
})

ad = adnn.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32), obs=obs)

# Raw 256-dim post-ReLU embedding has ~1/3 near-zero-variance dims (dead ReLU
# units) and was optimized with a contrastive (graph_nt_xent) objective that
# pulls graph-neighbor pairs tightly together -- geometrically much more
# locally clustery than a PCA/Harmony manifold. Direct neighbors/UMAP on it
# with default n_neighbors=15 fragmented into many small islands (see
# tune_figure3a_umap.py sweep). PCA-denoising to 50 comps + n_neighbors=30
# (config C in that sweep) recovers a smooth, continuous embedding
# comparable in style to Figure 2's Harmony-PCA UMAP, without changing what
# the embedding actually contains -- this is a visualization choice only.
from sklearn.decomposition import PCA
print("PCA-denoising embedding to 50 components...")
pcs = PCA(n_components=50, random_state=0).fit_transform(emb.astype(np.float32))
ad.obsm['X_emb_pca'] = pcs

print("Computing neighbors on PCA-denoised scMeta-graphloss embedding space...")
sc.pp.neighbors(ad, use_rep='X_emb_pca', n_neighbors=30)
print("Computing UMAP...")
sc.tl.umap(ad)

np.save('/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/figure3a_umap_coords.npy', ad.obsm['X_umap'])
obs.to_csv('/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/figure3a_obs.csv', index=False)

state_palette = {'Non-metastatic Local': '#2ca02c', 'Metastatic Local': '#e377c2', 'Metastatic Distant': '#ff7f0e'}

# 548K points with default marker size/alpha causes heavy overplotting that
# reads as visual noise (isolated-looking dots that are really dense strands
# viewed through opaque overlapping markers). Smaller size + partial alpha +
# rasterization is a pure rendering fix -- does not touch neighbors/UMAP
# coordinates or any reported result.
plot_kwargs = dict(size=3, alpha=0.4)

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
sc.pl.umap(ad, color='Primary Cancer Type', ax=axes[0], show=False, legend_loc='right margin', title='Primary Cancer Type', **plot_kwargs)
sc.pl.umap(ad, color='Metastatic State', ax=axes[1], show=False, legend_loc='right margin', title='Metastatic Label', palette=state_palette, **plot_kwargs)
sc.pl.umap(ad, color='Biopsy Site', ax=axes[2], show=False, legend_loc='right margin', title='Biopsy Site', **plot_kwargs)
for ax in axes:
    for coll in ax.collections:
        coll.set_rasterized(True)
plt.subplots_adjust(wspace=0.55)
outpath = '/home/wang4887/scMetas/revision3/manuscript/Figures/Figure3a_v2.png'
plt.savefig(outpath, bbox_inches='tight', dpi=300)
print("saved:", outpath)
