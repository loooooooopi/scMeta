"""
Diagnose why the scMeta-graphloss embedding UMAP (Figure3a_v2) looks
fragmented into many small islands compared to the old Figure3(a) (smooth
continuous streams). Hypothesis: the raw 256-dim embedding has ~1/3 dead
(near-zero-variance) dimensions (ReLU-saturated) and is used directly for
neighbors/UMAP with default n_neighbors=15 -- too small/noisy for 548K cells
in a raw (non-PCA-denoised) neural embedding space, unlike the Harmony
pipeline which always runs UMAP on a denoised 50-PC representation, not raw
expression. Try: (A) PCA-denoise the embedding to 50 PCs + default
neighbors, (B) raw embedding but larger n_neighbors, (C) PCA-denoise + larger
n_neighbors, and compare panel (a)-cancer-type-only plots side by side to
pick the closest match to the smooth, published style before committing to a
full 3-panel regeneration.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sc.set_figure_params(vector_friendly=True, dpi_save=200)

EMB_PATH = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings.npy'
META_PATH = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/full_embeddings_meta.csv'

emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
obs = pd.DataFrame({'Primary Cancer Type': meta['cancer_type'].astype('category')})

configs = [
    ('A_pca50_default_nn', dict(use_pca=True, n_pcs=50, n_neighbors=15)),
    ('B_raw_nn30', dict(use_pca=False, n_pcs=None, n_neighbors=30)),
    ('C_pca50_nn30', dict(use_pca=True, n_pcs=50, n_neighbors=30)),
    ('D_raw_nn50', dict(use_pca=False, n_pcs=None, n_neighbors=50)),
]

fig, axes = plt.subplots(1, len(configs), figsize=(7*len(configs), 6))

for ax, (name, cfg) in zip(axes, configs):
    print(f"=== {name}: {cfg} ===")
    ad = adnn.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32), obs=obs.copy())
    if cfg['use_pca']:
        from sklearn.decomposition import PCA
        pcs = PCA(n_components=cfg['n_pcs'], random_state=0).fit_transform(emb.astype(np.float32))
        ad.obsm['X_rep'] = pcs
    else:
        ad.obsm['X_rep'] = emb.astype(np.float32)
    print("  neighbors...")
    sc.pp.neighbors(ad, use_rep='X_rep', n_neighbors=cfg['n_neighbors'])
    print("  umap...")
    sc.tl.umap(ad)
    np.save(f'/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/umap_coords_{name}.npy', ad.obsm['X_umap'])
    sc.pl.umap(ad, color='Primary Cancer Type', ax=ax, show=False, legend_loc='right margin' if name==configs[-1][0] else None, title=name)

plt.tight_layout()
outpath = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/umap_param_sweep.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print("saved:", outpath)
