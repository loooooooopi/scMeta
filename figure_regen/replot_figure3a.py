"""
Re-renders Figure 3 panel (a) from the cached UMAP coordinates produced by
make_figure3a.py (no neighbors/UMAP recomputation -- the coordinates are fixed;
this script only controls how they are drawn).

Font sizing is driven backwards from the printed size. Figure3_v2.png is placed
at 0.85\\linewidth in a two-column figure*, i.e. about 414pt = 5.75in wide, and
combine_figure3.py normalises every panel to a common pixel width. So a label
set at F points inside a figure W inches wide ends up at F * 5.75 / W points on
the page. At W = 15in that means a 15pt label prints at ~5.75pt, which clears
the usual 5-6pt journal minimum; the earlier 22in-wide version put the same
labels at ~2.5pt, which does not.
"""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as adnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_W_IN = 15.0          # keep in sync with make_figure3b_genes.py
BASE_FS = 15

matplotlib.rcParams.update({
    'font.size': BASE_FS,
    'axes.titlesize': BASE_FS + 3,
    'axes.labelsize': BASE_FS,
    'xtick.labelsize': BASE_FS - 2,
    'ytick.labelsize': BASE_FS - 2,
    'legend.fontsize': BASE_FS - 3,
})
sc.set_figure_params(vector_friendly=True, dpi_save=300)

BASE = '/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen'
coords = np.load(f'{BASE}/figure3a_umap_coords.npy')
obs = pd.read_csv(f'{BASE}/figure3a_obs.csv')
for col in obs.columns:
    obs[col] = obs[col].astype('category')

ad = adnn.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32), obs=obs)
ad.obsm['X_umap'] = coords

state_palette = {'Non-metastatic Local': '#2ca02c', 'Metastatic Local': '#e377c2',
                 'Metastatic Distant': '#ff7f0e'}

# 548K points: small markers + partial alpha + rasterization control the
# overplotting noise. Purely a rendering choice; coordinates are untouched.
plot_kwargs = dict(size=3, alpha=0.4)

fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, 4.3))
sc.pl.umap(ad, color='Primary Cancer Type', ax=axes[0], show=False,
           legend_loc='right margin', title='Primary Cancer Type', **plot_kwargs)
sc.pl.umap(ad, color='Metastatic State', ax=axes[1], show=False,
           legend_loc='right margin', title='Metastatic Label',
           palette=state_palette, **plot_kwargs)
sc.pl.umap(ad, color='Biopsy Site', ax=axes[2], show=False,
           legend_loc='right margin', title='Biopsy Site', **plot_kwargs)
for ax in axes:
    for coll in ax.collections:
        coll.set_rasterized(True)

# All three panels share the same (arbitrary) UMAP axes, so the y-label is
# repeated information -- and on panels 2/3 it sits exactly where the previous
# panel's right-margin legend ends, clipping the longest legend entries. Keep it
# on the leftmost panel only; that removes the collision without having to widen
# the gaps and shrink the plots.
for ax in axes[1:]:
    ax.set_ylabel('')
plt.subplots_adjust(wspace=1.10)

outpath = '/home/wang4887/scMetas/revision3/manuscript/Figures/Figure3a_v2.png'
plt.savefig(outpath, bbox_inches='tight', dpi=300)
print("saved:", outpath)
