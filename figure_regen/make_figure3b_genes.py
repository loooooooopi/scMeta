"""
Figure 3(b) alternative: cross-cancer-type consistency of scMeta's gene-level
attribution.

Motivation: the original Figure 3(b) heatmap claimed GO Biological Process
pathways "shared across breast, colorectal, lung, and ovarian cancers". That
claim does not survive this revision -- at the GO-term level the significant
sets are small and almost entirely non-overlapping between cancer types
(go_comparison_v2/GO_comparison_summary.csv), and the one term that DOES recur
across all three testable cancer types (GOBP_COMPLEMENT_RECEPTOR_MEDIATED_
SIGNALING_PATHWAY) is exactly the artifact the differential-signal fix was
written to remove: it is driven by class-independent gradient magnitude and
vanishes under the class-contrastive differential signal that main.tex's
Methods (\S sec:feature_pro) actually documents.

There IS, however, a genuine and defensible cross-cancer shared signal -- at
the GENE level, under the corrected differential signal. This script draws it.

Critically, this uses the DIFFERENTIAL rankings (go_comparison_differential_v2/
*_differential.rnk, from run_go_comparison_differential.py), NOT the one-sided
magnitude rankings. That distinction matters a lot here: under the one-sided
signal the cross-cancer Spearman correlations look much stronger (rho
0.58-0.76) but the consistently top-ranked genes are dominated by housekeeping
genes (UBC, ACTB, ACTG1, TPI1, ENO1) whose gradients are large regardless of
class. Under the differential signal those largely wash out (e.g. ACTB falls to
rank 1220 in ovarian, UBC to 351 in lung), leaving a biologically credible list
led by MDK -- the gene main.tex already names, with citation, as a top hit
across cancer types.

Colorectal Cancer is omitted for the same reason it is omitted from the GO
comparison: too few Regional_Mets cells under the unified label rule for a
matched differential contrast.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

BASE = '/depot/natallah/data/Mengbo/scMetas/revision3/go_comparison_differential_v2'
CANCERS = ['Breast_Cancer', 'Lung_Cancer', 'Ovarian_Cancer']
NICE = {'Breast_Cancer': 'Breast', 'Lung_Cancer': 'Lung', 'Ovarian_Cancer': 'Ovarian'}
N_GENES = 20

# --- load differential rankings, rank by |differential signal| ---
ranks, scores = {}, {}
for c in CANCERS:
    df = pd.read_csv(f'{BASE}/{c}_differential.rnk', sep='\t', header=None,
                     names=['gene', 'score'])
    df['abs'] = df.score.abs()
    df = df.sort_values('abs', ascending=False).reset_index(drop=True)
    ranks[c] = pd.Series(df.index.values + 1, index=df.gene)
    scores[c] = df.set_index('gene').score

R = pd.DataFrame(ranks).dropna()
S = pd.DataFrame(scores).dropna()

# Cross-cancer rank correlation (reported in the subtitle, so the figure states
# the actual effect size rather than implying sharing by layout alone)
pairs = []
for i, a in enumerate(CANCERS):
    for b in CANCERS[i + 1:]:
        rho, p = spearmanr(S[a], S[b])
        pairs.append((NICE[a], NICE[b], rho, p))
        print(f'{a} vs {b}: rho={rho:.3f} p={p:.2e}')

# Rank genes by their WORST rank across cancer types -- a gene only makes the
# top of this list if it is prioritized in every cancer type, not if one
# extreme value carries an average.
R['worst'] = R[CANCERS].max(axis=1)
top = R.sort_values('worst').head(N_GENES)
genes = list(top.index)
mat = top[CANCERS].T.values  # rows = cancer types, cols = genes

# --- plot: genes as columns, cancer types as rows (wide, matches panel a) ---
# Rank is a magnitude -> single-hue sequential, dark = rank 1 (strongest).
# Built from the same blue used for "scMeta gradient" elsewhere in Figure 3.
blues = LinearSegmentedColormap.from_list('scmeta_blues', ['#f2f7fb', '#1f77b4', '#0b3d61'])
VMAX = 100  # ranks beyond 100 all read as "not prioritized"; keeps top ranks separable

# Font sizing is driven backwards from the printed size: Figure3_v2.png sits at
# 0.85\linewidth (~5.75in) and combine_figure3.py normalises panels to a common
# width, so a label set at F points in a figure W inches wide prints at
# F * 5.75 / W points. W = 15in keeps these labels near 6pt on the page.
# Keep FIG_W_IN in sync with replot_figure3a.py.
FIG_W_IN = 15.0
fig, ax = plt.subplots(figsize=(FIG_W_IN, 3.5))
disp = np.clip(mat, 1, VMAX)
im = ax.imshow(disp, cmap=blues.reversed(), aspect='auto', vmin=1, vmax=VMAX)

ax.set_xticks(np.arange(len(genes)))
ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=15, style='italic')
ax.set_yticks(np.arange(len(CANCERS)))
ax.set_yticklabels([NICE[c] for c in CANCERS], fontsize=16)

# 2px surface gap between cells
ax.set_xticks(np.arange(-.5, len(genes), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(CANCERS), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
ax.tick_params(which='minor', bottom=False, left=False)
for s in ax.spines.values():
    s.set_visible(False)

# Rank printed in each cell -- the color carries the pattern, the number carries
# the value, so the cell is readable without resolving the exact shade.
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = int(mat[i, j])
        ax.text(j, i, str(v), ha='center', va='center', fontsize=12,
                color='white' if disp[i, j] < VMAX * 0.45 else '#333333')

cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.025)
cbar.set_label('Attribution rank\n(of 1,579 genes)', fontsize=13)
cbar.ax.invert_yaxis()
cbar.ax.tick_params(labelsize=12)
cbar.outline.set_visible(False)

# Title/subtitle both in axes coords so they can't collide with each other or
# drift under bbox_inches='tight'.
ax.set_title('Genes consistently prioritized across cancer types by '
             '$\\it{scMeta}$ class-contrastive gradient attribution',
             fontsize=19, pad=30)
sub = ('Ranked by worst rank across all three cancer types.   '
       'Cross-cancer Spearman $\\rho$: '
       + ', '.join(f'{a}/{b} {r:.2f}' for a, b, r, _ in pairs)
       + '   (all $p < 10^{-3}$)')
ax.text(0.5, 1.055, sub, transform=ax.transAxes, ha='center', va='bottom',
        fontsize=13.5, color='dimgray', style='italic')

out = '/home/wang4887/scMetas/revision3/manuscript/Figures/Figure3b_genes_v2.png'
plt.savefig(out, bbox_inches='tight', dpi=300, facecolor='white')
print('saved:', out)

top[CANCERS].to_csv('/depot/natallah/data/Mengbo/scMetas/revision3/figure_regen/figure3b_gene_ranks.csv')
print(top[CANCERS].astype(int).to_string())
