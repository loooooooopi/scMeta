import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import textwrap

import os

# Built directly from run_go_comparison.py's outputs so this script is
# self-contained and reproducible from the repository alone.
GO_DIR = '/depot/natallah/data/Mengbo/scMetas/revision3/go_comparison_v2'
SOURCES = [
    ('scMeta gradient', '{c}_scMeta_only_terms.csv', '{c}_scMeta_gradient_GOBP'),
    ('Wilcoxon DEG', '{c}_DEG_only_terms.csv', '{c}_Wilcoxon_DEG_GOBP'),
]

rows = []
for cancer in ['Breast Cancer', 'Lung Cancer', 'Ovarian Cancer']:
    tag = cancer.replace(' ', '_')
    for method, terms_tpl, gsea_tpl in SOURCES:
        terms_path = os.path.join(GO_DIR, terms_tpl.format(c=tag))
        report_path = os.path.join(GO_DIR, gsea_tpl.format(c=tag),
                                   'gseapy.gene_set.prerank.report.csv')
        if not (os.path.exists(terms_path) and os.path.exists(report_path)):
            continue
        terms = pd.read_csv(terms_path)
        if terms.empty:
            continue
        report = pd.read_csv(report_path).set_index('Term')
        for t in terms['Term']:
            if t in report.index:
                rows.append({'cancer': cancer, 'method': method, 'term': t,
                             'NES': float(report.loc[t, 'NES']),
                             'FDR': float(report.loc[t, 'FDR q-val'])})

df = pd.DataFrame(rows)
print(f'{len(df)} significant terms loaded across {df.cancer.nunique()} cancer types')

def clean_term(t):
    t = t.replace('GOBP_', '').replace('_', ' ').title()
    return '\n'.join(textwrap.wrap(t, 28))

df['label'] = df['term'].apply(clean_term)

cancers = ['Breast Cancer', 'Lung Cancer', 'Ovarian Cancer']
method_colors = {'scMeta gradient': '#1f77b4', 'Wilcoxon DEG': '#d62728'}

# Wide aspect on purpose: this panel is stacked under panel (a) and normalised
# to its width, so a narrow figure gets scaled UP and eats vertical page space.
# Widening also gives the GO term labels room to sit on fewer lines.
fig, axes = plt.subplots(1, 3, figsize=(22, 4.8), sharex=True,
                          gridspec_kw={'wspace': 0.55})

xmax = df['NES'].abs().max() * 1.35

for ax, cancer in zip(axes, cancers):
    sub = df[df['cancer'] == cancer].copy()
    sub = sub.sort_values('NES')
    y = range(len(sub))
    colors = [method_colors[m] for m in sub['method']]
    ax.barh(y, sub['NES'], color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['label'], fontsize=7.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(cancer, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('NES', fontsize=9)
    ax.set_xlim(-xmax, xmax)
    ax.tick_params(axis='x', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

handles = [plt.Rectangle((0,0),1,1, color=c, ec='black', linewidth=0.5) for c in method_colors.values()]
fig.legend(handles, method_colors.keys(), loc='upper center', ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.04), fontsize=10)

fig.suptitle('Significantly Enriched GO Biological Process Terms (FDR < 0.05), Per Cancer Type',
             fontsize=11.5, y=1.13)
fig.text(0.5, 1.085, 'FDR q-values are listed in Supplementary Table 7',
          ha='center', fontsize=8.5, color='dimgray', style='italic')
plt.tight_layout(rect=[0, 0, 1, 0.93])
outpath = '/home/wang4887/scMetas/revision3/manuscript/Figures/Figure3b_go_pathways_v2.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print('saved to', outpath)
