"""
Full-N rerun of run_go_comparison_differential.py.

Identical method -- same class-contrastive differential gradient signal
(compute_differential_saliency), same 5-model fold averaging, same GSEA
prerank settings, same seed. The ONLY change is that the CAP_PER_CLASS = 5000
cell cap is removed.

That cap was a compute-tractability shortcut, not a methodological choice, and
it was costing a lot of power very unevenly across cancer types:

    Breast    Regional_Mets  5,399  available -> was using 5,000  (1.1x)
    Lung      Regional_Mets 15,153  available -> was using 5,000  (3.0x)
    Ovarian   No_Mets       86,682  available -> was using 5,000 (17.3x)

Ovarian, the most severely under-sampled, is also the cancer type with the
weakest cross-cancer rank correlations (Lung/Ovarian rho=0.09,
Breast/Ovarian rho=0.20) and 0 significant GO BP terms under the capped run --
consistent with a sample-size artifact rather than absence of signal.

This rerun is pre-registered in the sense that matters: it was decided on the
basis of the cap being arbitrary, and it is reported below whatever the
outcome. If the correlations and significant-term counts do not improve, that
is the result.

Writes to ../go_comparison_differential_full_v2/ so the capped results remain
intact for comparison.
"""
import os
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import gseapy as gp
import gc
import sys
sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
from label_rules import recompute_metastasis_label
from scMeta import scMeta
from inductive_saliency import build_malignant_graph, compute_differential_saliency

DATA_PATH = '/depot/natallah/data/Mengbo/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
GMT_PATH = './data/h.all.v2024.1.Hs.symbols.gmt'  # feature-space GMT (same as training/local_subpop)
GO_GMT = './data/c5.go.bp.v2024.1.Hs.symbols.gmt'  # downstream enrichment target
MODEL_DIR = './v2b_scMeta_models/'
OUTPUT_DIR = '../go_comparison_differential_full_v2/'
CANCER_TYPES = ['Breast Cancer', 'Lung Cancer', 'Ovarian Cancer']  # Colorectal has 0 Regional_Mets
HIDDEN_DIM = 256
NUM_CLASSES = 3
CONV_TYPE = 'TransformerConv'
SEED = 42
BATCH_SIZE = 512
CAP_PER_CLASS = None  # full N: removed the 5,000-cell tractability cap

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading atlas...")
    ad_full = sc.read_h5ad(DATA_PATH, backed='r')
    recompute_metastasis_label(ad_full)
    malignant_mask = (ad_full.obs["Final_cell_type"] == "Malignant").values
    ad = ad_full[malignant_mask].to_memory()
    ad_full.file.close()
    gc.collect()
    print(f"Malignant population (graph context): {ad.n_obs} cells")

    target_genes = []
    with open(GMT_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            target_genes.extend(parts[2:])
    target_genes = sorted(list(set(target_genes)))
    valid_genes = [g for g in target_genes if g in ad.var_names]
    print(f"Feature space: {len(valid_genes)} genes")

    print("Building malignant-cell graph...")
    graph_data = build_malignant_graph(ad, valid_genes)

    models = []
    for fold in range(1, 6):
        model_path = os.path.join(MODEL_DIR, f"5foldCV_fold{fold}_scMeta.pt")
        m = scMeta(input_dim=len(valid_genes), hidden_dim=HIDDEN_DIM,
                   num_classes=NUM_CLASSES, conv_type=CONV_TYPE).to(device)
        m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    summary_rows = []
    rng = np.random.default_rng(SEED)

    for ctype in CANCER_TYPES:
        ctype_tag = ctype.replace(" ", "_")
        print(f"\n{'='*60}\n{ctype}\n{'='*60}")

        regional_idx = np.where((ad.obs["Final_cancer_type"] == ctype).values &
                                 (ad.obs["metastasis_label"] == "Regional_Mets").values)[0]
        primary_idx = np.where((ad.obs["Final_cancer_type"] == ctype).values &
                                (ad.obs["metastasis_label"] == "No_Mets").values)[0]
        if len(regional_idx) == 0:
            print(f"  0 Regional_Mets cells, skipping")
            continue

        n = min(len(regional_idx), len(primary_idx)) if CAP_PER_CLASS is None else min(len(regional_idx), len(primary_idx), CAP_PER_CLASS)
        class2_idx = rng.choice(regional_idx, size=n, replace=False)
        class1_idx = rng.choice(primary_idx, size=n, replace=False)
        print(f"  Regional_Mets available={len(regional_idx)}, No_Mets available={len(primary_idx)}, using n={n} per class")

        delta_g_sum = np.zeros(len(valid_genes), dtype=np.float64)
        for fold_i, model in enumerate(models, 1):
            delta_g = compute_differential_saliency(
                model, graph_data, class1_idx, class2_idx, target_class=1,
                device=device, batch_size=BATCH_SIZE)
            delta_g_sum += delta_g
            print(f"  fold {fold_i} done")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        delta_g_avg = delta_g_sum / len(models)

        rnk_df = pd.DataFrame({'Gene': valid_genes, 'Score': delta_g_avg}).sort_values(
            by='Score', ascending=False)
        rnk_df.to_csv(os.path.join(OUTPUT_DIR, f'{ctype_tag}_differential.rnk'),
                       sep='\t', index=False, header=False)

        gsea_out = os.path.join(OUTPUT_DIR, f'{ctype_tag}_differential_GOBP')
        gp.prerank(rnk=rnk_df, gene_sets=GO_GMT, outdir=gsea_out, min_size=5,
                   max_size=1000, permutation_num=1000, seed=42, threads=4)
        res = pd.read_csv(os.path.join(gsea_out, 'gseapy.gene_set.prerank.report.csv'))
        sig_terms = set(res.loc[res['FDR q-val'] < 0.05, 'Term'])
        print(f"  {len(sig_terms)} GO BP terms significant (of {len(res)} tested)")

        # Compare against the existing Wilcoxon-only term list from the one-sided run
        deg_only_path = f'../go_comparison_v2/{ctype_tag}_DEG_only_terms.csv'
        wilcoxon_sig = set()
        if os.path.exists(deg_only_path):
            wilcoxon_sig = set(pd.read_csv(deg_only_path)['Term'])
        # Note: DEG_only_terms.csv is Wilcoxon-significant AND not in the *one-sided*
        # scMeta significant set -- close enough to "Wilcoxon significant" for a
        # sanity overlap check, since the one-sided set was small/mostly disjoint.
        shared = sig_terms & wilcoxon_sig
        print(f"  overlap with previously-saved Wilcoxon-only terms: {len(shared)}")

        pd.Series(sorted(sig_terms), name='Term').to_csv(
            os.path.join(OUTPUT_DIR, f'{ctype_tag}_differential_sig_terms.csv'), index=False)

        summary_rows.append({'cancer_type': ctype, 'n_per_class': n,
                              'n_sig_terms_differential': len(sig_terms)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'GO_comparison_differential_summary.csv'), index=False)
    print("\n\nDone.")
    print(summary_df)


if __name__ == "__main__":
    main()
