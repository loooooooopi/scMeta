"""
Diagnostic for Reviewer 2 point 2 follow-up: is the kNN graph used for message
passing actually homophilous with respect to metastasis_label / y_binary? If
connected cells don't share label at an above-chance rate, graph convolution
is averaging in noise rather than signal, which would explain why scMeta
doesn't beat the plain MLP baseline (train_v2.py results).

Read-only: loads the same Data object train_v2.py builds, does not train
anything or write to the atlas.
"""
import numpy as np
import pandas as pd
from train_v2 import prepare_data

data = prepare_data()

row, col = data.edge_index[0].numpy(), data.edge_index[1].numpy()
y_bin = data.y_binary.numpy()
y_3c = data.y.numpy()
cancer = data.cancer_type

valid = (y_bin[row] != -1) & (y_bin[col] != -1)
row_v, col_v = row[valid], col[valid]
print(f"Total edges: {len(row)}, edges with both endpoints labeled: {valid.sum()}")

# --- overall binary-label homophily ---
same_label = (y_bin[row_v] == y_bin[col_v])
observed_homophily = same_label.mean()
p1 = (y_bin[row_v] == 1).mean()
p0 = (y_bin[row_v] == 0).mean()
expected_homophily = p0**2 + p1**2 + 2*p0*(1-p0-p1)  # not quite right for 2-class, fix below
# for 2-class chance homophily = p0^2 + p1^2 (using same marginal on both sides)
expected_homophily_2class = p0**2 + p1**2
print(f"\n[Binary mets label] observed edge homophily: {observed_homophily:.4f}")
print(f"[Binary mets label] expected under random pairing: {expected_homophily_2class:.4f}")
print(f"[Binary mets label] class balance: No_Mets={p0:.3f} Mets={p1:.3f}")

# --- same-patient edge fraction ---
pid = data.patient_ids
same_patient = (pid[row_v] == pid[col_v])
print(f"\nFraction of edges within the same patient: {same_patient.mean():.4f}")

# --- same-cancer-type edge fraction ---
same_cancer = (cancer[row_v] == cancer[col_v])
print(f"Fraction of edges within the same cancer type: {same_cancer.mean():.4f}")

# --- homophily restricted to cross-patient edges only (removes trivial "same patient" signal) ---
cross_patient = ~same_patient
if cross_patient.sum() > 0:
    cp_homophily = same_label[cross_patient].mean()
    print(f"\n[Cross-patient edges only] n={cross_patient.sum()}, label homophily: {cp_homophily:.4f}")

# --- per-cancer-type breakdown ---
print("\nPer-cancer-type binary-label homophily (within-cancer-type edges only):")
for ct in sorted(set(cancer)):
    m = same_cancer & (cancer[row_v] == ct)
    if m.sum() == 0:
        continue
    h = same_label[m].mean()
    p1_ct = (y_bin[row_v][m] == 1).mean()
    p0_ct = (y_bin[row_v][m] == 0).mean()
    exp_h = p0_ct**2 + p1_ct**2
    print(f"  {ct}: n_edges={m.sum()}, observed={h:.4f}, expected_random={exp_h:.4f}, mets_frac={p1_ct:.3f}")

# --- 3-class homophily too (finer-grained) ---
valid3 = (y_3c[row] != -1) & (y_3c[col] != -1)
row3, col3 = row[valid3], col[valid3]
same3 = (y_3c[row3] == y_3c[col3])
classes, counts = np.unique(y_3c[row3], return_counts=True)
freqs = counts / counts.sum()
exp3 = (freqs**2).sum()
print(f"\n[3-class label] observed edge homophily: {same3.mean():.4f}, expected random: {exp3:.4f}")

print("\nDone.")
