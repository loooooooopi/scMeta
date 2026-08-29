"""
Plot Supplementary Figure 4 (v3): 3-panel negative-control figure for the
CORRECTED matched-tissue healthy-cell negative control (52,509 cells / 9
donors), replacing the stale negative_control_v2.pdf (43,522 cells / 7
donors, buggy tissue-name-matching gap).

Reads:
  regen_supfig4_fold_summary.csv    (per-fold + ensemble FPR / ROC AUC / AUPRC)
  regen_supfig4_organ_summary.csv   (ensemble FPR per organ, corrected mapping)

Writes:
  /home/wang4887/scMetas/revision3/manuscript/Figures/negative_control_v3.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fold_df = pd.read_csv("./regen_supfig4_fold_summary.csv")
organ_df = pd.read_csv("./regen_supfig4_organ_summary.csv")

folds = fold_df[fold_df["fold"] != "ensemble"].sort_values("fold")
ensemble = fold_df[fold_df["fold"] == "ensemble"].iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ---------- Panel (a): per-fold + ensemble FPR ----------
ax = axes[0]
labels = [f"Fold {int(f)}" for f in folds["fold"]] + ["Ensemble"]
values = list(folds["false_positive_rate"] * 100) + [ensemble["false_positive_rate"] * 100]
colors = ["#4C72B0"] * len(folds) + ["#55A868"]
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("False positive rate (%)")
ax.set_title("(a) Plain-argmax FPR per fold\n(matched-tissue healthy controls, n=52,509 / 9 donors)",
              fontsize=10)
ax.set_ylim(0, max(values) * 1.25)

# ---------- Panel (b): per-fold ROC AUC / AUPRC ----------
ax = axes[1]
x = np.arange(len(folds))
w = 0.35
ax.bar(x - w / 2, folds["roc_auc"], width=w, label="ROC AUC", color="#2CA02C", edgecolor="black", linewidth=0.6)
ax.bar(x + w / 2, folds["auprc"], width=w, label="AUPRC", color="#9467BD", edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {int(f)}" for f in folds["fold"]])
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("(b) ROC AUC / AUPRC per fold\n(malignant validation positives vs. healthy negatives)",
              fontsize=10)
ax.legend(loc="lower right", fontsize=8)

# ---------- Panel (c): ensemble FPR by organ ----------
ax = axes[2]
organ_order = ["Breast", "Colorectal", "Lung", "Ovary"]
organ_df = organ_df.set_index("organ").reindex(organ_order).reset_index()
vals = organ_df["false_positive_rate"] * 100
colors_c = ["#4C72B0" if o != "Ovary" else "#C44E52" for o in organ_df["organ"]]
bars = ax.bar(organ_df["organ"], vals, color=colors_c, edgecolor="black", linewidth=0.6)
for b, v, n, nd in zip(bars, vals, organ_df["n_cells"], organ_df["n_donors"]):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
            f"n={int(n):,} ({int(nd)} donor{'s' if nd != 1 else ''})",
            ha="center", va="bottom", fontsize=8)
ax.set_ylabel("False positive rate (%)")
ax.set_title("(c) Ensemble FPR by organ\n(corrected matched-tissue mapping)", fontsize=10)
ax.set_ylim(0, max(vals) * 1.3)

plt.tight_layout()
out_path = "/home/wang4887/scMetas/revision3/manuscript/Figures/negative_control_v3.pdf"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
