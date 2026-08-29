"""Shared patient-level aggregation + metrics for the corrected Table 1.

Aggregation rule (taken verbatim from the rule the stale Table 1 used, i.e.
`calculate_patient_level_metrics` in Reproducibility/baseline_models.ipynb and
scMeta_5foldCV/LOOCV.ipynb, and applied IDENTICALLY to all 7 models):

  1. Each cell gets a hard 3-class prediction (argmax over
     {No_Mets, Regional_Mets, Distant_Mets}).
  2. For a patient (Final_sample_id), let pct_mets = % of that patient's held-out
     cells predicted Regional or Distant.
  3. Patient hard call = 0 (No_Mets) if pct_primary > pct_mets else 1  --> MAJORITY VOTE.
  4. Patient continuous score used for AUROC/AUPRC = pct_mets.
  5. Accuracy / F1(weighted) on the hard calls; AUROC / AUPRC on pct_mets.
     AUROC/AUPRC are NaN when the fold's patients are all one class.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)


def patient_level_metrics(y_pred_3class, y_true_binary_cells, patient_ids_val):
    y_pred_3class = np.asarray(y_pred_3class)
    y_true_binary_cells = np.asarray(y_true_binary_cells)
    patient_ids_val = np.asarray(patient_ids_val)

    rows = []
    n_hetero = 0
    for pid in np.unique(patient_ids_val):
        m = patient_ids_val == pid
        yt = y_true_binary_cells[m]
        if len(np.unique(yt)) > 1:
            n_hetero += 1
        true_label = int(round(yt.mean()))  # majority of cell-level truth
        preds = y_pred_3class[m]
        n = len(preds)
        c = np.bincount(preds, minlength=3)
        pct_primary = c[0] / n * 100
        pct_mets = (c[1] + c[2]) / n * 100
        rows.append(dict(patient_id=pid, true_label=true_label,
                         pred_label=0 if pct_primary > pct_mets else 1,
                         pct_mets=pct_mets, n_cells=n))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return dict(accuracy=np.nan, f1=np.nan, auroc=np.nan, auprc=np.nan,
                    n_patients=0, n_pos_patients=0, n_neg_patients=0,
                    n_heterogeneous_patients=0), df

    yt = df['true_label'].values
    yp = df['pred_label'].values
    ys = df['pct_mets'].values
    defined = len(np.unique(yt)) > 1
    return dict(
        accuracy=accuracy_score(yt, yp),
        f1=f1_score(yt, yp, average='weighted', zero_division=0),
        auroc=roc_auc_score(yt, ys) if defined else np.nan,
        auprc=average_precision_score(yt, ys) if defined else np.nan,
        n_patients=len(df),
        n_pos_patients=int((yt == 1).sum()),
        n_neg_patients=int((yt == 0).sum()),
        n_heterogeneous_patients=n_hetero,
    ), df
