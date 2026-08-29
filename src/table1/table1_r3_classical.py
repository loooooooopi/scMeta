"""Classical ML rows of the corrected Table 1 (RF / LogReg / SGD / ElasticNet).

One PRE-SPECIFIED hyperparameter configuration per model -- the configuration
that the original notebook's grid search selected (recorded in
Baseline_5fold_CV_cells/model_comparison_summary.csv). No re-tuning is done here.

Trained 3-class (No_Mets / Regional_Mets / Distant_Mets) exactly as the
notebooks did; evaluated on held-out No_Mets+Regional_Mets cells aggregated to
patient level by majority vote (see table1_r3_common.py).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from table1_r3_prep import load_cache
from table1_r3_common import patient_level_metrics

OUT_DIR = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/runs'

MODEL_SPECS = {
    'Random Forest': dict(
        cls='RandomForestClassifier',
        params=dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=4),
    ),
    'Logistic Regression': dict(
        cls='LogisticRegression',
        params=dict(C=1, penalty='l1', solver='saga', max_iter=1000, random_state=42),
    ),
    'SGD': dict(
        cls='SGDClassifier',
        params=dict(loss='log_loss', alpha=0.001, penalty='l2', max_iter=1000, random_state=42),
    ),
    'Elastic Net': dict(
        cls='LogisticRegression',
        params=dict(C=0.1, penalty='elasticnet', l1_ratio=0.5, solver='saga',
                    max_iter=1000, random_state=42),
    ),
}


def make_estimator(name):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    spec = MODEL_SPECS[name]
    return {'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier}[spec['cls']](**spec['params'])


def predict_3class_proba(model, X):
    p = model.predict_proba(X)
    full = np.zeros((X.shape[0], 3))
    for j, c in enumerate(model.classes_.astype(int)):
        if 0 <= c < 3:
            full[:, c] = p[:, j]
    return full


def run_task(model_name, scheme, split_name, train_idx, val_idx):
    out = os.path.join(OUT_DIR, f'classical__{model_name.replace(" ","_")}__{scheme}__{split_name}.csv')
    if os.path.exists(out):
        print(f'skip {out}', flush=True)
        return
    d = load_cache()
    X, y3, yb = d['X'], d['y3'], d['yb']
    pid = d['meta']['patient_id'].values

    Xtr = np.asarray(X[train_idx], dtype=np.float32)
    model = make_estimator(model_name)
    model.fit(Xtr, y3[train_idx])
    del Xtr

    Xva = np.asarray(X[val_idx], dtype=np.float32)
    proba = predict_3class_proba(model, Xva)
    del Xva
    y_pred_3 = proba.argmax(axis=1)

    m, _ = patient_level_metrics(y_pred_3, yb[val_idx], pid[val_idx])
    m.update(model=model_name, scheme=scheme, split=split_name,
             n_val_cells=len(val_idx), hyperparams=str(MODEL_SPECS[model_name]['params']))
    pd.DataFrame([m]).to_csv(out, index=False)
    print(f'done {out}: {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--models', default='all')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    d = load_cache()
    tasks = []
    models = list(MODEL_SPECS) if args.models == 'all' else args.models.split(',')
    for mname in models:
        for i, (tr, va) in enumerate(d['folds']):
            tasks.append((mname, '5foldCV', f'fold{i+1}', tr, va))
        for proj, tr, va in d['loocv']:
            tasks.append((mname, 'LOOCV', proj, tr, va))
    print(f'{len(tasks)} tasks', flush=True)
    Parallel(n_jobs=args.workers, backend='loky')(
        delayed(run_task)(*t) for t in tasks)


if __name__ == '__main__':
    main()
