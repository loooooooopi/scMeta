"""Aggregate per-fold / per-study runs into the corrected Table 1."""
import glob
import os
import numpy as np
import pandas as pd

RUNS = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/runs'
OUT_RAW = '/home/wang4887/scMetas/revision3/Github/table1_corrected_results.csv'
OUT_SUM = '/home/wang4887/scMetas/revision3/Github/table1_corrected_summary.csv'

ORDER = ['Random Forest', 'Logistic Regression', 'SGD', 'Elastic Net',
         'scMeta-GAT', 'scMeta-SAGE', 'scMeta-Tran']
METRICS = [('accuracy', 'Accuracy'), ('f1', 'F1'), ('auroc', 'AUROC'), ('auprc', 'AUPRC')]


def main():
    files = sorted(glob.glob(os.path.join(RUNS, '*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['model'] = pd.Categorical(df['model'], ORDER, ordered=True)
    df = df.sort_values(['scheme', 'model', 'split'])
    df.to_csv(OUT_RAW, index=False)
    print(f'{len(df)} runs -> {OUT_RAW}')

    rows = []
    for scheme in ['5foldCV', 'LOOCV']:
        for model in ORDER:
            sub = df[(df.scheme == scheme) & (df.model == model)]
            r = {'Model': model, 'Protocol': scheme, 'n_folds_total': len(sub)}
            for key, label in METRICS:
                v = sub[key].dropna()
                r[f'{label}_mean'] = v.mean() if len(v) else np.nan
                r[f'{label}_sd'] = v.std(ddof=1) if len(v) > 1 else np.nan
                r[f'{label}_n'] = len(v)
                r[f'{label}'] = (f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'
                                 if len(v) > 1 else (f'{v.mean():.3f}' if len(v) == 1 else 'n/a'))
            rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(OUT_SUM, index=False)
    print(s[['Model', 'Protocol', 'n_folds_total', 'Accuracy', 'F1', 'AUROC', 'AUPRC',
             'AUROC_n']].to_string(index=False))


if __name__ == '__main__':
    main()
