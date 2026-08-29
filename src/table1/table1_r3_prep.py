"""
Shared data preparation for the corrected Table 1 regeneration (R3 pipeline).

Uses EXACTLY train_v2.py's data construction:
  - labels recomputed from label_rules.recompute_metastasis_label (biopsy-site rule)
  - malignant cells only
  - MSigDB Hallmark gene intersection (same `valid_genes` construction)
  - graph from ad.obsp['connectivities']

Adds the two evaluation protocols of Table 1:
  - patient-stratified 5-fold CV (identical to train_v2.py's kfold_splits)
  - LOOCV over studies (Project_ID), as in Reproducibility/scMeta_LOOCV.ipynb

Caches everything to CACHE_DIR so downstream jobs do not re-read the 8 GB atlas.
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = '/home/wang4887/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad'
HALLMARK_GENES_PATH = '/home/wang4887/scMetas/revision3/Github/data/h.all.v2024.1.Hs.symbols.gmt'
CACHE_DIR = '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/table1/table1_r3/cache'


def build_cache():
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.model_selection import StratifiedKFold
    import sys
    sys.path.insert(0, '/depot/natallah/data/Mengbo/scMetas/revision3/Github/src/model')
    from label_rules import recompute_metastasis_label

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading {DATA_PATH} ...", flush=True)
    ad = sc.read_h5ad(DATA_PATH)
    print(f"  n_obs={ad.n_obs}", flush=True)

    print("Recomputing metastasis_label via label_rules.py ...", flush=True)
    recompute_metastasis_label(ad)

    ad = ad[ad.obs['Final_cell_type'] == 'Malignant'].copy()
    print(f"Malignant cells: {ad.n_obs}", flush=True)

    hallmark_genes = set()
    with open(HALLMARK_GENES_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            hallmark_genes.update(parts[2:])
    valid_genes = sorted([g for g in hallmark_genes if g in ad.var_names])
    print(f"Hallmark features: {len(valid_genes)}", flush=True)
    ad = ad[:, valid_genes].copy()

    lab = ad.obs['metastasis_label']
    valid_mask = lab.isin(['No_Mets', 'Regional_Mets', 'Distant_Mets']).values
    if (~valid_mask).sum():
        print(f"Dropping {(~valid_mask).sum()} cells with undefined labels", flush=True)
        ad = ad[valid_mask].copy()
        lab = ad.obs['metastasis_label']

    y3 = np.full(ad.n_obs, -1, dtype=np.int64)
    y3[(lab == 'No_Mets').values] = 0
    y3[(lab == 'Regional_Mets').values] = 1
    y3[(lab == 'Distant_Mets').values] = 2
    yb = (y3 > 0).astype(np.int64)

    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    X = np.ascontiguousarray(X, dtype=np.float32)

    adj = ad.obsp['connectivities'].tocoo()
    edge_index = np.vstack((adj.row, adj.col)).astype(np.int64)

    patient_ids = ad.obs['Final_sample_id'].astype(str).values
    project_ids = ad.obs['Project_ID'].astype(str).values
    cancer_type = ad.obs['Final_cancer_type'].astype(str).values

    np.save(f'{CACHE_DIR}/X.npy', X)
    np.save(f'{CACHE_DIR}/edge_index.npy', edge_index)
    np.save(f'{CACHE_DIR}/y3.npy', y3)
    np.save(f'{CACHE_DIR}/yb.npy', yb)
    pd.DataFrame({'patient_id': patient_ids, 'project_id': project_ids,
                  'cancer_type': cancer_type, 'y3': y3}).to_parquet(f'{CACHE_DIR}/meta.parquet')
    with open(f'{CACHE_DIR}/genes.txt', 'w') as f:
        f.write('\n'.join(valid_genes))

    # ---- splits ----
    patient_df = pd.DataFrame({'patient_id': patient_ids,
                               'cell_idx': np.arange(ad.n_obs), 'y_class': y3})
    pl = patient_df[patient_df['y_class'].isin([0, 1])]
    psum = pl.groupby('patient_id').agg({'y_class': list}).reset_index()
    psum['strat_label'] = psum['y_class'].apply(lambda x: 0 if 0 in x else (1 if 1 in x else -1))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pid_list = psum['patient_id'].values
    strat = psum['strat_label'].values
    folds = []
    for tr, va in skf.split(pid_list, strat):
        trp, vap = pid_list[tr], pid_list[va]
        train_idx = patient_df[patient_df['patient_id'].isin(trp)]['cell_idx'].values
        val_idx = patient_df[(patient_df['patient_id'].isin(vap)) &
                             (patient_df['y_class'].isin([0, 1]))]['cell_idx'].values
        folds.append((train_idx, val_idx))

    loocv = []
    for proj in sorted(set(project_ids)):
        val_idx = np.where((project_ids == proj) & np.isin(y3, [0, 1]))[0]
        if len(val_idx) == 0:
            print(f"  LOOCV: skipping study {proj} (no No_Mets/Regional_Mets cells)", flush=True)
            continue
        train_idx = np.where(project_ids != proj)[0]
        loocv.append((proj, train_idx, val_idx))

    np.savez(f'{CACHE_DIR}/splits.npz',
             **{f'fold{i}_train': t for i, (t, v) in enumerate(folds)},
             **{f'fold{i}_val': v for i, (t, v) in enumerate(folds)},
             **{f'loo_{p}_train': t for p, t, v in loocv},
             **{f'loo_{p}_val': v for p, t, v in loocv},
             loo_names=np.array([p for p, t, v in loocv]))

    print("\n== summary ==")
    print(f"cells={ad.n_obs} genes={len(valid_genes)}")
    print(f"class counts: No_Mets={int((y3==0).sum())} Regional={int((y3==1).sum())} Distant={int((y3==2).sum())}")
    print(f"patients (No/Regional)={len(psum)}  total studies={len(set(project_ids))}  LOOCV usable studies={len(loocv)}")
    for p, t, v in loocv:
        yv = yb[v]
        print(f"  {p}: n_val={len(v)} pos={int(yv.sum())} neg={int((yv==0).sum())} "
              f"npat={len(set(patient_ids[v]))} single_class={len(np.unique(yv))==1}")
    print("Cache written to", CACHE_DIR)


def load_cache():
    X = np.load(f'{CACHE_DIR}/X.npy', mmap_mode='r')
    edge_index = np.load(f'{CACHE_DIR}/edge_index.npy')
    y3 = np.load(f'{CACHE_DIR}/y3.npy')
    yb = np.load(f'{CACHE_DIR}/yb.npy')
    meta = pd.read_parquet(f'{CACHE_DIR}/meta.parquet')
    sp_ = np.load(f'{CACHE_DIR}/splits.npz', allow_pickle=True)
    folds = [(sp_[f'fold{i}_train'], sp_[f'fold{i}_val']) for i in range(5)]
    loo_names = [str(x) for x in sp_['loo_names']]
    loocv = [(p, sp_[f'loo_{p}_train'], sp_[f'loo_{p}_val']) for p in loo_names]
    return dict(X=X, edge_index=edge_index, y3=y3, yb=yb, meta=meta, folds=folds, loocv=loocv)


if __name__ == '__main__':
    build_cache()
