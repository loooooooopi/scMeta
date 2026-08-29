"""
Read-only regeneration of the metastasis label under the unified biopsy-site
rule (see label_rules.py). Does NOT modify the original atlas h5ad in any way
-- it only reads obs metadata (backed mode) and writes new output files:

  revision3/Github/Pre-processing/label_truth_table.csv
      Per-dataset truth table: (cancer_type, tissue_backup) -> old label
      counts vs new label counts. This is the artifact Reviewer 2 asked for.

  revision3/Github/Pre-processing/metastasis_label_v2.csv.gz
      Per-cell side table (indexed by obs_names) with the old and new labels
      plus the synchronous-mets sensitivity flag, to be merged into
      training/eval scripts without duplicating the 7.4GB atlas file.
"""
import os
import sys
import scanpy as sc
import pandas as pd

ATLAS_PATH = "/home/wang4887/scMetas/luopin/Data/processed_data/All_integrated.harmony.h5ad"
OUT_DIR = "/home/wang4887/scMetas/revision3/Github/Pre-processing"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "model"))
from label_rules import assign_metastasis_label_vectorized, assign_synchronous_mets_flag, VALID_CANCER_TYPES


def main():
    print(f"Reading obs metadata (backed mode, read-only) from {ATLAS_PATH} ...")
    ad = sc.read_h5ad(ATLAS_PATH, backed="r")
    obs = ad.obs[["Final_cancer_type", "Final_tissue", "Final_tissue_backup",
                  "Final_patient_stage", "metastasis_label"]].copy()
    obs = obs.rename(columns={"metastasis_label": "metastasis_label_old"})

    print("Computing new labels under the unified biopsy-site rule ...")
    obs["metastasis_label_new"] = assign_metastasis_label_vectorized(
        obs["Final_cancer_type"], obs["Final_tissue_backup"]
    )

    print("Computing synchronous-mets sensitivity flag for primary-organ-site cells ...")
    obs["primary_site_had_synchronous_distant_mets"] = assign_synchronous_mets_flag(
        obs["Final_cancer_type"], obs["Final_tissue_backup"], obs["Final_patient_stage"]
    )

    n_unrecognized_cancer = obs["metastasis_label_new"].isna().sum()
    if n_unrecognized_cancer:
        print(f"WARNING: {n_unrecognized_cancer} cells have a Final_cancer_type "
              f"outside {VALID_CANCER_TYPES}; metastasis_label_new left as NaN for these.")

    # --- Truth table: per (cancer_type, tissue_backup), old vs new label counts ---
    truth_table = (
        obs.groupby(["Final_cancer_type", "Final_tissue_backup", "metastasis_label_old", "metastasis_label_new"],
                     dropna=False, observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["Final_cancer_type", "Final_tissue_backup"])
    )
    truth_table_path = os.path.join(OUT_DIR, "label_truth_table.csv")
    truth_table.to_csv(truth_table_path, index=False)
    print(f"Wrote truth table ({len(truth_table)} rows) to {truth_table_path}")

    # --- Overall old vs new label distribution, for a quick sanity check ---
    print("\n=== Old label distribution ===")
    print(obs["metastasis_label_old"].value_counts(dropna=False))
    print("\n=== New label distribution ===")
    print(obs["metastasis_label_new"].value_counts(dropna=False))
    print("\n=== Old -> New transition matrix (all cells) ===")
    print(pd.crosstab(obs["metastasis_label_old"], obs["metastasis_label_new"], dropna=False))

    print("\n=== primary_site_had_synchronous_distant_mets distribution (per cancer type, No_Mets cells only) ===")
    no_mets_mask = obs["metastasis_label_new"] == "No_Mets"
    print(pd.crosstab(obs.loc[no_mets_mask, "Final_cancer_type"],
                       obs.loc[no_mets_mask, "primary_site_had_synchronous_distant_mets"]))

    # --- Per-cell side table for merging into training scripts ---
    side_table = obs[["metastasis_label_old", "metastasis_label_new",
                       "primary_site_had_synchronous_distant_mets"]].copy()
    side_table.index.name = "obs_names"
    side_table_path = os.path.join(OUT_DIR, "metastasis_label_v2.csv.gz")
    side_table.reset_index().to_csv(side_table_path, index=False)
    print(f"\nWrote per-cell side table ({len(side_table)} rows) to {side_table_path}")


if __name__ == "__main__":
    main()
