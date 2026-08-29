"""
Unified, explicit metastasis-label rule applied uniformly across all four
cancer types, in response to Reviewer 2's point that the previous labelling
mixed a patient-stage (TNM/FIGO M1) criterion with a biopsy-site criterion
inconsistently across studies.

Rule: a cell's class is determined SOLELY by the anatomical site the sample
was biopsied from, relative to the primary organ for that cancer type.
Patient-level stage (TNM/FIGO) is intentionally NOT used, so a cell's label
depends only on where it was physically sampled from -- not on the overall
stage of the patient it came from.

Classes:
  No_Mets       - biopsy site is the primary organ itself
  Regional_Mets - biopsy site is a locoregional site for that cancer type
  Distant_Mets  - biopsy site is a distant organ/site

Tissue categories below use `Final_tissue_backup`, which preserves the
finer-grained site labels (e.g. distinguishing ovarian peritoneal-cavity
sites from truly unknown sites) that get collapsed together under the
coarser `Final_tissue` field used elsewhere in the atlas.
"""

PRIMARY_ORGAN = {
    "Breast Cancer": {"Breast"},
    "Colorectal Cancer": {"Colon"},
    "Lung Cancer": {"Lung"},
    "Ovarian Cancer": {"Ovary", "Ovarium"},
}

REGIONAL_SITES = {
    # Axillary/regional lymph nodes: standard AJCC regional site for breast cancer.
    "Breast Cancer": {"Lymph node", "Axilla"},
    # No locoregional-lymph-node-labelled tissue present in this cohort for colon.
    "Colorectal Cancer": set(),
    # Hilar/mediastinal lymph nodes: standard AJCC regional site for lung cancer.
    # NOTE: malignant pleural effusion/pleural dissemination is AJCC M1a (distant),
    # not regional -- Pleura is intentionally NOT listed here.
    "Lung Cancer": {"Lymph node", "Neck"},
    # FIGO stage II/III intraperitoneal spread (pelvic/abdominal peritoneal
    # implants, omental caking, malignant ascites, serosal bowel implants) is
    # locoregional for ovarian cancer, unlike parenchymal-organ or
    # extra-abdominal distant disease (FIGO IV).
    "Ovarian Cancer": {
        "Peritoneum", "Peritoneal",
        "Omentum",
        "Ascites",
        "Upper Quadrant",
        "Bowel",
    },
}

# Sites intentionally NOT covered above (e.g. Bone, Liver, Brain, Adrenal,
# Blood, Pleura, and ovarian "Other") fall through to Distant_Mets by default.
# "Other" (ovarian, unspecified site) is treated as Distant_Mets conservatively
# -- an unknown site is not assumed to be locoregional.

VALID_CANCER_TYPES = set(PRIMARY_ORGAN)


def assign_metastasis_label(cancer_type: str, tissue_backup: str) -> str:
    """Map a single cell's (cancer type, fine-grained biopsy site) to one of
    {No_Mets, Regional_Mets, Distant_Mets}, or None if cancer_type is unrecognized.
    """
    if cancer_type not in VALID_CANCER_TYPES:
        return None

    if tissue_backup in PRIMARY_ORGAN[cancer_type]:
        return "No_Mets"
    if tissue_backup in REGIONAL_SITES[cancer_type]:
        return "Regional_Mets"
    return "Distant_Mets"


def assign_metastasis_label_vectorized(cancer_type_series, tissue_backup_series):
    """Vectorized version for a pandas Series pair of equal length/index."""
    import pandas as pd

    out = pd.Series("Distant_Mets", index=cancer_type_series.index, dtype=object)
    unknown_cancer = ~cancer_type_series.isin(VALID_CANCER_TYPES)
    out[unknown_cancer] = None

    for cancer_type in VALID_CANCER_TYPES:
        mask = cancer_type_series == cancer_type
        out[mask & tissue_backup_series.isin(PRIMARY_ORGAN[cancer_type])] = "No_Mets"
        if REGIONAL_SITES[cancer_type]:
            out[mask & tissue_backup_series.isin(REGIONAL_SITES[cancer_type])] = "Regional_Mets"

    return out


def assign_synchronous_mets_flag(cancer_type_series, tissue_backup_series, patient_stage_series):
    """For primary-organ-site cells only, flag whether the patient already had a
    confirmed Stage IV / M1 distant metastasis diagnosis AT THE TIME of that
    sample (i.e. synchronous metastatic disease), based on the recorded
    Final_patient_stage. This is a side-column for sensitivity analysis -- it is
    NOT folded into metastasis_label, which stays a pure biopsy-site label.

    Returns a pandas Series of {"True", "False", "Unknown", None}:
      "True"    - primary-organ-site cell, patient recorded as Stage IV at sampling
      "False"   - primary-organ-site cell, patient recorded as Stage I/II/III
      "Unknown" - primary-organ-site cell, stage not recorded ("Unknown"/NaN)
      None      - not a primary-organ-site cell (flag not applicable) or unrecognized cancer type
    """
    import pandas as pd
    import numpy as np

    out = pd.Series(None, index=cancer_type_series.index, dtype=object)
    stage_str = patient_stage_series.astype(str)

    for cancer_type in VALID_CANCER_TYPES:
        primary_mask = (cancer_type_series == cancer_type) & tissue_backup_series.isin(PRIMARY_ORGAN[cancer_type])
        out[primary_mask & (stage_str == "Stage IV")] = "True"
        out[primary_mask & stage_str.isin(["Stage I", "Stage II", "Stage III"])] = "False"
        out[primary_mask & stage_str.isin(["Unknown", "nan", "None", "NaN"])] = "Unknown"

    return out


def recompute_metastasis_label(adata):
    """Overwrite adata.obs['metastasis_label'] IN PLACE with the unified
    biopsy-site rule, replacing whatever value was baked into the atlas file
    by the old, unreproducible labelling step. Call this immediately after
    loading the atlas and before any code that reads metastasis_label, so
    every consumer (training, GSEA, benchmarks) uses the same single rule.
    """
    adata.obs["metastasis_label"] = assign_metastasis_label_vectorized(
        adata.obs["Final_cancer_type"], adata.obs["Final_tissue_backup"]
    )
