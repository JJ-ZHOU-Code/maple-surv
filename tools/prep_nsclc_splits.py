"""
Prepare TCGA-NSCLC survival data by combining:
  - LUAD: existing mahmoodlab_tcga_luad_survival.csv (patient_id, e, t)
  - LUSC: downloaded from GDC API

Output format matches LUAD CSV: pathology_id, patient_id, e, t, subtype
Creates 5-fold CV splits stratified by subtype + survival.
"""
import os, json, time, requests, random
import pandas as pd
import numpy as np
from pathlib import Path


GDC_API = "https://api.gdc.cancer.gov"
LUAD_CSV = "data_split/5foldcv/tcga_luad/mahmoodlab_tcga_luad_survival.csv"
LUSC_GDC_TSV = "/home/zjj/zjj/data/TCGA/lusc/download/gdc_sample_sheet.2024-11-30.tsv"
LUNG_FEATS = "/home/zjj/zjj/data/TCGA/lung/ExpData/feats-l0-s1024-CONCH/pt_files"
OUT_DIR = "data_split/5foldcv/tcga_nsclc"


def fetch_lusc_clinical():
    """Download TCGA-LUSC case-level survival data from GDC API."""
    filters = json.dumps({
        "op": "in",
        "content": {"field": "cases.project.project_id", "value": ["TCGA-LUSC"]}
    })
    fields = ",".join([
        "submitter_id",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.age_at_diagnosis",
    ])
    params = {"filters": filters, "fields": fields, "format": "json", "size": 1000}
    url = f"{GDC_API}/cases"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    print(f"[GDC] Fetched {len(hits)} LUSC cases.")
    return hits


def parse_survival(hits):
    """
    Extract OS (overall survival) from GDC case data.
    Returns DataFrame with columns: patient_id, e, t_days
    """
    rows = []
    for h in hits:
        pid = h["submitter_id"]
        if "diagnoses" not in h or len(h["diagnoses"]) == 0:
            continue
        diag = h["diagnoses"][0]
        status = diag.get("vital_status", "")
        days_death = diag.get("days_to_death")
        days_follow = diag.get("days_to_last_follow_up")

        if status.lower() == "dead" and days_death is not None:
            e = 1
            t_days = days_death
        elif days_follow is not None:
            e = 0
            t_days = days_follow
        else:
            continue  # skip incomplete records

        t_months = round(t_days / 30.44, 2)  # days → months (matching LUAD format)
        rows.append({"patient_id": pid, "e": float(e), "t": t_months})

    df = pd.DataFrame(rows).drop_duplicates("patient_id")
    print(f"[parse] {len(df)} LUSC patients with valid OS data.")
    return df


def add_pathology_ids(surv_df, gdc_tsv, lung_feats):
    """
    Cross-reference patient IDs with GDC sample sheet to get slide IDs.
    Keep only slides that are in the lung CONCH features directory.
    """
    gdc = pd.read_csv(gdc_tsv, sep="\t", dtype=str)
    # GDC sample sheet: "Case ID" = TCGA-XX-XXXX, "File Name" = slide_id.svs
    gdc["patient_id"] = gdc["Case ID"].str.strip()
    gdc["pathology_id"] = gdc["File Name"].str.replace(".svs", "", regex=False)

    # Available lung CONCH pt files
    avail_slides = {f.replace(".pt", "") for f in os.listdir(lung_feats) if f.endswith(".pt")}

    merged = pd.merge(surv_df, gdc[["patient_id", "pathology_id"]], on="patient_id", how="inner")
    merged = merged[merged["pathology_id"].isin(avail_slides)]
    print(f"[merge] {len(merged)} LUSC slide-patient rows with CONCH features available.")
    return merged


def create_nsclc_csv(luad_csv, lusc_df):
    """Combine LUAD + LUSC into one NSCLC survival CSV with subtype labels."""
    luad = pd.read_csv(luad_csv, dtype={"patient_id": str, "pathology_id": str})
    luad["subtype"] = 0  # LUAD = 0

    # Only keep LUAD patients that have CONCH features in the lung dir
    lung_feats_path = LUNG_FEATS
    avail_slides = {f.replace(".pt", "") for f in os.listdir(lung_feats_path) if f.endswith(".pt")}
    luad = luad[luad["pathology_id"].isin(avail_slides)].copy()
    print(f"[nsclc] LUAD rows with lung CONCH features: {len(luad)}")

    lusc_df["subtype"] = 1  # LUSC = 1
    lusc_df = lusc_df[["pathology_id", "patient_id", "e", "t", "subtype"]]

    nsclc = pd.concat([luad[["pathology_id", "patient_id", "e", "t", "subtype"]], lusc_df], ignore_index=True)
    nsclc = nsclc.drop_duplicates("patient_id")
    print(f"[nsclc] Total NSCLC patients: {len(nsclc)} (LUAD={nsclc.subtype.eq(0).sum()}, LUSC={nsclc.subtype.eq(1).sum()})")
    return nsclc


def create_stratified_kfold_splits(nsclc_df, k=5, seed=42):
    """
    Create K-fold CV splits stratified by subtype.
    Returns list of dicts: [{"train": [...], "val": [...]}]
    """
    rng = np.random.RandomState(seed)
    # Patient-level (one row per patient already after drop_duplicates)
    pids = nsclc_df["patient_id"].values
    subtypes = nsclc_df["subtype"].values

    # Separate by subtype
    luad_idx = np.where(subtypes == 0)[0]
    lusc_idx = np.where(subtypes == 1)[0]
    rng.shuffle(luad_idx)
    rng.shuffle(lusc_idx)

    folds_luad = np.array_split(luad_idx, k)
    folds_lusc = np.array_split(lusc_idx, k)

    splits = []
    for i in range(k):
        val_idx = np.concatenate([folds_luad[i], folds_lusc[i]])
        train_idx = np.concatenate([idx for j, idx in enumerate(folds_luad + folds_lusc) if j % k != i and j < k] +
                                   [idx for j, idx in enumerate(folds_luad + folds_lusc) if j % k != i and j >= k])
        # Actually: just take all except fold i
        all_luad_except_i = np.concatenate([folds_luad[j] for j in range(k) if j != i])
        all_lusc_except_i = np.concatenate([folds_lusc[j] for j in range(k) if j != i])
        train_idx = np.concatenate([all_luad_except_i, all_lusc_except_i])

        train_pids = pids[train_idx].tolist()
        val_pids = pids[val_idx].tolist()
        splits.append({"train": train_pids, "val": val_pids})
        print(f"  Fold {i}: train={len(train_pids)}, val={len(val_pids)}")
    return splits


def save_splits(nsclc_df, splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nsclc_df.to_csv(os.path.join(out_dir, "tcga_nsclc_survival.csv"), index=False)
    print(f"[save] Survival CSV saved to {out_dir}/tcga_nsclc_survival.csv")

    for i, split in enumerate(splits):
        rows = []
        max_len = max(len(split["train"]), len(split["val"]))
        for j in range(max_len):
            t = split["train"][j] if j < len(split["train"]) else ""
            v = split["val"][j] if j < len(split["val"]) else ""
            rows.append({"": j, "train": t, "val": v})
        df = pd.DataFrame(rows).set_index("")
        df.to_csv(os.path.join(out_dir, f"splits_{i}.csv"))
    print(f"[save] {len(splits)} split files saved.")


if __name__ == "__main__":
    print("=== Preparing TCGA-NSCLC survival dataset ===")
    
    # 1. Download LUSC clinical data
    print("\n[1] Fetching LUSC clinical data from GDC...")
    hits = fetch_lusc_clinical()
    lusc_surv = parse_survival(hits)
    
    # 2. Add pathology IDs via GDC sample sheet
    print("\n[2] Cross-referencing with slide IDs...")
    lusc_with_slides = add_pathology_ids(lusc_surv, LUSC_GDC_TSV, LUNG_FEATS)
    
    # 3. Combine LUAD + LUSC
    print("\n[3] Building NSCLC dataset...")
    nsclc_df = create_nsclc_csv(LUAD_CSV, lusc_with_slides)
    
    # 4. Create 5-fold CV splits
    print("\n[4] Creating 5-fold CV splits (stratified by subtype)...")
    splits = create_stratified_kfold_splits(nsclc_df, k=5, seed=42)
    
    # 5. Save
    print(f"\n[5] Saving to {OUT_DIR}...")
    save_splits(nsclc_df, splits, OUT_DIR)
    
    print("\n=== Done! ===")
