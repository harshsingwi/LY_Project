"""
Hyperspectral Preprocessing Pipeline
-----------------------------------
✔ Converts RAW → ENVI (reduced)
✔ Reads description-2.csv
✔ Maps symptoms → (healthy / biotic / abiotic)
✔ Automatically splits processed_data/ into train/val/test
✔ Train & val: labeled into class folders
✔ Test: kept unlabeled (no class folders)
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from spectral import *
import spectral.io.envi as envi
from sklearn.model_selection import train_test_split


# =====================================================================
# 1. SYMPTOM → CLASS MAPPING
# =====================================================================
SYMPTOM_TO_CLASS = {
    "healthy": "healthy",

    # Biotic stresses (disease, pests, fungi)
    "flavescence dorée": "biotic",
    "buffalo treehopper": "biotic",
    "green leafhopper": "biotic",
    "wood diseases": "biotic",
    "mildew": "biotic",
    "damaged": "biotic",
    "discoloration": "biotic",

    # Abiotic stresses
    "water stress": "abiotic",
    "deficiency": "abiotic",
    "chlorosis": "abiotic",
    "senescence": "abiotic"
}


# =====================================================================
# 2. HELPERS
# =====================================================================
def _normalize_symptom(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("dor�e", "dorée")
    return t


# =====================================================================
# 3. RAW → ENVI REDUCED CONVERSION
# =====================================================================
def convert_raw_to_envi(raw_folder, hdr_folder, output_folder):
    """
    Convert RAW files to ENVI format and reduce spatial dimensions.
    All reduced .hdr/.img are saved flat inside `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)
    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith(".raw")])

    print(f"\nFound {len(raw_files)} RAW files to process.")

    for raw_file in tqdm(raw_files, desc="Converting RAW → ENVI"):
        base = raw_file.replace(".raw", "")
        raw_path = os.path.join(raw_folder, raw_file)
        hdr_path = os.path.join(hdr_folder, base + ".hdr")

        if not os.path.exists(hdr_path):
            print(f"[WARN] Missing HDR for {raw_file}")
            continue

        try:
            img = envi.open(hdr_path, raw_path)
            cube = img.load()

            rows, cols, bands = cube.shape
            block = 4
            new_r, new_c = rows // block, cols // block

            reduced = np.zeros((new_r, new_c, bands), dtype=np.float32)
            for i in range(new_r):
                for j in range(new_c):
                    reduced[i, j] = np.mean(
                        cube[i * block:(i + 1) * block,
                             j * block:(j + 1) * block, :],
                        axis=(0, 1),
                    )

            out_hdr = os.path.join(output_folder, base + ".hdr")
            metadata = {
                "lines": new_r,
                "samples": new_c,
                "bands": bands,
                "data type": 4,
                "interleave": "bsq",
                "byte order": 0,
            }

            if "wavelength" in img.metadata:
                metadata["wavelength"] = img.metadata["wavelength"]

            envi.save_image(out_hdr, reduced, metadata=metadata, force=True)

        except Exception as e:
            print(f"[ERROR] Failed {raw_file}: {e}")


# =====================================================================
# 4. LOAD CSV AND MAP LABELS
# =====================================================================
def load_labels_from_csv(csv_path):
    """
    Reads description-2.csv and returns: { image_id (e.g. 2020-09-10_084) → class }

    Uses the SECOND column as image id (matches RAW/HDR filenames),
    and the 'symptom' column (handles encoding issues).
    """
    print("\nLoading CSV:", csv_path)
    df = pd.read_csv(csv_path, sep=";", encoding="latin1")

    # Symptom column
    if "symptom" in df.columns:
        symptom_col = "symptom"
    else:
        non_num = [c for c in df.columns if df[c].dtype == "object"]
        symptom_col = non_num[-1]

    # ID column (second column) – should look like 2020-09-10_084
    id_col = df.columns[1]

    label_map = {}

    for _, row in df.iterrows():
        image_id = str(row[id_col]).strip()
        if not image_id:
            continue

        symptom_raw = row[symptom_col]
        symptom = _normalize_symptom(symptom_raw)

        if symptom not in SYMPTOM_TO_CLASS:
            print(f"[WARN] Unmapped symptom: '{symptom_raw}' → set to abiotic")
            mapped = "abiotic"
        else:
            mapped = SYMPTOM_TO_CLASS[symptom]

        label_map[image_id] = mapped

    print(f"Loaded {len(label_map)} labels from CSV.")
    return label_map


# =====================================================================
# 5. ORGANIZE FILES BASED ON CSV LABELS + SPLIT
# =====================================================================
def organize_by_csv(
    processed_folder,
    csv_path,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """
    1. Reads labels from CSV.
    2. Matches processed_data/*.hdr with CSV image ids like 2020-09-10_084.
    3. Stratified split into train / val / test:
         - train & val: organized into class folders
         - test: kept unlabeled in processed_data/test/

    Final structure:
        processed_data/
            train/healthy|biotic|abiotic
            val/healthy|biotic|abiotic
            test/*.hdr, *.img, *.dat
    """
    print("\nOrganizing processed files based on CSV labels...")

    labels = load_labels_from_csv(csv_path)

    hdr_files = [f for f in os.listdir(processed_folder) if f.endswith(".hdr")]

    bases = []
    y = []

    for hdr in hdr_files:
        base = hdr.replace(".hdr", "")

        if base not in labels:
            print(f"[WARN] {base} not found in CSV — SKIPPED")
            continue

        bases.append(base)
        y.append(labels[base])

    if not bases:
        print("[ERROR] No files matched between processed_data and CSV.")
        return {}

    bases = np.array(bases)
    y = np.array(y)

    print(f"Matched {len(bases)} files with labels from CSV.")

    # --- Train / Test split ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        bases,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # --- Train / Val split ---
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )

    print("\nSplit counts:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)} (kept unlabeled in 'test/')")

    # --- Create folder structure ---
    splits_root = processed_folder
    train_root = os.path.join(splits_root, "train")
    val_root = os.path.join(splits_root, "val")
    test_root = os.path.join(splits_root, "test")

    for p in [train_root, val_root, test_root]:
        os.makedirs(p, exist_ok=True)

    class_names = ["healthy", "biotic", "abiotic"]
    for cname in class_names:
        os.makedirs(os.path.join(train_root, cname), exist_ok=True)
        os.makedirs(os.path.join(val_root, cname), exist_ok=True)

    def move_set(bases_subset, labels_subset, split_name):
        moved_counts = {"healthy": 0, "biotic": 0, "abiotic": 0, "test": 0}
        for base, lbl in zip(bases_subset, labels_subset):
            if split_name == "test":
                dst_dir = test_root
                cls = "test"
            else:
                dst_dir = os.path.join(
                    train_root if split_name == "train" else val_root,
                    lbl,
                )
                cls = lbl

            for ext in [".hdr", ".img", ".dat"]:
                src = os.path.join(processed_folder, base + ext)
                if os.path.exists(src):
                    dst = os.path.join(dst_dir, base + ext)
                    shutil.move(src, dst)
            moved_counts[cls] += 1
        return moved_counts

    train_counts = move_set(X_train, y_train, "train")
    val_counts = move_set(X_val, y_val, "val")
    test_counts = move_set(X_test, y_test, "test")

    print("\nOrganization complete:")
    print("  Train:", train_counts)
    print("  Val:  ", val_counts)
    print("  Test: ", test_counts)

    return {
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
    }


# =====================================================================
# 6. MAIN ENTRY POINT (optional standalone)
# =====================================================================
def main():
    base = os.getcwd()

    raw_folder = os.path.join(base, "raw_images")
    hdr_folder = os.path.join(base, "raw_hdr_data")
    processed = os.path.join(base, "processed_data")
    csv_path = os.path.join(base, "description-2.csv")

    print("=" * 70)
    print("HYPERSPECTRAL PREPROCESSING PIPELINE (AUTO-LABELED + SPLIT)")
    print("=" * 70)

    # Step 1 — Convert & Reduce
    convert_raw_to_envi(raw_folder, hdr_folder, processed)

    # Step 2 — Organize using CSV + split
    organize_by_csv(processed, csv_path)

    print("\nPreprocessing COMPLETE ✓")
    print("Train/Val/Test ready in: processed_data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
