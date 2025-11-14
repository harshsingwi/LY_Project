"""
Hyperspectral Preprocessing Pipeline
-----------------------------------
✔ Converts RAW → ENVI (reduced)
✔ Reads description-2.csv
✔ Maps symptoms → (healthy / biotic / abiotic)
✔ Automatically organizes processed_data/
✔ No manual labeling needed
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from spectral import *
import spectral.io.envi as envi


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
# 2. RAW → ENVI REDUCED CONVERSION
# =====================================================================
def convert_raw_to_envi(raw_folder, hdr_folder, output_folder):
    """
    Convert RAW files to ENVI format and reduce spatial dimensions.
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
                        cube[i*block:(i+1)*block, j*block:(j+1)*block, :],
                        axis=(0, 1)
                    )

            out_hdr = os.path.join(output_folder, base + ".hdr")
            metadata = {
                "lines": new_r,
                "samples": new_c,
                "bands": bands,
                "data type": 4,
                "interleave": "bsq",
                "byte order": 0
            }

            if "wavelength" in img.metadata:
                metadata["wavelength"] = img.metadata["wavelength"]

            envi.save_image(out_hdr, reduced, metadata=metadata, force=True)

        except Exception as e:
            print(f"[ERROR] Failed {raw_file}: {e}")


# =====================================================================
# 3. LOAD CSV AND MAP LABELS
# =====================================================================
def load_labels_from_csv(csv_path):
    """
    Reads description-2.csv and returns: { image_id → class }
    """
    print("\nLoading CSV:", csv_path)
    df = pd.read_csv(csv_path, sep=";", encoding="latin1")

    label_map = {}

    for _, row in df.iterrows():
        directory = str(row["directoryName"]).strip()
        symptom = str(row["symptom"]).strip().lower()

        if symptom not in SYMPTOM_TO_CLASS:
            print(f"[WARN] Unmapped symptom found: {symptom} → set to abiotic")
            mapped = "abiotic"  # default fallback
        else:
            mapped = SYMPTOM_TO_CLASS[symptom]

        label_map[directory] = mapped

    print(f"Loaded {len(label_map)} labels from CSV.")
    return label_map


# =====================================================================
# 4. ORGANIZE FILES BASED ON CSV LABELS
# =====================================================================
def organize_by_csv(processed_folder, csv_path):
    """
    Moves ENVI files (.hdr + .img/.dat) into class folders based on CSV mapping.
    """
    print("\nOrganizing processed files based on CSV labels...")

    labels = load_labels_from_csv(csv_path)

    folders = {
        "healthy": os.path.join(processed_folder, "healthy"),
        "biotic": os.path.join(processed_folder, "biotic"),
        "abiotic": os.path.join(processed_folder, "abiotic")
    }
    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    moved = {"healthy": 0, "biotic": 0, "abiotic": 0}

    hdr_files = [f for f in os.listdir(processed_folder) if f.endswith(".hdr")]

    for hdr in hdr_files:
        base = hdr.replace(".hdr", "")

        if base not in labels:
            print(f"[WARN] {base} not found in CSV — SKIPPED")
            continue

        class_label = labels[base]
        dst_folder = folders[class_label]

        # Move both .hdr and .img/.dat
        for ext in [".hdr", ".img", ".dat"]:
            src = os.path.join(processed_folder, base + ext)
            if os.path.exists(src):
                dst = os.path.join(dst_folder, base + ext)
                os.rename(src, dst)

        moved[class_label] += 1

    print("\nOrganization complete:")
    for k, v in moved.items():
        print(f"  {k}: {v} files moved")

    return moved


# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================
def main():
    base = os.getcwd()

    raw_folder = os.path.join(base, "raw_images")
    hdr_folder = os.path.join(base, "raw_hdr_data")
    processed = os.path.join(base, "processed_data")
    csv_path = os.path.join(base, "description-2.csv")

    print("="*70)
    print("HYPERSPECTRAL PREPROCESSING PIPELINE (AUTO-LABELED)")
    print("="*70)

    # Step 1 — Convert & Reduce
    convert_raw_to_envi(raw_folder, hdr_folder, processed)

    # Step 2 — Organize using CSV
    organize_by_csv(processed, csv_path)

    print("\nPreprocessing COMPLETE ✓")
    print("Processed data ready for:  python train_model.py")
    print("="*70)


if __name__ == "__main__":
    main()
