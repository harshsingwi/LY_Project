"""
Hyperspectral Image Preprocessing Module
Auto-labels using CSV + converts RAW->ENVI + organizes into classes
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
import spectral.io.envi as envi
from tqdm import tqdm

###############################################
# 1. Convert RAW → Reduced ENVI
###############################################

def convert_raw_to_envi(raw_folder, hdr_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    raw_files = sorted([f for f in os.listdir(raw_folder) if f.endswith('.raw')])
    print(f"Found {len(raw_files)} RAW files")

    for raw_file in tqdm(raw_files, desc="Converting RAW → ENVI"):
        base_name = raw_file.replace('.raw', '')
        raw_path = os.path.join(raw_folder, raw_file)
        hdr_path = os.path.join(hdr_folder, f"{base_name}.hdr")

        if not os.path.exists(hdr_path):
            print(f"⚠ HDR missing for {raw_file}")
            continue

        try:
            img = envi.open(hdr_path, raw_path)
            data = img.load()

            rows, cols, bands = data.shape
            block = 4
            new_rows = rows // block
            new_cols = cols // block

            reduced = np.zeros((new_rows, new_cols, bands), dtype=np.float32)

            for i in range(new_rows):
                for j in range(new_cols):
                    region = data[i*block:(i+1)*block, j*block:(j+1)*block, :]
                    reduced[i, j, :] = np.mean(region, axis=(0, 1))

            # Save ENVI file (.hdr + .img)
            out_hdr = os.path.join(output_folder, f"{base_name}.hdr")

            metadata = {
                'lines': new_rows,
                'samples': new_cols,
                'bands': bands,
                'data type': 4,
                'interleave': 'bsq',
                'byte order': 0
            }

            if 'wavelength' in img.metadata:
                metadata['wavelength'] = img.metadata['wavelength']

            envi.save_image(out_hdr, reduced, metadata=metadata, force=True)

        except Exception as e:
            print(f"Error processing {raw_file}: {e}")

###############################################
# 2. AUTO CREATE LABELS USING CSV
###############################################

def normalize_symptom(symptom):
    """Map ANY symptom into healthy / biotic / abiotic"""

    symptom = symptom.lower().strip()

    # 1. Healthy
    if symptom == "healthy":
        return "healthy"
    
    # 2. Biotic keywords (disease / infection)
    biotic_keywords = [
        "flavescence", "esca", "mildew", "rot", "virus", "fung", "infection", "pest"
    ]
    if any(k in symptom for k in biotic_keywords):
        return "biotic"

    # 3. Abiotic keywords (stress / deficiency)
    abiotic_keywords = [
        "stress", "deficiency", "drought", "sunburn", "chlorosis",
        "nutrient", "temperature", "heat", "cold"
    ]
    if any(k in symptom for k in abiotic_keywords):
        return "abiotic"

    # Default fallback
    return "unknown"

def create_label_file_from_csv(processed_folder, csv_file, labels_file="labels.json"):

    processed_folder = Path(processed_folder)

    # Load CSV
    mapping = {}
    with open(csv_file, encoding="latin1") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            dirname = row["directoryName"].strip()
            symptom = row["symptom"].strip()

            cls = normalize_symptom(symptom)
            mapping[dirname] = cls

    # Create labels.json
    labels = {}
    hdr_files = sorted([f for f in os.listdir(processed_folder) if f.endswith(".hdr")])

    for hdr in hdr_files:
        key = hdr.replace(".hdr", "")

        cls = mapping.get(key, "unknown")

        labels[key] = {
            "class": cls,
            "notes": ""
        }

    labels_path = processed_folder / labels_file
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)

    print(f"\n✔ Auto-labeled using CSV. Saved: {labels_path}")

###############################################
# 3. ORGANIZE INTO FOLDERS
###############################################

def organize_by_labels(processed_folder, labels_file='labels.json'):
    processed_folder = Path(processed_folder)
    labels_path = processed_folder / labels_file

    if not labels_path.exists():
        print("❌ labels.json missing")
        return

    with open(labels_path) as f:
        labels = json.load(f)

    class_dirs = {
        "healthy": processed_folder/"healthy",
        "biotic": processed_folder/"biotic",
        "abiotic": processed_folder/"abiotic",
        "unknown": processed_folder/"unknown"
    }
    for d in class_dirs.values():
        d.mkdir(exist_ok=True)

    for name, info in labels.items():
        cls = info["class"]

        src_hdr = processed_folder / f"{name}.hdr"
        src_img = processed_folder / f"{name}.img"

        dst = class_dirs.get(cls, class_dirs["unknown"])

        if src_hdr.exists():
            src_hdr.rename(dst / src_hdr.name)
        if src_img.exists():
            src_img.rename(dst / src_img.name)

    print("✔ Organization complete.")

###############################################
# 4. MAIN
###############################################

if __name__ == "__main__":

    base = os.path.dirname(os.path.abspath(__file__))

    raw_folder = os.path.join(base, "raw_images")
    hdr_folder = os.path.join(base, "raw_hdr_data")
    processed = os.path.join(base, "processed_data")
    csv_file = os.path.join(base, "description-2.csv")

    print("\n==== HYPERSPECTRAL PREPROCESSING ====")

    convert_raw_to_envi(raw_folder, hdr_folder, processed)

    create_label_file_from_csv(processed, csv_file)

    organize_by_labels(processed)

    print("\n==== DONE ====\n")
