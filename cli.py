"""
cli.py — Command Line Interface for Grapevine Hyperspectral ML Pipeline

Commands:
    python cli.py preprocess
    python cli.py train
    python cli.py predict <file_or_folder> [--batch]
    python cli.py info
"""

import argparse
import os
import sys
import json

# Import project modules
import hyperspectral_preprocessing
import train_model

from predict import (
    load_models,
    predict_single_image,
    predict_batch
)

# ----------------------------------------------------------------------
# Utility print helpers
# ----------------------------------------------------------------------

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ----------------------------------------------------------------------
# PREPROCESS COMMAND
# ----------------------------------------------------------------------

def run_preprocess(args):
    header("RUNNING PREPROCESSING PIPELINE")

    base = "."

    raw_folder = os.path.join(base, "raw_images")
    hdr_folder = os.path.join(base, "raw_hdr_data")
    processed_folder = os.path.join(base, "processed_data")

    print("Raw images folder:      ", raw_folder)
    print("Raw HDR folder:         ", hdr_folder)
    print("Processed output folder:", processed_folder)

    # Convert RAW → ENVI
    hyperspectral_preprocessing.convert_raw_to_envi(
        raw_folder, hdr_folder, processed_folder
    )

    # Create label file (labels.json)
    hyperspectral_preprocessing.create_label_file(processed_folder)

    print("\nPreprocessing complete ✔")
    print("If using auto-labeling, now update labels.json accordingly.")

# ----------------------------------------------------------------------
# TRAIN COMMAND
# ----------------------------------------------------------------------

def run_train(args):
    header("TRAINING MODEL")
    train_model.main()
    print("\nTraining completed ✔")
    print("Saved models are inside: saved_models/")

# ----------------------------------------------------------------------
# PREDICT COMMAND
# ----------------------------------------------------------------------

def run_predict(args):
    header("RUNNING PREDICTION")

    model, scaler, pca, metadata = load_models()
    class_names = metadata["class_names"]

    input_path = args.path

    # Batch mode
    if args.batch:
        if not os.path.isdir(input_path):
            print(f"[ERROR] Folder not found: {input_path}")
            return

        predict_batch(input_path, model, scaler, pca, class_names)
        return

    # Single image mode
    else:
        if not os.path.exists(input_path):
            print(f"[ERROR] File not found: {input_path}")
            return

        pred_class, conf, prob_dict = predict_single_image(
            input_path, model, scaler, pca, class_names
        )

        if pred_class is None:
            print("[ERROR] Prediction failed.")
            return

        print("\n--- Prediction Result ---")
        print("File:         ", input_path)
        print("Predicted:    ", pred_class)
        print(f"Confidence:    {conf:.4f} ({conf*100:.2f}%)")
        print("\nClass Probabilities:")
        for cls, p in prob_dict.items():
            print(f"  {cls}: {p:.4f} ({p*100:.2f}%)")

# ----------------------------------------------------------------------
# INFO COMMAND
# ----------------------------------------------------------------------

def run_info(args):
    header("MODEL INFO")

    metadata_path = os.path.join("saved_models", "model_metadata.json")

    if not os.path.exists(metadata_path):
        print("No model found. Train one using:\n  python cli.py train\n")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(json.dumps(metadata, indent=4))

# ----------------------------------------------------------------------
# ARGPARSE SETUP
# ----------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="CLI for Grapevine Hyperspectral ML Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p = sub.add_parser("preprocess", help="Convert RAW → ENVI & create labels.json")
    p.set_defaults(func=run_preprocess)

    # train
    t = sub.add_parser("train", help="Train SVM model on processed_data/")
    t.set_defaults(func=run_train)

    # predict
    pr = sub.add_parser("predict", help="Predict class for file or folder")
    pr.add_argument("path", help="Path to .hdr file or folder")
    pr.add_argument("--batch", action="store_true", help="Run prediction for all .hdr files in folder")
    pr.set_defaults(func=run_predict)

    # info
    i = sub.add_parser("info", help="Show metadata of trained model")
    i.set_defaults(func=run_info)

    return parser

# ----------------------------------------------------------------------
# MAIN ENTRY
# ----------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
