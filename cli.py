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
import joblib

# Import your modules
import hyperspectral_preprocessing
import train_model
from predict import (
    load_models,
    predict_new_image,
    predict_batch
)

# ----------------------------------------------------------------------
# Utility printing
# ----------------------------------------------------------------------

def header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# ----------------------------------------------------------------------
# PREPROCESS COMMAND
# ----------------------------------------------------------------------

def run_preprocess(args):
    header("RUNNING PREPROCESSING PIPELINE")

    base_path = "."

    raw_folder = os.path.join(base_path, "raw_images")
    hdr_folder = os.path.join(base_path, "raw_hdr_data")
    processed_folder = os.path.join(base_path, "processed_data")

    print("Raw images folder:      ", raw_folder)
    print("Raw HDR folder:         ", hdr_folder)
    print("Processed output folder:", processed_folder)

    # Step 1 — Convert RAW to reduced ENVI
    hyperspectral_preprocessing.convert_raw_to_envi(
        raw_folder, hdr_folder, processed_folder
    )

    # Step 2 — Auto-create labels.json
    hyperspectral_preprocessing.create_label_file(processed_folder)

    print("\nPreprocessing complete ✔")
    print("➡ Now update labels.json OR use auto-labeling if available.")

# ----------------------------------------------------------------------
# TRAIN COMMAND
# ----------------------------------------------------------------------

def run_train(args):
    header("TRAINING ML MODEL")

    train_model.main()

    print("\nTraining completed ✔")
    print("Saved models available in: saved_models/")

# ----------------------------------------------------------------------
# PREDICT COMMAND
# ----------------------------------------------------------------------

def run_predict(args):
    header("PREDICTION")

    model, scaler, pca, metadata = load_models()
    class_names = metadata["class_names"]

    input_path = args.path

    if args.batch:
        if not os.path.isdir(input_path):
            print(f"[ERROR] {input_path} is not a folder.")
            return
        predict_batch(input_path, model, scaler, pca, class_names)
    else:
        if not os.path.isfile(input_path):
            print(f"[ERROR] {input_path} is not a valid file.")
            return

        pred_class, conf, all_probs = predict_new_image(
            input_path, model, scaler, pca, class_names
        )

        if pred_class:
            print("\nPrediction Result")
            print("-"*50)
            print("Image:       ", input_path)
            print("Class:       ", pred_class)
            print(f"Confidence:   {conf:.4f} ({conf*100:.2f}%)")
            print("\nClass probabilities:")
            for cls, p in all_probs.items():
                print(f"  {cls}: {p:.4f} ({p*100:.2f}%)")

# ----------------------------------------------------------------------
# INFO COMMAND
# ----------------------------------------------------------------------

def run_info(args):
    header("MODEL INFO")

    model_path = "saved_models/model_metadata.json"

    if not os.path.exists(model_path):
        print("No trained model found. Train it using:\n\n  python cli.py train\n")
        return

    with open(model_path, "r") as f:
        meta = json.load(f)

    print(json.dumps(meta, indent=4))

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
    p = sub.add_parser("preprocess",
                       help="Convert RAW → ENVI and create labels.json")
    p.set_defaults(func=run_preprocess)

    # train
    t = sub.add_parser("train",
                       help="Train SVM model on processed_data/")
    t.set_defaults(func=run_train)

    # predict
    pr = sub.add_parser("predict",
                        help="Predict class for a file or folder")
    pr.add_argument("path", help="Path to .hdr file or folder")
    pr.add_argument("--batch", action="store_true",
                    help="Process all .hdr files in folder")
    pr.set_defaults(func=run_predict)

    # info
    i = sub.add_parser("info",
                       help="Show saved model metadata")
    i.set_defaults(func=run_info)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
