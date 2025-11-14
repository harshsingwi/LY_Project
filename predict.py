"""
Prediction Script for New Hyperspectral Images
Compatible with the updated training pipeline
"""

import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path
import spectral.io.envi as envi

# -----------------------------------------------------------
# LOADING TRAINED MODELS
# -----------------------------------------------------------

def load_models(model_folder='saved_models'):
    """
    Load trained SVM model, scaler, PCA transformer, and metadata
    """
    print("\nLoading trained model...")

    model = joblib.load(os.path.join(model_folder, "svm.pkl"))
    scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
    pca = joblib.load(os.path.join(model_folder, "pca.pkl"))

    # Default class names (same order as training)
    class_names = ["Healthy", "Biotic", "Abiotic"]

    print("✓ Models loaded successfully")
    print("✓ PCA components:", pca.n_components_)

    return model, scaler, pca, class_names


# -----------------------------------------------------------
# HYPERSPECTRAL IMAGE LOADING (hdr + img)
# -----------------------------------------------------------

def load_hyperspectral_image(hdr_path):
    """
    Load ENVI image (.hdr + .img)
    """
    try:
        img_path = hdr_path.replace(".hdr", ".img")
        img = envi.open(hdr_path, img_path)
        cube = img.load()
        return cube
    except Exception as e:
        print(f"[ERROR] Cannot load hyperspectral image: {hdr_path}")
        print("Reason:", str(e))
        return None


def extract_mean_spectrum(cube):
    """Extract mean spectrum (bands,)"""
    return np.mean(cube, axis=(0, 1))


# -----------------------------------------------------------
# SINGLE IMAGE PREDICTION
# -----------------------------------------------------------

def predict_single_image(hdr_path, model, scaler, pca, class_names):
    print(f"\nProcessing file: {hdr_path}")

    cube = load_hyperspectral_image(hdr_path)
    if cube is None:
        return None, None, None

    print(f"Image shape: {cube.shape}")

    spectrum = extract_mean_spectrum(cube).reshape(1, -1)

    X_scaled = scaler.transform(spectrum)
    X_pca = pca.transform(X_scaled)

    pred_label = model.predict(X_pca)[0]
    pred_proba = model.predict_proba(X_pca)[0]

    predicted_class = class_names[pred_label]
    confidence = float(pred_proba[pred_label])

    all_probs = {class_names[i]: float(pred_proba[i]) for i in range(len(class_names))}

    return predicted_class, confidence, all_probs


# -----------------------------------------------------------
# BATCH PREDICTION FOR A FOLDER
# -----------------------------------------------------------

def predict_batch(folder, model, scaler, pca, class_names):
    hdr_files = [f for f in os.listdir(folder) if f.endswith(".hdr")]

    if not hdr_files:
        print("\n[ERROR] No .hdr files found in folder:", folder)
        return

    print(f"\nFound {len(hdr_files)} images in folder '{folder}'")
    print("=" * 70)

    results = []

    for f in hdr_files:
        hdr_path = os.path.join(folder, f)

        predicted_class, confidence, all_probs = predict_single_image(
            hdr_path, model, scaler, pca, class_names
        )

        if predicted_class is None:
            continue

        print("\n-----------------------------------------")
        print(f"File: {f}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

        res = {
            "filename": f,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs
        }
        results.append(res)

    # Save predictions
    out_path = os.path.join(folder, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nPredictions saved to:", out_path)
    print("=" * 70)


# -----------------------------------------------------------
# MAIN SCRIPT ENTRY POINT
# -----------------------------------------------------------

def main():

    if len(sys.argv) < 2:
        print("""
Usage:
  python predict.py <path_to_hdr_file>
  python predict.py <folder_path> --batch

Examples:
  python predict.py test_images/img_001.hdr
  python predict.py test_images/ --batch
""")
        sys.exit(1)

    model, scaler, pca, class_names = load_models()

    input_path = sys.argv[1]

    # Batch mode
    if len(sys.argv) > 2 and sys.argv[2] == "--batch":
        if not os.path.isdir(input_path):
            print("[ERROR] Batch mode requires a folder path")
            return

        predict_batch(input_path, model, scaler, pca, class_names)

    # Single file mode
    else:
        if not os.path.isfile(input_path):
            print("[ERROR] File not found:", input_path)
            return

        pred_class, conf, probs = predict_single_image(input_path, model, scaler, pca, class_names)

        if pred_class is None:
            return

        print("\n============= PREDICTION RESULT =============")
        print("File:", input_path)
        print("Predicted Class:", pred_class)
        print("Confidence:", f"{conf:.4f} ({conf*100:.2f}%)")
        print("All Probabilities:")
        for cls, prob in probs.items():
            print(f"  {cls}: {prob:.4f}")
        print("=============================================")


if __name__ == "__main__":
    main()
