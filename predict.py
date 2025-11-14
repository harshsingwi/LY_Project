"""
Prediction module for Grapevine Hyperspectral Classification
Uses: SVM + PCA + StandardScaler
Supports: Single image & batch prediction
"""

import os
import json
import joblib
import numpy as np
import spectral.io.envi as envi


# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------

def load_models(model_folder="saved_models"):
    """Load trained SVM model, PCA, Scaler, and metadata."""

    model_path = os.path.join(model_folder, "svm.pkl")
    scaler_path = os.path.join(model_folder, "scaler.pkl")
    pca_path = os.path.join(model_folder, "pca.pkl")
    metadata_path = os.path.join(model_folder, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("svm.pkl not found. Train the model first.")

    print("\nLoading trained models...")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Models loaded successfully ✔")
    print(f"PCA components:       {metadata['n_components']}")
    print(f"Test set accuracy:    {metadata['test_accuracy']:.4f}")

    return model, scaler, pca, metadata


# ----------------------------------------------------------------------
# IMAGE LOADING
# ----------------------------------------------------------------------

def load_hyperspectral_image(hdr_path):
    """Load ENVI hyperspectral image using .hdr and .img."""

    try:
        img_path = hdr_path.replace(".hdr", ".img")
        img = envi.open(hdr_path, img_path)
        cube = img.load()
        return cube
    except Exception as e:
        print(f"[ERROR] Failed to load image: {hdr_path} → {e}")
        return None


# ----------------------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------------------

def extract_mean_spectrum(cube):
    """Extract mean reflectance (bands only)."""
    return np.mean(cube, axis=(0, 1))


# ----------------------------------------------------------------------
# SINGLE IMAGE PREDICTION
# ----------------------------------------------------------------------

def predict_single_image(hdr_path, model, scaler, pca, class_names):
    """
    Predict class for ONE hyperspectral image.

    Returns:
        predicted_class (str)
        confidence (float)
        prob_dict (dict)
    """

    print(f"\nProcessing image: {hdr_path}")

    cube = load_hyperspectral_image(hdr_path)
    if cube is None:
        return None, 0.0, {}

    spectrum = extract_mean_spectrum(cube)
    X = spectrum.reshape(1, -1)

    # Preprocessing
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Prediction
    probs = model.predict_proba(X_pca)[0]
    label = model.predict(X_pca)[0]

    predicted_class = class_names[label]
    confidence = probs[label]

    prob_dict = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    return predicted_class, confidence, prob_dict


# ----------------------------------------------------------------------
# BATCH PREDICTION
# ----------------------------------------------------------------------

def predict_batch(folder_path, model, scaler, pca, class_names):
    """Predict all .hdr images inside a folder."""

    hdr_files = [f for f in os.listdir(folder_path) if f.endswith(".hdr")]

    if len(hdr_files) == 0:
        print(f"\n[ERROR] No .hdr files found in folder: {folder_path}")
        return

    print(f"\nPredicting {len(hdr_files)} images from {folder_path}...")
    print("=" * 70)

    results = []

    for file in hdr_files:
        hdr_path = os.path.join(folder_path, file)

        pred_class, conf, prob_dict = predict_single_image(
            hdr_path, model, scaler, pca, class_names
        )

        if pred_class is None:
            continue

        result = {
            "filename": file,
            "predicted_class": pred_class,
            "confidence": conf,
            "probabilities": prob_dict
        }
        results.append(result)

        print("\n" + "-" * 60)
        print("File:", file)
        print("Predicted:", pred_class)
        print(f"Confidence: {conf:.4f} ({conf*100:.2f}%)")
        print("Probabilities:")
        for cls, p in prob_dict.items():
            print(f"  {cls}: {p:.4f}")

    # Save batch result
    output_file = os.path.join(folder_path, "predictions.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\nPredictions saved to:", output_file)
    print("=" * 70)
