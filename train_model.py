"""
train_model.py – FINAL OPTIMIZED VERSION
Best configuration:
- PCA: hybrid (min 60 comps OR 99% variance, whichever is larger)
- Balancing: SMOTE + class_weight='balanced'
- Feature extraction: MEDIAN spectrum (+ Savitzky–Golay smoothing)
- Stronger SVM grid search
- Robust visualizations
"""

import os, json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import spectral.io.envi as envi

# Optional smoothing
from scipy.signal import savgol_filter

# SMOTE oversampling
from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ===============================
#        DATA LOADER
# ===============================
class HyperspectralDataLoader:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.class_mapping = {"healthy": 0, "biotic": 1, "abiotic": 2}
        self.class_names = ["Healthy", "Biotic", "Abiotic"]

    def load_cube(self, hdr_path):
        hdr_path = str(hdr_path)
        # try .img then .dat
        for ext in (".img", ".dat"):
            candidate = hdr_path.replace(".hdr", ext)
            if os.path.exists(candidate):
                try:
                    img = envi.open(hdr_path, candidate)
                    return img.load().astype(np.float32)
                except:
                    pass
        return None

    def extract_median_spectrum(self, cube):
        spec = np.median(cube, axis=(0, 1))

        # Stronger smoothing
        window = 11 if cube.shape[2] >= 11 else (cube.shape[2] // 2 * 2 + 1)
        if window >= 5:
            spec = savgol_filter(spec, window_length=window, polyorder=3)
        return spec

    def load_dataset(self):
        X, y, files = [], [], []

        print("Loading dataset:", self.data_folder)
        for cname, label in self.class_mapping.items():
            folder = self.data_folder / cname
            if not folder.exists():
                continue

            hdrs = sorted([f for f in os.listdir(folder) if f.endswith(".hdr")])
            print(f"{cname}: {len(hdrs)} samples")

            for h in tqdm(hdrs, desc=f"Loading {cname}"):
                hdr_path = folder / h
                cube = self.load_cube(hdr_path)
                if cube is None:
                    continue

                spec = self.extract_median_spectrum(cube)
                X.append(spec)
                y.append(label)
                files.append(h)

        X = np.array(X)
        y = np.array(y)
        print("Loaded X:", X.shape, " y:", y.shape)
        return X, y


# ===============================
#         PLOTTING HELPERS
# ===============================
def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)

def plot_conf_matrix(cm, labels, path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    save_plot(path)

def plot_roc_curves(y_true, y_prob, labels, path):
    plt.figure(figsize=(7, 6))
    for i, cls in enumerate(labels):
        y_bin = (y_true == i).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            auc = roc_auc_score(y_bin, y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
        except:
            continue

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    save_plot(path)


# ===============================
#        MAIN TRAINING
# ===============================
def main():
    DATA = "processed_data"
    SAVE = "saved_models"
    VIS = os.path.join(SAVE, "visuals")
    os.makedirs(SAVE, exist_ok=True)
    os.makedirs(VIS, exist_ok=True)

    loader = HyperspectralDataLoader(DATA)
    X, y = loader.load_dataset()

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # PCA — hybrid strategy
    pca_full = PCA(n_components=0.99, random_state=42)
    pca_full.fit(X_train_s)

    MIN_COMPONENTS = 60
    components = max(MIN_COMPONENTS, pca_full.n_components_)

    print(f"PCA 99% variance → {pca_full.n_components_}, using {components} components")

    pca = PCA(n_components=components, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    # Apply SMOTE
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_p, y_train = sm.fit_resample(X_train_p, y_train)

    # Stronger SVM grid
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")

    param_grid = {
        "C": [0.1, 1, 10, 50, 100, 200],
        "gamma": ["scale", 0.1, 0.01, 0.001, 0.0001],
    }

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_p, y_train)
    model = grid.best_estimator_

    print("Best Params:", grid.best_params_)

    # Evaluation function
    def evaluate(name, Xp, yt):
        yp = model.predict(Xp)
        yp_prob = model.predict_proba(Xp)

        cm = confusion_matrix(yt, yp)
        plot_conf_matrix(cm, loader.class_names, os.path.join(VIS, f"cm_{name}.png"))

        try:
            plot_roc_curves(yt, yp_prob, loader.class_names, os.path.join(VIS, f"roc_{name}.png"))
        except:
            pass

        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="weighted")
        print(f"[{name}] Acc={acc:.4f}  F1={f1:.4f}")
        print(classification_report(yt, yp, target_names=loader.class_names))
        return acc, f1

    train_acc, train_f1 = evaluate("train", X_train_p, y_train)
    val_acc, val_f1 = evaluate("val", X_val_p, y_val)
    test_acc, test_f1 = evaluate("test", X_test_p, y_test)

    # Save models
    joblib.dump(model, os.path.join(SAVE, "svm.pkl"))
    joblib.dump(scaler, os.path.join(SAVE, "scaler.pkl"))
    joblib.dump(pca, os.path.join(SAVE, "pca.pkl"))

    metadata = {
        "class_names": loader.class_names,
        "pca_components": int(components),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }
    with open(os.path.join(SAVE, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Training complete. Models saved to saved_models/")

if __name__ == "__main__":
    main()
