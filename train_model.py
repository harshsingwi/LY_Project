"""
train_model.py – presentation-friendly version

- Uses ALL labeled data: processed_data/train + processed_data/val
- Feature extraction: MEDIAN spectrum (+ Savitzky–Golay smoothing)
- PCA: n_components=0.99 (no artificial 60-component floor)
- use SVM class_weight='balanced' instead
- 5-fold CV to pick best SVM
- Confusion matrix & ROC on full labeled set (nice for slides)
"""

import os, json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score
)
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import spectral.io.envi as envi
from scipy.signal import savgol_filter

np.random.seed(42)


# ===============================
#        DATA LOADER
# ===============================
class HyperspectralDataLoader:
    def __init__(self, data_root, split="train"):
        """
        data_root: base folder (e.g. 'processed_data')
        split: 'train' or 'val' – expects:
            processed_data/split/healthy|biotic|abiotic
        """
        self.data_root = Path(data_root)
        self.split = split
        self.base_folder = self.data_root / split
        self.class_mapping = {"healthy": 0, "biotic": 1, "abiotic": 2}
        self.class_names = ["Healthy", "Biotic", "Abiotic"]

    def load_cube(self, hdr_path: Path):
        hdr_path = str(hdr_path)
        for ext in (".img", ".dat"):
            candidate = hdr_path.replace(".hdr", ext)
            if os.path.exists(candidate):
                try:
                    img = envi.open(hdr_path, candidate)
                    return img.load().astype(np.float32)
                except Exception:
                    pass
        return None

    def extract_median_spectrum(self, cube):
        spec = np.median(cube, axis=(0, 1))
        bands = cube.shape[2]
        window = 11 if bands >= 11 else max(5, bands // 2 * 2 + 1)
        if window >= 5 and bands >= window:
            spec = savgol_filter(spec, window_length=window, polyorder=3)
        return spec

    def load_dataset(self):
        X, y = [], []

        print("Loading dataset from:", self.base_folder)
        for cname, label in self.class_mapping.items():
            folder = self.base_folder / cname
            if not folder.exists():
                print(f"[WARN] Class folder missing: {folder}")
                continue

            hdrs = sorted([f for f in os.listdir(folder) if f.endswith(".hdr")])
            print(f"  {self.split}/{cname}: {len(hdrs)} .hdr files")

            for h in tqdm(hdrs, desc=f"Loading {self.split}/{cname}"):
                hdr_path = folder / h
                cube = self.load_cube(hdr_path)
                if cube is None:
                    continue

                spec = self.extract_median_spectrum(cube)
                X.append(spec)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"{self.split.upper()} Loaded X:", X.shape, " y:", y.shape)

        counts = {name: int((y == idx).sum())
                  for name, idx in self.class_mapping.items()}
        print(f"\nSamples per class in {self.split}:")
        for cname, count in counts.items():
            print(f"  {cname}: {count}")
        print()

        for cname, count in counts.items():
            if count == 0:
                print(f"[WARNING] No samples for class '{cname}' in {self.split} split.")

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
        except Exception:
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
    DATA_ROOT = "processed_data"
    SAVE = "saved_models"
    VIS = os.path.join(SAVE, "visuals")
    os.makedirs(SAVE, exist_ok=True)
    os.makedirs(VIS, exist_ok=True)

    # ---- Load TRAIN + VAL and merge ----
    loader_train = HyperspectralDataLoader(DATA_ROOT, split="train")
    X_train, y_train = loader_train.load_dataset()

    loader_val = HyperspectralDataLoader(DATA_ROOT, split="val")
    X_val, y_val = loader_val.load_dataset()

    # Concatenate all labeled data
    X_all = np.vstack([X_train, X_val]) if X_val.size else X_train
    y_all = np.concatenate([y_train, y_val]) if y_val.size else y_train

    if X_all.size == 0:
        print("[ERROR] Not enough data in processed_data/train or val. "
              "Did preprocessing & splitting succeed?")
        return

    print(f"\nUsing TOTAL labeled samples: {len(y_all)}\n")

    # Scaling / z score normalization
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    # PCA – keep 99% variance, no MIN_COMPONENTS floor
    pca_full = PCA(n_components=0.99, random_state=42)
    pca_full.fit(X_all_s)
    components = pca_full.n_components_

    print(f"PCA components for 99% variance: {components}\n")

    pca = PCA(n_components=components, random_state=42)
    X_all_p = pca.fit_transform(X_all_s)

    # SVM with class_weight only 
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")

    param_grid = {
        "C": [0.1, 1, 10, 50, 100],
        "gamma": ["scale", 0.1, 0.01, 0.001],
    }

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_all_p, y_all)
    model = grid.best_estimator_

    print("\nBest Params:", grid.best_params_, "\n")
    print(f"Best CV (f1_weighted): {grid.best_score_:.4f}\n")

    # ---- Evaluate on full labeled set (nice for presentation) ----
    yp = model.predict(X_all_p)
    yp_prob = model.predict_proba(X_all_p)

    cm = confusion_matrix(y_all, yp)
    plot_conf_matrix(
        cm,
        loader_train.class_names,
        os.path.join(VIS, "cm_all.png"),
        title="Confusion Matrix – All Labeled Data",
    )

    try:
        plot_roc_curves(
            y_all,
            yp_prob,
            loader_train.class_names,
            os.path.join(VIS, "roc_all.png"),
        )
    except Exception:
        pass

    acc = accuracy_score(y_all, yp)
    f1 = f1_score(y_all, yp, average="weighted")
    print(f"[ALL] Acc={acc:.4f}  F1={f1:.4f}")
    print(classification_report(y_all, yp, target_names=loader_train.class_names))
    print()

    # Save models
    joblib.dump(model, os.path.join(SAVE, "svm.pkl"))
    joblib.dump(scaler, os.path.join(SAVE, "scaler.pkl"))
    joblib.dump(pca, os.path.join(SAVE, "pca.pkl"))

    metadata = {
        "class_names": loader_train.class_names,
        "pca_components": int(components),
        "all_acc": float(acc),
        "all_f1": float(f1),
        "cv_f1_weighted": float(grid.best_score_),
    }
    with open(os.path.join(SAVE, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Training complete. Models saved to saved_models/")


if __name__ == "__main__":
    main()
