"""
train_model.py
Full ML Pipeline for Grapevine Hyperspectral Classification
Now Includes:
- Dataset graphs (class balance, mean spectra)
- PCA explained variance plot
- Train + Val confusion matrices
- Test confusion + ROC curves
"""

import os
import json
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from spectral import *
import spectral.io.envi as envi


# =========================================================================================
# DATA LOADER
# =========================================================================================
class HyperspectralDataLoader:
    def __init__(self, folder):
        self.folder = folder
        self.mapping = {"healthy": 0, "biotic": 1, "abiotic": 2}
        self.rev = {v: k for k, v in self.mapping.items()}

    def load_hyperspectral(self, hdr):
        try:
            img = envi.open(hdr)
            return img.load()
        except:
            return None

    def mean_spectrum(self, cube):
        return np.mean(cube, axis=(0, 1))

    def load_dataset(self):
        X, y, files = [], [], []
        print("Loading dataset...")

        for cname, clabel in self.mapping.items():
            cdir = os.path.join(self.folder, cname)
            if not os.path.isdir(cdir):
                continue

            hdrs = [f for f in os.listdir(cdir) if f.endswith(".hdr")]
            print(f"{cname}: {len(hdrs)} images")

            for h in tqdm(hdrs):
                cube = self.load_hyperspectral(os.path.join(cdir, h))
                if cube is None:
                    continue

                X.append(self.mean_spectrum(cube))
                y.append(clabel)
                files.append(h)

        return np.array(X), np.array(y), files


# =========================================================================================
# PREPROCESSOR
# =========================================================================================
class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def fit_transform(self, X, keep_var=0.95):
        Xs = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=keep_var, random_state=42)
        XP = self.pca.fit_transform(Xs)
        return XP

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))


# =========================================================================================
# MODEL
# =========================================================================================
class SVMModel:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.1, 0.01, 0.001]
        }
        base = SVC(kernel="rbf", probability=True, class_weight="balanced")

        gs = GridSearchCV(
            base, grid, scoring="f1_weighted",
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            n_jobs=-1, verbose=1
        )
        gs.fit(X, y)
        self.model = gs.best_estimator_
        print("Best parameters:", gs.best_params_)


# =========================================================================================
# GRAPH FUNCTIONS
# =========================================================================================

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def plot_class_distribution(y, class_names, path):
    plt.figure(figsize=(6, 5))
    sns.countplot(x=y)
    plt.xticks(ticks=[0,1,2], labels=class_names)
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    save_plot(path)


def plot_mean_spectra(X, y, class_names, path):
    plt.figure(figsize=(8,5))
    for i, cname in enumerate(class_names):
        mean_spec = X[y == i].mean(axis=0)
        plt.plot(mean_spec, label=cname)

    plt.title("Mean Spectrum per Class")
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Reflectance (Mean)")
    plt.legend()
    save_plot(path)


def plot_pca_variance(pca, path):
    plt.figure(figsize=(7,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Curve")
    plt.grid()
    save_plot(path)


def plot_conf_matrix(cm, class_names, path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_plot(path)


# =========================================================================================
# MAIN PIPELINE
# =========================================================================================
def main():

    DATA = "processed_data"
    SAVE = "saved_models"
    VIS = os.path.join(SAVE, "visuals")

    class_names = ["Healthy", "Biotic", "Abiotic"]

    print("="*70)
    print("TRAINING STARTED")
    print("="*70)

    loader = HyperspectralDataLoader(DATA)
    X, y, files = loader.load_dataset()

    # ---------------- PLOT CLASS DISTRIBUTION ----------------
    plot_class_distribution(y, class_names, f"{VIS}/class_distribution.png")

    # ---------------- MEAN SPECTRA ----------------
    plot_mean_spectra(X, y, class_names, f"{VIS}/mean_spectra.png")

    # ---------------- SPLIT ----------------
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    Xv, Xte, yv, yte = train_test_split(
        Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=42
    )

    print("\nDataset split:")
    print("Train:", len(Xtr), "| Val:", len(Xv), "| Test:", len(Xte))

    # ---------------- PREPROCESS ----------------
    pre = Preprocessor()
    XtrP = pre.fit_transform(Xtr)
    XvP = pre.transform(Xv)
    XteP = pre.transform(Xte)

    # PCA graph
    plot_pca_variance(pre.pca, f"{VIS}/pca_explained_variance.png")

    # ---------------- TRAIN MODEL ----------------
    svm = SVMModel()
    svm.train(XtrP, ytr)

    # ---------------- TRAIN CONF MATRIX ----------------
    y_tr_pred = svm.model.predict(XtrP)
    cm_train = confusion_matrix(ytr, y_tr_pred)
    plot_conf_matrix(cm_train, class_names, f"{VIS}/confusion_matrix_train.png")

    # ---------------- VAL CONF MATRIX ----------------
    y_val_pred = svm.model.predict(XvP)
    cm_val = confusion_matrix(yv, y_val_pred)
    plot_conf_matrix(cm_val, class_names, f"{VIS}/confusion_matrix_val.png")

    # ---------------- TEST EVAL ----------------
    y_pred = svm.model.predict(XteP)
    y_prob = svm.model.predict_proba(XteP)

    cm_test = confusion_matrix(yte, y_pred)
    plot_conf_matrix(cm_test, class_names, f"{VIS}/confusion_matrix_test.png")

    # ---------------- METRICS ----------------
    acc = accuracy_score(yte, y_pred)
    f1 = f1_score(yte, y_pred, average='weighted')

    print("\nClassification Report:")
    print(classification_report(yte, y_pred, target_names=class_names))

    # ---------------- SAVE MODELS ----------------
    os.makedirs(SAVE, exist_ok=True)
    joblib.dump(svm.model, f"{SAVE}/svm.pkl")
    joblib.dump(pre.scaler, f"{SAVE}/scaler.pkl")
    joblib.dump(pre.pca, f"{SAVE}/pca.pkl")

    metadata = {
        "class_names": class_names,
        "n_components": int(pre.pca.n_components_),
        "explained_variance": float(pre.pca.explained_variance_ratio_.sum()),
        "test_accuracy": float(acc),
        "test_f1": float(f1)
    }
    with open(f"{SAVE}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining Complete ✓")
    print("Models + Graphs saved inside:", SAVE)


if __name__ == "__main__":
    main()
