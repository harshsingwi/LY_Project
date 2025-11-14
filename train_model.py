"""
Grapevine Leaf Stress Classification using Hyperspectral Imaging
Complete ML Pipeline with SVM Classifier (Fixed + Improved)
"""

import os
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm

# Hyperspectral
import spectral.io.envi as envi

# ML
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# -------------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------------

class HyperspectralDataLoader:
    """Loads ENVI images (.hdr + .img) and extracts mean spectrum."""
    
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.class_mapping = {
            'healthy': 0,
            'biotic': 1,
            'abiotic': 2,
        }
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
    
    def load_hyperspectral_image(self, hdr_path):
        """Load ENVI .hdr + .img file pair correctly."""
        try:
            img_path = hdr_path.replace(".hdr", ".img")
            img = envi.open(hdr_path, img_path)
            data = img.load()
            return data
        except Exception as e:
            print(f"[ERROR] Cannot load image: {hdr_path} | {e}")
            return None
    
    def extract_mean_spectrum(self, cube):
        """Mean reflectance spectrum across all spatial pixels."""
        return np.mean(cube, axis=(0, 1))  # (bands,)
    
    def load_dataset(self):
        X, y, filenames = [], [], []
        
        print("Loading dataset from:", self.data_folder)
        
        for class_name, class_label in self.class_mapping.items():
            class_path = self.data_folder / class_name
            
            if not class_path.exists():
                print(f"[WARNING] Skipping missing folder: {class_name}")
                continue

            hdr_files = sorted([f for f in os.listdir(class_path) if f.endswith(".hdr")])
            print(f"→ {class_name}: {len(hdr_files)} files")

            for hdr in tqdm(hdr_files, desc=f"{class_name}"):
                cube = self.load_hyperspectral_image(str(class_path / hdr))
                if cube is None:
                    continue

                spectrum = self.extract_mean_spectrum(cube)
                X.append(spectrum)
                y.append(class_label)
                filenames.append(hdr)
        
        X, y = np.array(X), np.array(y)
        print("\nDataset Loaded:")
        print(" Shape:", X.shape)
        print(" Class distribution:", np.bincount(y))
        
        return X, y, filenames

# -------------------------------------------------------------------
# PREPROCESSOR
# -------------------------------------------------------------------

class HyperspectralPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
    
    def fit_transform(self, X, variance=0.95):
        print("\nPreprocessing:")
        print("→ Standardizing features...")
        Xs = self.scaler.fit_transform(X)

        print(f"→ PCA (retain {variance*100:.1f}% variance)...")
        self.pca = PCA(n_components=variance, random_state=42)
        Xp = self.pca.fit_transform(Xs)

        print(f"   PCA: {X.shape[1]} → {Xp.shape[1]} dims")
        print(f"   Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        return Xp
    
    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

# -------------------------------------------------------------------
# CLASSIFIER
# -------------------------------------------------------------------

class SVMClassifier:
    def __init__(self):
        self.model = None
    
    def train(self, Xtr, ytr, Xval, yval, tune=True):
        print("\nTraining SVM Classifier...")

        if tune:
            params = {
                'C': [1, 10, 50, 100],
                'gamma': ['scale', 0.1, 0.01],
                'kernel': ['rbf']
            }

            base = SVC(probability=True, class_weight='balanced')
            gs = GridSearchCV(
                base, params,
                cv=StratifiedKFold(n_splits=3, shuffle=True),
                scoring='f1_weighted',
                n_jobs=-1
            )

            gs.fit(Xtr, ytr)
            self.model = gs.best_estimator_
            print("→ Best params:", gs.best_params_)
        else:
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                class_weight='balanced'
            )
            self.model.fit(Xtr, ytr)

        # Validation accuracy
        preds = self.model.predict(Xval)
        acc = accuracy_score(yval, preds)
        print(f"→ Validation Accuracy: {acc:.4f}")

# -------------------------------------------------------------------
# EVALUATOR
# -------------------------------------------------------------------

class ModelEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, y_true, y_pred, y_proba, name="Test"):
        print(f"\n===== {name} Evaluation =====")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7,5))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"cm_{name.lower()}.png")
        plt.close()

        # ROC-AUC (OVR)
        try:
            aucs = []
            for i in range(len(self.class_names)):
                yt = (y_true == i).astype(int)
                ys = y_proba[:, i]
                aucs.append(roc_auc_score(yt, ys))

            print("\nROC-AUC:")
            for cname, auc in zip(self.class_names, aucs):
                print(f"  {cname}: {auc:.3f}")
        except:
            print("[WARN] Cannot compute ROC AUC.")

        return acc, prec, rec, f1

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def main():

    print("\n===== TRAINING PIPELINE START =====")

    DATA = "processed_data"
    SAVE = "saved_models"
    os.makedirs(SAVE, exist_ok=True)

    CLASS_NAMES = ["Healthy", "Biotic", "Abiotic"]

    # Load data
    loader = HyperspectralDataLoader(DATA)
    X, y, fns = loader.load_dataset()

    # Split
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    Xval, Xte, yval, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42
    )

    # Preprocess
    prep = HyperspectralPreprocessor()
    Xtr_pca = prep.fit_transform(Xtr)
    Xval_pca = prep.transform(Xval)
    Xte_pca = prep.transform(Xte)

    # Train SVM
    svm = SVMClassifier()
    svm.train(Xtr_pca, ytr, Xval_pca, yval, tune=True)

    # Evaluate
    eval = ModelEvaluator(CLASS_NAMES)

    print("\nEVALUATING TRAIN SET:")
    ytr_pred = svm.model.predict(Xtr_pca)
    ytr_prob = svm.model.predict_proba(Xtr_pca)
    eval.evaluate(ytr, ytr_pred, ytr_prob, "Train")

    print("\nEVALUATING TEST SET:")
    yte_pred = svm.model.predict(Xte_pca)
    yte_prob = svm.model.predict_proba(Xte_pca)
    test_acc, test_prec, test_rec, test_f1 = eval.evaluate(yte, yte_pred, yte_prob, "Test")

    # Save models
    joblib.dump(svm.model, f"{SAVE}/svm.pkl")
    joblib.dump(prep.scaler, f"{SAVE}/scaler.pkl")
    joblib.dump(prep.pca, f"{SAVE}/pca.pkl")

    print("\nSaved:")
    print("  svm.pkl")
    print("  scaler.pkl")
    print("  pca.pkl")

    print("\n===== TRAINING COMPLETE =====")


if __name__ == "__main__":
    main()
