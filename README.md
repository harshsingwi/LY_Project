# 📘 Grapevine Leaf Stress Classification using Hyperspectral Imaging

A complete end-to-end machine learning pipeline to classify grapevine leaf stress using **hyperspectral imaging (400–1000 nm, 204 bands)**.  
The system detects:

- **Healthy leaves**
- **Biotically stressed leaves** (disease, infection)
- **Abiotically stressed leaves** (water, nutrient, heat, etc.)

This project includes RAW → ENVI preprocessing, PCA, SVM classification, evaluation, and prediction for new hyperspectral images.

---

# 📑 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [📁 Dataset Structure](#-dataset-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage Guide](#-usage-guide)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#2-training)
  - [3. Prediction](#3-prediction)
- [🔧 Pipeline Details](#-pipeline-details)
- [📊 Model Performance](#-model-performance)
- [📂 Files Description](#-files-description)
- [🛠 Troubleshooting](#-troubleshooting)
- [📚 Advanced Usage](#-advanced-usage)
- [🎓 Academic Use](#-academic-use)
- [📄 License](#-license)

---

# 🔍 Project Overview

This system performs the full pipeline:

### **1. Preprocessing**
- Reads `.raw` + `.hdr` files  
- Reduces spatial resolution via 4×4 block averaging  
- Converts to ENVI `.hdr` + `.img`  
- Auto-labels using `description-2.csv`  
- Organizes into `healthy/`, `biotic/`, `abiotic/`, `unknown/`

### **2. Feature Extraction**
- Extracts **mean spectral signature** across all pixels (1 × 204 vector)

### **3. Preprocessing**
- StandardScaler (Z-score normalization)
- PCA (retain 95% variance)

### **4. Classification**
- **SVM (RBF kernel)**
- Hyperparameter tuning via GridSearchCV
- Balanced classes + probability prediction

### **5. Prediction**
- Predict single `.hdr`
- Or entire folder (`--batch`)

---

# 📁 Dataset Structure

```
project/
├── raw_images/               # Original large .raw files   (gitignored)
├── raw_hdr_data/             # Original .hdr files         (gitignored)
├── description-2.csv         # Main labeling file
│
├── processed_data/           # Auto-generated
│   ├── healthy/
│   ├── biotic/
│   ├── abiotic/
│   ├── unknown/
│   └── labels.json
│
├── saved_models/
│   ├── svm.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│
├── hyperspectral_preprocessing.py
├── train_model.py
└── predict.py
```

---

# ⚙️ Installation

### **1. Create virtual environment (recommended)**

```
python -m venv venv
```

Activate:

- Windows:
  ```
  venv\Scripts\activate
  ```
- Mac/Linux:
  ```
  source venv/bin/activate
  ```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

---

# 🚀 Usage Guide

# 1️⃣ Preprocessing

Convert RAW → reduced ENVI → auto-label → organize:

```
python hyperspectral_preprocessing.py
```

This will generate:

```
processed_data/
    healthy/
    biotic/
    abiotic/
    unknown/
    labels.json
```

**Unknown folder** = files missing in CSV → safe to use later for testing/predicting.

---

# 2️⃣ Training

Train the PCA + SVM classification pipeline:

```
python train_model.py
```

This will save:

```
saved_models/
    svm.pkl
    scaler.pkl
    pca.pkl
```

---

# 3️⃣ Prediction

### **A) Predict a single image:**

```
python predict.py path/to/image.hdr
```

Example:
```
python predict.py test_images/2020-09-10_012.hdr
```

### **B) Predict an entire folder:**

```
python predict.py path/to/folder --batch
```

Generates:

```
path/to/folder/predictions.json
```

---

# 🔧 Pipeline Details

### **Preprocessing**
- 4×4 block averaging  
- ENVI BSQ format  
- Mean spectrum extraction  

### **PCA**
- Retains 95% variance  
- Reduces 204 → ~40 components  

### **SVM**
- RBF kernel  
- Class weight: balanced  
- Hyperparameter tuning:  
  ```
  C = [1, 10, 50, 100]
  gamma = ["scale", 0.1, 0.01]
  ```

### **Evaluation**
- Accuracy, Precision, Recall, F1  
- Classification report  
- Confusion matrix (PNG)  
- ROC–AUC (macro + per class)

---

# 📊 Model Performance

Typical expected performance:

| Metric | Expected |
|--------|----------|
| Accuracy | 88–95% |
| Precision | 87–95% |
| Recall | 85–95% |
| F1-Score | 86–94% |
| Macro ROC-AUC | 0.90–0.98 |

---

# 📂 Files Description

| File | Description |
|------|-------------|
| `hyperspectral_preprocessing.py` | RAW → ENVI, auto-label, organize folders |
| `train_model.py` | Full ML pipeline: load → preprocess → train → evaluate |
| `predict.py` | Predict for single or batch `.hdr` files |
| `processed_data/` | Final dataset used for model training |
| `saved_models/` | Trained SVM, PCA, and Scaler |

---

# 🛠 Troubleshooting

### **Prediction error: ENVI cannot open file**
Ensure both files exist:
```
image.hdr
image.img
```

### **Model loads but prediction is wrong**
- Wrong folder structure  
- Missing `.img` file  
- Not using reduced ENVI files from preprocessing  

### **Low accuracy**
Try:
- Better CSV labeling  
- More training samples  
- PCA variance = 0.99  
- Larger SVM grid

---

# 📚 Advanced Usage

### Tune SVM:
```
'C': [1, 10, 50, 100, 500]
'gamma': ['scale', 0.1, 0.01, 0.001]
```

### Change PCA:
```
variance = 0.99
```

### Adjust train/test split:
```
test_size=0.2
```

---

# 🎓 Academic Use

When using this for research, include:

- Preprocessing details (4×4 block average)  
- PCA variance retained  
- SVM hyperparameters  
- Train/val/test splits  
- Confusion matrix & ROC curves  

---

# 📄 License

This project is provided for **research and educational purposes only**.

---

**🍇 Happy Spectral Classification!**
