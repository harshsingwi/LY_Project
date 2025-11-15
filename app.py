import streamlit as st
import os
import numpy as np
import joblib
from pathlib import Path
import spectral.io.envi as envi
from PIL import Image

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Leaf Stress Classifier",
    page_icon="🍇",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center; margin-bottom:30px;'>🍇 Grapevine Leaf Stress Classifier</h1>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------------------
MODEL_DIR = "saved_models"
RGB_DIR = "rawp_images"

@st.cache_resource
def load_all_models():
    model = joblib.load(f"{MODEL_DIR}/svm.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    pca = joblib.load(f"{MODEL_DIR}/pca.pkl")

    import json
    with open(f"{MODEL_DIR}/model_metadata.json", "r") as f:
        metadata = json.load(f)

    return model, scaler, pca, metadata

model, scaler, pca, metadata = load_all_models()
CLASS_NAMES = metadata["class_names"]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def load_cube(hdr_path):
    img_path = hdr_path.replace(".hdr", ".img")
    img = envi.open(hdr_path, img_path)
    return img.load().astype(np.float32)

def extract_median_spectrum(cube):
    return np.median(cube, axis=(0,1))

def predict_spectrum(spectrum):
    X = spectrum.reshape(1, -1)
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)
    pred = model.predict(Xp)[0]
    proba = model.predict_proba(Xp)[0]
    return pred, proba

def find_rgb(base):
    rgb_name = f"REFLECTANCE_{base}.png"
    path = Path(RGB_DIR) / rgb_name
    return path if path.exists() else None

# ------------------------------------------------------------
# SIDEBAR INPUT (DEFAULT = processed_data)
# ------------------------------------------------------------
st.sidebar.header("📁 Input Options")

mode = st.sidebar.radio(
    "Choose input:",
    ["Pick from processed_data", "Upload HDR file"],
    index=0
)

hdr_path = None

if mode == "Upload HDR file":
    uploaded = st.sidebar.file_uploader("Upload HDR", type=["hdr"])
    if uploaded:
        tmp = "uploaded_temp.hdr"
        with open(tmp, "wb") as f:
            f.write(uploaded.read())
        hdr_path = tmp

else:
    folders = ["healthy", "biotic", "abiotic"]
    sel_folder = st.sidebar.selectbox("Folder", folders)

    folder_path = Path("processed_data") / sel_folder
    hdr_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".hdr")])

    sel_hdr = st.sidebar.selectbox("HDR File", hdr_files)
    hdr_path = str(folder_path / sel_hdr)

# ------------------------------------------------------------
# MAIN VIEW
# ------------------------------------------------------------
if hdr_path:

    base = Path(hdr_path).stem

    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align:center;'>📄 Processing File: {base}</h3>",
        unsafe_allow_html=True
    )

    cube = load_cube(hdr_path)
    spectrum = extract_median_spectrum(cube)
    pred_idx, proba = predict_spectrum(spectrum)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = proba[pred_idx]

    # ------------------------------------------------------------
    # SAME UI AS BEFORE (RGB left — Prediction right)
    # ------------------------------------------------------------

    col1, col2 = st.columns([1, 1])

    # LEFT → RGB IMAGE (unchanged)
    with col1:
        st.subheader("🌈 RGB Visualization")

        rgb_path = find_rgb(base)
        if rgb_path:
            img = Image.open(rgb_path)
            st.image(img, caption=f"RGB Image ({base})", use_container_width=True)
        else:
            st.warning("No RGB image found.")

    # RIGHT → PREDICTION CARD (FIXED TEXT VISIBILITY)
    with col2:
        st.subheader("🔮 Prediction")

        # FIX: DARK BACKGROUND + WHITE TEXT (VISIBLE NOW)
        st.markdown(
            f"""
            <div style="
                padding:25px;
                border-radius:12px;
                background-color:#111;
                border:1px solid #333;
                text-align:center;
                color:white;
            ">
                <h2 style="color:white; margin-bottom:8px;">{pred_class}</h2>
                <p style="font-size:20px;">
                    Confidence: <b style="color:#4CAF50;">{confidence*100:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Class Probabilities")
        for cls, p in zip(CLASS_NAMES, proba):
            st.write(f"• **{cls}** — {p*100:.2f}%")

    st.markdown("---")
