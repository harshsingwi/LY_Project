import streamlit as st
import os
import numpy as np
import joblib
from pathlib import Path
import spectral.io.envi as envi
from PIL import Image
from scipy.signal import savgol_filter

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Leaf Stress Classifier",
    page_icon="🍇",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; margin-bottom:30px;'>🍇 Grapevine Leaf Stress Classifier</h1>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
MODEL_DIR = "saved_models"
RGB_DIR = "rawp_images"
RAW_DIR = "raw_images"
HDR_DIR = "raw_hdr_data"

# ------------------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------------------
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
def load_cube_from_envi(hdr_path: str):
    """Load reduced ENVI cube (.hdr + .img/.dat) from processed_data."""
    img_path_img = hdr_path.replace(".hdr", ".img")
    img_path_dat = hdr_path.replace(".hdr", ".dat")
    img_path = img_path_img if os.path.exists(img_path_img) else img_path_dat
    img = envi.open(hdr_path, img_path)
    return img.load().astype(np.float32)

def load_cube_from_raw(base: str):
    """
    Load original RAW + HDR and apply same 4x4 spatial reduction
    as preprocessing (in-memory, no files written).
    """
    raw_path = os.path.join(RAW_DIR, base + ".raw")
    hdr_path = os.path.join(HDR_DIR, base + ".hdr")

    if not (os.path.exists(raw_path) and os.path.exists(hdr_path)):
        raise FileNotFoundError(
            f"RAW/HDR pair not found for base '{base}'. "
            "Make sure raw_images/ and raw_hdr_data/ contain this file."
        )

    img = envi.open(hdr_path, raw_path)
    cube = img.load().astype(np.float32)

    rows, cols, bands = cube.shape
    block = 4
    new_r, new_c = rows // block, cols // block

    reduced = np.zeros((new_r, new_c, bands), dtype=np.float32)
    for i in range(new_r):
        for j in range(new_c):
            reduced[i, j] = np.mean(
                cube[i * block:(i + 1) * block,
                     j * block:(j + 1) * block, :],
                axis=(0, 1),
            )
    return reduced

def extract_median_spectrum(cube):
    """Same feature extraction as training."""
    spec = np.median(cube, axis=(0, 1))
    bands = cube.shape[2]
    window = 11 if bands >= 11 else max(5, bands // 2 * 2 + 1)
    if window >= 5 and bands >= window:
        spec = savgol_filter(spec, window_length=window, polyorder=3)
    return spec

def predict_spectrum(spectrum):
    X = spectrum.reshape(1, -1)
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)
    proba = model.predict_proba(Xp)[0]
    idx = int(np.argmax(proba))
    return idx, proba


def find_rgb(base):
    rgb_name = f"REFLECTANCE_{base}.png"
    path = Path(RGB_DIR) / rgb_name
    return path if path.exists() else None

# ------------------------------------------------------------
# SIDEBAR INPUT
# ------------------------------------------------------------
st.sidebar.header("📁 Input Options")

mode = st.sidebar.radio(
    "Choose input:",
    ["Pick from dataset (processed_data)", "Upload RAW file"],
    index=0
)

hdr_path = None
base = None
cube = None

if mode == "Upload RAW file":
    uploaded = st.sidebar.file_uploader("Upload RAW file", type=["raw"])
    if uploaded is not None:
        base = Path(uploaded.name).stem
        st.sidebar.info(
            "Converting RAW → hyperspectral cube & generating RGB preview..."
        )
        try:
            cube = load_cube_from_raw(base)
            hdr_path = f"[from_raw] {base}"  # just a label for UI
        except Exception as e:
            st.sidebar.error(str(e))

else:
    # Browse the pre-split processed_data folders
    split = st.sidebar.selectbox("Subset", ["train", "val", "test"], index=0)

    if split in ("train", "val"):
        folders = ["healthy", "biotic", "abiotic"]
        sel_folder = st.sidebar.selectbox("Class folder", folders)
        folder_path = Path("processed_data") / split / sel_folder
    else:
        folder_path = Path("processed_data") / "test"  # unlabeled

    if folder_path.exists():
        hdr_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".hdr")])
    else:
        hdr_files = []

    if len(hdr_files) == 0:
        st.sidebar.warning("No .hdr files found in selected folder.")
    else:
        sel_hdr = st.sidebar.selectbox("HDR File", hdr_files)
        hdr_path = str(folder_path / sel_hdr)
        base = Path(hdr_path).stem

# ------------------------------------------------------------
# MAIN VIEW
# ------------------------------------------------------------
if hdr_path and (cube is not None or (not hdr_path.startswith("[from_raw]") and os.path.exists(hdr_path))):

    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align:center;'>📄 Processing File: {base}</h3>",
        unsafe_allow_html=True
    )

    if cube is None:
        cube = load_cube_from_envi(hdr_path)

    spectrum = extract_median_spectrum(cube)
    pred_idx, proba = predict_spectrum(spectrum)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = proba[pred_idx]

    col1, col2 = st.columns([1, 1])

    # LEFT → RGB IMAGE
    with col1:
        st.subheader("🌈 RGB Visualization")

        rgb_path = find_rgb(base)
        if rgb_path:
            img = Image.open(rgb_path)
            st.image(img, caption=f"RGB Image ({base})", use_container_width=True)
        else:
            st.warning("No RGB image found (PNG).")

    # RIGHT → PREDICTION CARD
    with col2:
        st.subheader("🔮 Prediction")

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
