"""
=============================================================================
  Pneumonia Detection — Interactive Streamlit Web Dashboard
=============================================================================

  Launch:
      streamlit run app.py

  Features:
      • Upload a chest X-ray (.jpg / .jpeg / .png)
      • One-click AI analysis with confidence scoring
      • Side-by-side Grad-CAM explainability visualization
      • AI-generated preliminary radiology report (via local Ollama LLM)
      • GPU-aware model caching — loads once, persists across interactions
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION — must execute before any TF operation
# ═══════════════════════════════════════════════════════════════════════════════
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from report_generator import generate_report

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "pneumonia_model_best.keras")
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Header ─────────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 50%, #2a6496 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a0c4e8;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Result cards ───────────────────────────────────────────────────── */
    .result-card {
        background: #f8f9fa;
        border-left: 5px solid;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .result-normal  { border-color: #28a745; }
    .result-pneumonia { border-color: #dc3545; }

    /* ── Disclaimer ─────────────────────────────────────────────────────── */
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 1rem;
    }

    /* ── Hide Streamlit default footer ──────────────────────────────────── */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached — loaded once, persists across reruns)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model into GPU memory...")
def load_model():
    """Load the trained Keras model. Cached so it persists across sessions."""
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: `{MODEL_PATH}`")
        st.stop()
    return keras.models.load_model(MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def find_base_model(model):
    """Locate the MobileNetV2 sub-model inside the loaded model."""
    for layer in model.layers:
        if hasattr(layer, "layers") and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("MobileNetV2 sub-model not found in model layers.")


def predict_image(model, img_array):
    """
    Run inference on a single image.

    Args:
        model     : Loaded Keras model
        img_array : np.ndarray, (224, 224, 3), [0..255]

    Returns:
        pred_class : str
        confidence : float [0..1]
    """
    img_batch = np.expand_dims(img_array, axis=0)
    raw_prob = float(
        model(tf.cast(img_batch, tf.float32), training=False).numpy()[0, 0]
    )

    if raw_prob >= 0.5:
        return "PNEUMONIA", raw_prob
    else:
        return "NORMAL", 1.0 - raw_prob


def make_gradcam_heatmap(img_array, model):
    """
    Generate a Grad-CAM heatmap for explainability.

    Args:
        img_array : np.ndarray, (224, 224, 3), [0..255]
        model     : Loaded Keras model

    Returns:
        heatmap : np.ndarray, (7, 7), [0..1]
    """
    base_model    = find_base_model(model)
    gap_layer     = model.get_layer("global_avg_pool")
    dropout_layer = model.get_layer("dropout")
    dense_layer   = model.get_layer("prediction")

    img_batch    = np.expand_dims(img_array, axis=0)
    img_tensor   = tf.cast(img_batch, tf.float32)
    preprocessed = preprocess_input(img_tensor)

    with tf.GradientTape() as tape:
        conv_output = base_model(preprocessed, training=False)
        tape.watch(conv_output)
        x = gap_layer(conv_output)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)
        pred_score = preds[0, 0]

    grads = tape.gradient(pred_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """Blend Grad-CAM heatmap onto the original image."""
    heatmap_resized = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],
        (img.shape[0], img.shape[1]),
    ).numpy().squeeze()

    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    img_norm = img.astype(np.float32) / 255.0

    superimposed = heatmap_colored * alpha + img_norm * (1.0 - alpha)
    return np.clip(superimposed, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ System Info")
    st.markdown(f"**Model:** MobileNetV2 (Fine-Tuned)")
    st.markdown(f"**Input Size:** {IMG_SIZE[0]}×{IMG_SIZE[1]}px")
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")

    gpu_list = tf.config.list_physical_devices("GPU")
    if gpu_list:
        st.success(f"GPU: {gpu_list[0].name}")
    else:
        st.warning("Running on CPU")

    st.divider()
    st.markdown("## 📋 How to Use")
    st.markdown(
        "1. Upload a chest X-ray image\n"
        "2. Click **Analyze Scan**\n"
        "3. Review the prediction, confidence, and Grad-CAM\n"
        "4. Optionally generate an AI radiology report"
    )

    st.divider()
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <b>Medical Disclaimer:</b> This system is a research '
        'prototype and must NOT be used for actual clinical diagnosis. '
        'All findings must be reviewed by a licensed radiologist.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>🫁 Pneumonia Detection AI</h1>'
    '<p>Deep Learning-Powered Chest X-Ray Analysis with Explainable AI</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Load model (cached) ─────────────────────────────────────────────────────
model = load_model()

# ── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Best results with posterior-anterior (PA) chest radiographs.",
)

if uploaded_file is not None:
    # Load and prepare image
    image_pil = Image.open(uploaded_file).convert("RGB")
    img_resized = image_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)

    # Reset results if a new file is uploaded
    if st.session_state.get("_last_filename") != uploaded_file.name:
        st.session_state.pop("results", None)
        st.session_state["_last_filename"] = uploaded_file.name

    # ── Image display columns ────────────────────────────────────────────
    col_original, col_gradcam = st.columns(2)
    with col_original:
        st.image(image_pil, caption="📷 Uploaded X-Ray", use_container_width=True)

    # ── Analyze button ───────────────────────────────────────────────────
    if st.button("🔍 Analyze Scan", type="primary", use_container_width=True):
        with st.spinner("🧠 Running neural network inference..."):
            pred_class, confidence = predict_image(model, img_array)
            heatmap = make_gradcam_heatmap(img_array, model)
            overlay = overlay_gradcam(img_array, heatmap)

        st.session_state["results"] = {
            "pred_class": pred_class,
            "confidence": confidence,
            "overlay": overlay,
        }

    # ── Display results ──────────────────────────────────────────────────
    if "results" in st.session_state:
        r = st.session_state["results"]
        pred_class = r["pred_class"]
        confidence = r["confidence"]
        overlay    = r["overlay"]

        # Grad-CAM image
        with col_gradcam:
            st.image(overlay, caption="🔥 Grad-CAM Attention Map", use_container_width=True)

        st.divider()

        # ── Metrics row ──────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)

        with m1:
            if pred_class == "PNEUMONIA":
                st.error(f"### 🔴 {pred_class}")
            else:
                st.success(f"### 🟢 {pred_class}")

        with m2:
            st.metric("Confidence Score", f"{confidence:.1%}")

        with m3:
            risk = "HIGH" if pred_class == "PNEUMONIA" and confidence > 0.8 else \
                   "MODERATE" if pred_class == "PNEUMONIA" else "LOW"
            st.metric("Risk Level", risk)

        # ── Confidence progress bar ──────────────────────────────────────
        st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

        # ── AI Report Generation ─────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 AI-Generated Preliminary Report")
        st.caption("Powered by Qwen 2.5 via local Ollama instance")

        if st.button("📝 Generate Radiology Report", use_container_width=True):
            with st.spinner("✍️ Drafting report via local LLM..."):
                report = generate_report(pred_class, confidence)
            st.session_state["report"] = report

        if "report" in st.session_state:
            st.markdown(st.session_state["report"])

else:
    # ── Empty state ──────────────────────────────────────────────────────
    st.info("👆 Upload a chest X-ray image to begin analysis.")

    st.markdown("#### 📌 Quick Start")
    st.code(
        "# Test with a sample from the dataset\n"
        "# Upload any image from: dataset/test/PNEUMONIA/ or dataset/test/NORMAL/",
        language="bash",
    )
