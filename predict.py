"""
=============================================================================
  Pneumonia Detection CLI — Single-Image Inference with Grad-CAM
=============================================================================

  Usage:
      python predict.py --image path/to/chest_xray.jpg
      python predict.py --image path/to/chest_xray.jpg --model pneumonia_model.keras

  Outputs saved to: ./outputs/
      • prediction result printed to console
      • Grad-CAM overlay saved as outputs/<filename>_gradcam.png
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CONFIGURATION — must execute BEFORE any TensorFlow operation
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
import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "pneumonia_model_best.keras")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(image_path: str) -> tuple:
    """
    Load an image from disk and prepare it for inference.

    Returns:
        img_display : np.ndarray (224, 224, 3) in [0..255] for visualization
        img_batch   : np.ndarray (1, 224, 224, 3) in [0..255] for model input
    """
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)        # (224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)      # (1, 224, 224, 3)

    return img_array, img_batch


def find_base_model(model: keras.Model) -> keras.Model:
    """
    Locate the MobileNetV2 sub-model inside the loaded Keras model.

    When a model saved via the Functional API contains a nested Model
    (e.g., MobileNetV2 used as a layer), it appears as a layer whose
    type is also keras.Model.
    """
    for layer in model.layers:
        if hasattr(layer, "layers") and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError(
        "Could not find a MobileNetV2 sub-model in the loaded model. "
        "Ensure the model was built with MobileNetV2 as a named layer."
    )


def run_inference(model: keras.Model, img_batch: np.ndarray) -> tuple:
    """
    Run a single forward pass and return the predicted class and confidence.

    Returns:
        pred_class  : str — "NORMAL" or "PNEUMONIA"
        confidence  : float — model confidence in the predicted class
        raw_prob    : float — raw sigmoid output (P(PNEUMONIA))
    """
    raw_prob = float(model(tf.cast(img_batch, tf.float32), training=False).numpy()[0, 0])

    if raw_prob >= 0.5:
        pred_class = "PNEUMONIA"
        confidence = raw_prob
    else:
        pred_class = "NORMAL"
        confidence = 1.0 - raw_prob

    return pred_class, confidence, raw_prob


def make_gradcam_heatmap(img_batch: np.ndarray, model: keras.Model) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given image batch.

    This function manually traces through the model's components:
      1. preprocess_input → MobileNetV2 backbone (last conv features)
      2. GradientTape watches the conv output
      3. Classifier head (GAP → Dropout → Dense) produces prediction
      4. Gradient of prediction w.r.t. conv features → heatmap

    Args:
        img_batch : np.ndarray, shape (1, 224, 224, 3), values [0..255]
        model     : The loaded Keras model

    Returns:
        heatmap : np.ndarray, shape (7, 7), values [0..1]
    """
    base_model    = find_base_model(model)
    gap_layer     = model.get_layer("global_avg_pool")
    dropout_layer = model.get_layer("dropout")
    dense_layer   = model.get_layer("prediction")

    img_tensor   = tf.cast(img_batch, tf.float32)
    preprocessed = preprocess_input(img_tensor)

    with tf.GradientTape() as tape:
        conv_output = base_model(preprocessed, training=False)      # (1, 7, 7, 1280)
        tape.watch(conv_output)

        x = gap_layer(conv_output)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)
        pred_score = preds[0, 0]

    grads = tape.gradient(pred_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))            # (1280,)

    conv_output = conv_output[0]                                     # (7, 7, 1280)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]            # (7, 7, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        img     : (H, W, 3) in [0..255]
        heatmap : (h, w) in [0..1]
        alpha   : heatmap opacity

    Returns:
        superimposed : (H, W, 3) in [0..1]
    """
    heatmap_resized = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],
        (img.shape[0], img.shape[1]),
    ).numpy().squeeze()

    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]         # (H, W, 3)
    img_norm = img.astype(np.float32) / 255.0

    superimposed = heatmap_colored * alpha + img_norm * (1.0 - alpha)
    return np.clip(superimposed, 0, 1)


def save_gradcam_figure(img: np.ndarray, overlay: np.ndarray,
                        pred_class: str, confidence: float,
                        save_path: str):
    """Save a side-by-side figure: original image | Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img.astype(np.uint8))
    axes[0].set_title("Original X-Ray", fontsize=12)
    axes[0].axis("off")

    color = "red" if pred_class == "PNEUMONIA" else "green"
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM — {pred_class} ({confidence:.1%})",
                      fontsize=12, color=color, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection — Single Image CLI Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predict.py --image dataset/test/PNEUMONIA/person1_virus_6.jpeg",
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to a chest X-ray image (.jpg, .jpeg, .png)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=DEFAULT_MODEL,
        help=f"Path to the trained .keras model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loading model: {args.model}")
    model = keras.models.load_model(args.model)

    # ── Preprocess image ─────────────────────────────────────────────────────
    print(f"[INFO] Processing image: {args.image}")
    img_display, img_batch = load_and_preprocess_image(args.image)

    # ── Run inference ────────────────────────────────────────────────────────
    pred_class, confidence, raw_prob = run_inference(model, img_batch)

    print("\n" + "=" * 50)
    print(f"  PREDICTION  : {pred_class}")
    print(f"  CONFIDENCE  : {confidence:.2%}")
    print(f"  RAW P(PNEU) : {raw_prob:.4f}")
    print("=" * 50)

    # ── Generate Grad-CAM ────────────────────────────────────────────────────
    print("\n[INFO] Generating Grad-CAM heatmap...")
    heatmap = make_gradcam_heatmap(img_batch, model)
    overlay = overlay_gradcam(img_display, heatmap)

    # ── Save output ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam.png")

    save_gradcam_figure(img_display, overlay, pred_class, confidence, save_path)
    print(f"[INFO] Grad-CAM saved → {save_path}")

    # ── Free resources ───────────────────────────────────────────────────────
    del model
    keras.backend.clear_session()
    print("[INFO] Done. Resources released.")


if __name__ == "__main__":
    main()
