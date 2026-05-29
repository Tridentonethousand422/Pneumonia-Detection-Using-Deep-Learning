"""
=============================================================================
  Pneumonia Detection from Chest X-Ray Images — Advanced Pipeline
  Transfer Learning with MobileNetV2 + Fine-Tuning
=============================================================================

  ADVANCED FEATURES IMPLEMENTED
  ─────────────────────────────
  1. TWO-PHASE TRAINING WITH GRADUAL UNFREEZING (Fine-Tuning)
     The base model trains in two phases: first with a fully frozen
     MobileNetV2 to train the classification head, then with the top
     ~55 layers of the backbone unfrozen at a 10x-lower learning rate.
     This avoids catastrophic forgetting while adapting high-level
     ImageNet features to chest X-ray pathology textures.

  2. CLASS IMBALANCE HANDLING VIA COMPUTED CLASS WEIGHTS
     The Kaggle Chest X-Ray dataset is severely imbalanced (~3:1
     PNEUMONIA to NORMAL). Without correction the model learns a
     bias toward the majority class, producing dangerously high
     False Negative rates (missed pneumonia). We compute inverse-
     frequency class weights and pass them to model.fit().

  3. GRAD-CAM EXPLAINABILITY HEATMAPS
     Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes
     which spatial regions of the X-ray most influence each prediction.
     This is essential in medical AI for clinical trust, debugging
     false predictions, and verifying the model attends to lung
     parenchyma — not spurious artifacts like text labels or borders.

  4. TEST-TIME AUGMENTATION (TTA) FOR ROBUST INFERENCE
     Instead of a single deterministic forward pass at test time, TTA
     applies N stochastic augmentations (horizontal flip, rotation,
     zoom) per image and averages the predicted probabilities. This
     reduces spatial-sensitivity variance and typically boosts accuracy
     by 1-3% — meaningful in a clinical diagnostic context.

  Hardware : NVIDIA RTX 5050 Mobile (8 GB VRAM) via WSL2
  Dataset  : Chest X-Ray Images (Pneumonia) — Kaggle
  Task     : Binary Classification (NORMAL vs PNEUMONIA)
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0. GPU CONFIGURATION — must execute BEFORE any other TensorFlow operation
# ═══════════════════════════════════════════════════════════════════════════════
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Memory growth enabled on {len(gpus)} GPU(s): {gpus}")
    except RuntimeError as e:
        print(f"[WARNING] GPU memory growth could not be set: {e}")
else:
    print("[INFO] No GPU detected — training will run on CPU.")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION & HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
TRAIN_DIR       = os.path.join(BASE_DIR, "train")
VAL_DIR         = os.path.join(BASE_DIR, "val")
TEST_DIR        = os.path.join(BASE_DIR, "test")

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
PHASE1_EPOCHS   = 10           # Feature extraction (frozen backbone)
PHASE2_EPOCHS   = 10           # Fine-tuning (top layers unfrozen)
PHASE1_LR       = 1e-4         # Standard Adam LR for randomly-initialized head
PHASE2_LR       = 1e-5         # 10x lower for fine-tuning stability
FINE_TUNE_FROM  = 100          # Unfreeze MobileNetV2 layers from this index onward
TTA_ROUNDS      = 10           # Number of augmented forward passes for TTA
AUTOTUNE        = tf.data.AUTOTUNE
SEED            = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_datasets(train_dir, val_dir, test_dir, img_size, batch_size):
    """
    Load train/val/test splits using tf.keras.utils.image_dataset_from_directory.

    Returns:
        train_ds, val_ds, test_ds  — tf.data.Dataset objects
        class_names                — e.g. ["NORMAL", "PNEUMONIA"]
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size,
        label_mode="binary", shuffle=True, seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size,
        label_mode="binary", shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size,
        label_mode="binary", shuffle=False,
    )

    class_names = train_ds.class_names

    # Prefetch only — .cache() is intentionally omitted to prevent
    # WSL2 system RAM exhaustion on datasets with thousands of images.
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLASS WEIGHT COMPUTATION  [ADVANCED FEATURE #2]
# ═══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(train_dir):
    """
    Compute class weights inversely proportional to class frequencies.

    The Kaggle Chest X-Ray dataset is heavily imbalanced:
      ~3,875 PNEUMONIA  vs  ~1,341 NORMAL  in the training set.

    Without compensation, the model biases toward predicting PNEUMONIA
    for every input, producing dangerously high False Negative rates
    (missed pneumonia cases flagged as healthy).

    Formula:  weight_i = total_samples / (n_classes × count_i)

    This is mathematically equivalent to sklearn.utils.class_weight
    .compute_class_weight("balanced", ...) but computed directly from
    the filesystem to avoid an extra full-dataset iteration.

    Returns:
        class_weights — dict {0: w_NORMAL, 1: w_PNEUMONIA}
        class_counts  — dict {"NORMAL": n, "PNEUMONIA": n}
    """
    train_path = pathlib.Path(train_dir)
    class_counts = {}

    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith((".", "_")):
            count = sum(1 for f in class_dir.iterdir() if f.is_file())
            class_counts[class_dir.name] = count

    total = sum(class_counts.values())
    n_classes = len(class_counts)

    # Alphabetical enumeration matches image_dataset_from_directory's class order
    class_weights = {}
    for i, (name, count) in enumerate(sorted(class_counts.items())):
        class_weights[i] = round(total / (n_classes * count), 4)

    return class_weights, class_counts


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
def build_augmentation_layer():
    """
    Build a Keras Sequential augmentation layer.

    These layers are embedded inside the model graph. They are only
    active when training=True (model.fit), and act as a transparent
    pass-through during inference (model.predict / model.evaluate).
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(img_size, augmentation_layer):
    """
    Build the PneumoniaDetector model with a MobileNetV2 backbone.

    Architecture:
      Input (224×224×3)
        → Data Augmentation  (active only during training)
        → MobileNetV2 preprocess_input  (scales pixels → [-1, 1])
        → MobileNetV2 backbone  (frozen in Phase 1)
        → GlobalAveragePooling2D
        → Dropout(0.2)
        → Dense(1, sigmoid)

    Returns:
        model      — the complete Keras functional model
        base_model — the MobileNetV2 sub-model (needed for fine-tuning
                     and Grad-CAM layer access)
    """
    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False     # Fully frozen for Phase 1

    inputs = keras.Input(shape=(*img_size, 3), name="input_image")
    x = augmentation_layer(inputs)
    x = preprocess_input(x)          # MobileNetV2-specific [-1, 1] scaling
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="PneumoniaDetector_MobileNetV2")
    return model, base_model


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════
def build_callbacks(checkpoint_path):
    """Build standard training callbacks for each training phase."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2,
            patience=3, min_lr=1e-7, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COMPILE HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def compile_model(model, learning_rate):
    """Compile (or recompile) the model with specified learning rate."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GRAD-CAM IMPLEMENTATION  [ADVANCED FEATURE #3]
# ═══════════════════════════════════════════════════════════════════════════════
def make_gradcam_heatmap(img_array, model, base_model):
    """
    Generate a Grad-CAM heatmap for a single image.

    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
    the spatial regions of the input X-ray that most influenced the
    model's prediction.  In a medical context this is crucial for:
      • Clinical trust — radiologists can verify the model's reasoning
      • Error analysis — reveals if the model attends to artifacts
                         (text labels, borders, external devices)
      • Regulatory compliance — explainability is increasingly required
                                by FDA/CE for AI-aided diagnostics

    How it works:
      1. Forward-pass the preprocessed image through the frozen base
         model to obtain the last convolutional feature map (7×7×1280).
      2. Forward-pass through the classifier head (GAP → Dropout → Dense)
         while recording gradients with tf.GradientTape.
      3. Compute the gradient of the predicted class score with respect
         to each spatial location of the feature map.
      4. Global-average-pool the gradients to obtain per-channel
         importance weights (1280 scalars).
      5. Weighted-sum the feature map channels → (7×7) heatmap.
      6. Apply ReLU (keep only positive influence) and normalize.

    Args:
        img_array  : Raw image tensor, shape (1, 224, 224, 3), [0..255]
        model      : The trained Keras model (for classifier head weights)
        base_model : The MobileNetV2 sub-model (for feature extraction)

    Returns:
        heatmap — np.ndarray, shape (7, 7), values ∈ [0, 1]
    """
    # Preprocess (skip augmentation, apply MobileNetV2 scaling only)
    img_tensor = tf.cast(img_array, tf.float32)
    preprocessed = preprocess_input(img_tensor)

    # Extract trained classifier head layers (weights shared with full model)
    gap_layer     = model.get_layer("global_avg_pool")
    dropout_layer = model.get_layer("dropout")
    dense_layer   = model.get_layer("prediction")

    with tf.GradientTape() as tape:
        # Forward through the backbone
        conv_output = base_model(preprocessed, training=False)  # (1, 7, 7, 1280)
        tape.watch(conv_output)

        # Forward through the classification head (same trained weights)
        x = gap_layer(conv_output)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)
        pred_score = preds[0, 0]

    # Gradient of the prediction w.r.t. each spatial cell of the feature map
    grads = tape.gradient(pred_score, conv_output)

    # Per-channel importance weights via global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))       # (1280,)

    # Weighted combination of feature channels
    conv_output = conv_output[0]                                # (7, 7, 1280)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]       # (7, 7, 1)
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    """
    Blend a Grad-CAM heatmap onto the original X-ray image.

    Uses the "jet" colormap to produce a red-hot highlight on
    high-activation regions and cool blues on low-activation regions.

    Args:
        img     : Original image, shape (H, W, 3), values [0..255]
        heatmap : Grad-CAM output, shape (h, w), values [0..1]
        alpha   : Heatmap opacity (0 = invisible, 1 = fully opaque)

    Returns:
        superimposed — np.ndarray, shape (H, W, 3), values [0..1]
    """
    # Resize heatmap (7×7) to full image resolution
    heatmap_resized = tf.image.resize(
        heatmap[tf.newaxis, ..., tf.newaxis],       # (1, 7, 7, 1)
        (img.shape[0], img.shape[1]),
    ).numpy().squeeze()

    # Apply jet colormap → (H, W, 4), take RGB channels only
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]

    # Normalize image to [0, 1]
    img_norm = img.astype(np.float32) / 255.0

    # Alpha blend
    superimposed = heatmap_colored * alpha + img_norm * (1 - alpha)
    return np.clip(superimposed, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TEST-TIME AUGMENTATION  [ADVANCED FEATURE #4]
# ═══════════════════════════════════════════════════════════════════════════════
def predict_with_tta(model, dataset, n_rounds=10):
    """
    Perform Test-Time Augmentation (TTA) for more robust inference.

    Standard inference uses a single deterministic forward pass.
    TTA instead applies N stochastic augmentations to each test image
    and averages the predicted probabilities.  The augmentation layers
    (RandomFlip, RandomRotation, RandomZoom) are called with
    training=True to enable their stochastic behavior.

    Because the model's internal augmentation layer is a no-op at
    inference time (training=False), we apply a separate standalone
    augmentation pipeline BEFORE feeding images to the model.

    In medical imaging, even a 1% accuracy improvement from TTA
    can translate to meaningful reductions in diagnostic errors.

    Args:
        model    : Trained Keras model
        dataset  : tf.data.Dataset (should have shuffle=False)
        n_rounds : Total forward passes per image (including 1 original)

    Returns:
        y_prob — np.ndarray of averaged probabilities, shape (N,)
    """
    tta_augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="tta_augmentation")

    all_preds = []

    for images, _ in dataset:
        batch_preds = []

        # 1. Original (unaugmented) prediction
        batch_preds.append(model(images, training=False).numpy())

        # 2. Augmented predictions (n_rounds - 1 additional passes)
        for _ in range(n_rounds - 1):
            augmented = tta_augment(images, training=True)
            batch_preds.append(model(augmented, training=False).numpy())

        # Element-wise average across all rounds
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)

    return np.concatenate(all_preds, axis=0).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. EVALUATION & METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_predictions(y_true, y_pred, class_names, title=""):
    """Print a detailed classification report and return the confusion matrix."""
    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICATION REPORT {title}")
    print(f"{'=' * 60}\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1, 0]      # Actual=PNEUMONIA, Predicted=NORMAL → missed diagnosis
    fp = cm[0, 1]      # Actual=NORMAL,    Predicted=PNEUMONIA
    print(f"  [CRITICAL] False Negatives (missed PNEUMONIA): {fn}")
    print(f"  [INFO]     False Positives (healthy flagged)  : {fp}")
    return cm


# ═══════════════════════════════════════════════════════════════════════════════
# 12. VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, title, save_path):
    """Render a styled confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Count"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [INFO] Saved → {save_path}")


def plot_training_history(hist_phase1, hist_phase2, save_path="training_history.png"):
    """
    Plot combined loss and accuracy curves from both training phases.
    A vertical dashed line marks the transition from Phase 1 (Feature
    Extraction) to Phase 2 (Fine-Tuning).
    """
    def merge(h1, h2):
        merged = {}
        for key in h1.history:
            merged[key] = h1.history[key] + h2.history[key]
        return merged

    history = merge(hist_phase1, hist_phase2)
    phase1_len = len(hist_phase1.history["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — Loss —
    axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].axvline(x=phase1_len - 0.5, color="gray", ls="--", alpha=0.7,
                    label="Fine-tune start")
    axes[0].set_title("Loss Over Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # — Accuracy —
    axes[1].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].axvline(x=phase1_len - 0.5, color="gray", ls="--", alpha=0.7,
                    label="Fine-tune start")
    axes[1].set_title("Accuracy Over Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [INFO] Saved → {save_path}")


def plot_gradcam_grid(model, base_model, test_ds, class_names,
                      y_true, y_pred, n_samples=8,
                      save_path="gradcam_results.png"):
    """
    Generate a visual grid of Grad-CAM overlays for selected test images.

    Layout:
      Row 1 — Original X-ray with ground-truth label
      Row 2 — Grad-CAM overlay with predicted label (green = correct, red = wrong)

    The function prioritizes showing misclassified samples (if any exist)
    to highlight the model's failure modes, which is the most actionable
    information for a radiologist or ML engineer reviewing the model.
    """
    # Collect images and labels from the dataset (up to total test size)
    images_list, labels_list = [], []
    for images_batch, labels_batch in test_ds:
        for i in range(images_batch.shape[0]):
            images_list.append(images_batch[i].numpy())
            labels_list.append(int(labels_batch[i].numpy().item()))
            if len(images_list) >= len(y_true):
                break
        if len(images_list) >= len(y_true):
            break

    # Select a mix of incorrect and correct predictions
    incorrect = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    correct   = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    n_incorrect = min(len(incorrect), n_samples // 2)
    n_correct   = n_samples - n_incorrect
    selected = (incorrect[:n_incorrect] + correct[:n_correct])[:n_samples]

    if not selected:
        print("  [WARNING] No samples available for Grad-CAM grid.")
        return

    cols = len(selected)
    fig, axes = plt.subplots(2, cols, figsize=(3.5 * cols, 7))
    if cols == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(selected):
        img = images_list[idx]
        true_lbl = class_names[int(y_true[idx])]
        pred_lbl = class_names[int(y_pred[idx])]
        is_correct = (true_lbl == pred_lbl)

        # Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img[np.newaxis, ...], model, base_model)
        overlay = overlay_gradcam(img, heatmap)

        # Row 1: original
        axes[0, col].imshow(img.astype(np.uint8))
        axes[0, col].set_title(f"True: {true_lbl}", fontsize=10)
        axes[0, col].axis("off")

        # Row 2: Grad-CAM
        color = "green" if is_correct else "red"
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Pred: {pred_lbl}", fontsize=10,
                               color=color, fontweight="bold")
        axes[1, col].axis("off")

    fig.suptitle("Grad-CAM Explainability — Model Attention Regions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [INFO] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  MAIN EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── STEP 1: Load Data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 / 9 : LOADING DATASET")
    print("=" * 60)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
    )
    print(f"\n  Class mapping: 0 → {class_names[0]}, 1 → {class_names[1]}")

    # ── STEP 2: Compute Class Weights  [ADVANCED #2] ────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 / 9 : COMPUTING CLASS WEIGHTS")
    print("=" * 60)

    class_weights, class_counts = compute_class_weights(TRAIN_DIR)
    for name, count in sorted(class_counts.items()):
        print(f"  {name:>12s} : {count:>5d} images")
    print(f"  Weights      : {class_weights}")

    # ── STEP 3: Build Model ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 / 9 : BUILDING MODEL")
    print("=" * 60)

    augmentation = build_augmentation_layer()
    model, base_model = build_model(IMG_SIZE, augmentation)
    model.summary()

    # ── STEP 4: Phase 1 — Feature Extraction  [ADVANCED #1a] ────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 / 9 : PHASE 1 — FEATURE EXTRACTION (Frozen Base)")
    print(f"               LR = {PHASE1_LR},  max epochs = {PHASE1_EPOCHS}")
    print("=" * 60 + "\n")

    compile_model(model, PHASE1_LR)

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=build_callbacks("pneumonia_phase1_best.keras"),
        class_weight=class_weights,
    )

    # ── STEP 5: Phase 2 — Fine-Tuning  [ADVANCED #1b] ───────────────────────
    phase1_end = len(history_phase1.epoch)

    print("\n" + "=" * 60)
    print("  STEP 5 / 9 : PHASE 2 — FINE-TUNING")
    print(f"               Unfreezing layers [{FINE_TUNE_FROM}..end]")
    print(f"               LR = {PHASE2_LR},  max epochs = {PHASE2_EPOCHS}")
    print("=" * 60)

    # Unfreeze the top layers of MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen    = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Base model: {trainable} trainable / {frozen} frozen layers\n")

    # Recompile is REQUIRED after changing trainability
    compile_model(model, PHASE2_LR)

    total_epochs = phase1_end + PHASE2_EPOCHS
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=phase1_end,
        callbacks=build_callbacks("pneumonia_model_best.keras"),
        class_weight=class_weights,
    )

    # ── STEP 6: Standard Test Evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 / 9 : EVALUATING ON TEST SET (Standard Inference)")
    print("=" * 60)

    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test Precision: {test_prec:.4f}")
    print(f"  Test Recall   : {test_rec:.4f}")

    # Collect ground truth & predictions
    y_true = np.concatenate(
        [labels.numpy() for _, labels in test_ds], axis=0
    ).flatten().astype(int)

    y_prob_std = model.predict(test_ds, verbose=1).flatten()
    y_pred_std = (y_prob_std >= 0.5).astype(int)

    cm_std = evaluate_predictions(y_true, y_pred_std, class_names,
                                  title="(Standard Inference)")
    plot_confusion_matrix(cm_std, class_names,
                          title="Confusion Matrix — Standard Inference",
                          save_path="confusion_matrix_standard.png")

    # ── STEP 7: Test-Time Augmentation  [ADVANCED #4] ────────────────────────
    print("\n" + "=" * 60)
    print(f"  STEP 7 / 9 : TEST-TIME AUGMENTATION ({TTA_ROUNDS} rounds)")
    print("=" * 60)

    y_prob_tta = predict_with_tta(model, test_ds, n_rounds=TTA_ROUNDS)
    y_pred_tta = (y_prob_tta >= 0.5).astype(int)

    cm_tta = evaluate_predictions(y_true, y_pred_tta, class_names,
                                  title=f"(TTA — {TTA_ROUNDS} rounds)")
    plot_confusion_matrix(cm_tta, class_names,
                          title=f"Confusion Matrix — TTA ({TTA_ROUNDS} rounds)",
                          save_path="confusion_matrix_tta.png")

    # ── Comparison Summary ───────────────────────────────────────────────────
    def _acc(yt, yp):
        return np.mean(yt == yp)
    def _prec(yt, yp):
        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        return tp / (tp + fp + 1e-8)
    def _rec(yt, yp):
        tp = np.sum((yp == 1) & (yt == 1))
        fn = np.sum((yp == 0) & (yt == 1))
        return tp / (tp + fn + 1e-8)

    print("\n" + "=" * 60)
    print("  STANDARD vs TTA COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<14s} {'Standard':>10s} {'TTA':>10s} {'Delta':>10s}")
    print(f"  {'─' * 44}")
    for name, fn in [("Accuracy", _acc), ("Precision", _prec), ("Recall", _rec)]:
        v_std = fn(y_true, y_pred_std)
        v_tta = fn(y_true, y_pred_tta)
        delta = v_tta - v_std
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<14s} {v_std:>10.4f} {v_tta:>10.4f} {sign}{delta:>9.4f}")

    fn_std = cm_std[1, 0]
    fn_tta = cm_tta[1, 0]
    delta_fn = fn_tta - fn_std
    sign_fn = "+" if delta_fn >= 0 else ""
    print(f"  {'False Neg':<14s} {fn_std:>10d} {fn_tta:>10d} {sign_fn}{delta_fn:>9d}")

    # ── STEP 8: Grad-CAM Visualization  [ADVANCED #3] ────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8 / 9 : GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    plot_gradcam_grid(model, base_model, test_ds, class_names,
                      y_true, y_pred_std,
                      n_samples=8, save_path="gradcam_results.png")

    # ── STEP 9: Training History & Save ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 9 / 9 : SAVING OUTPUTS")
    print("=" * 60)

    plot_training_history(history_phase1, history_phase2)

    model.save("pneumonia_model_final.keras")
    print(f"  [INFO] Final model saved → pneumonia_model_final.keras")

    # ── DONE ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ██  PIPELINE COMPLETE")
    print("  ██")
    print("  ██  Outputs:")
    print("  ██    • pneumonia_model_final.keras     (final trained model)")
    print("  ██    • pneumonia_model_best.keras       (best fine-tune checkpoint)")
    print("  ██    • confusion_matrix_standard.png    (standard evaluation)")
    print("  ██    • confusion_matrix_tta.png         (TTA evaluation)")
    print("  ██    • gradcam_results.png              (explainability heatmaps)")
    print("  ██    • training_history.png             (loss & accuracy curves)")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
