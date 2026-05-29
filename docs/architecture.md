# 🏗️ Model Architecture & Training Strategy

This document describes the deep learning architecture, training decisions, and the reasoning behind each advanced feature in the Pneumonia Detection system.

---

## Overview

| Property | Value |
|---|---|
| **Task** | Binary Classification — NORMAL vs PNEUMONIA |
| **Base Model** | MobileNetV2 (ImageNet pre-trained) |
| **Input Shape** | (224, 224, 3) |
| **Output** | Sigmoid neuron → P(PNEUMONIA) ∈ [0, 1] |
| **Framework** | TensorFlow / Keras |
| **Hardware** | NVIDIA RTX 5050 Mobile 8 GB via WSL2 |

---

## Model Architecture

```
Input (224 × 224 × 3)
    │
    ▼
Data Augmentation Layer        ← RandomFlip, RandomRotation, RandomZoom
    │                             Active only during training=True
    ▼
MobileNetV2 preprocess_input   ← Rescales pixels from [0,255] → [−1, 1]
    │
    ▼
MobileNetV2 Backbone           ← 154 layers, 2.2M parameters
    │                             Frozen in Phase 1 / Partially unfrozen in Phase 2
    ▼
GlobalAveragePooling2D         ← Reduces (7, 7, 1280) → (1280,)
    │
    ▼
Dropout(0.2)                   ← Regularization
    │
    ▼
Dense(1, activation='sigmoid') ← Binary output
```

---

## Training Strategy

### Phase 1 — Feature Extraction

| Parameter | Value |
|---|---|
| Backbone | **Fully frozen** |
| Optimizer | Adam |
| Learning Rate | `1e-4` |
| Max Epochs | 10 |
| Early Stopping | patience=5 |

The randomly-initialized classification head is trained while the MobileNetV2 backbone remains frozen. This allows the head to converge to a reasonable solution before any backbone weights are modified, preventing the large gradients from a random head from corrupting the pre-trained features.

### Phase 2 — Fine-Tuning

| Parameter | Value |
|---|---|
| Unfrozen layers | MobileNetV2 layers from index 100 onward (~55 layers) |
| Optimizer | Adam (recompiled) |
| Learning Rate | `1e-5` (10× lower than Phase 1) |
| Max Epochs | 10 additional |
| Early Stopping | patience=5 |

The top layers of MobileNetV2 are unfrozen at a much lower learning rate. This allows the high-level feature detectors (originally tuned for ImageNet categories) to gradually adapt to chest X-ray pathology textures without catastrophic forgetting of the lower-level edge and texture detectors.

---

## Advanced Features

### 1. Class Imbalance Handling

The Kaggle dataset has a **~3:1 PNEUMONIA-to-NORMAL ratio** in the training set. Without correction, a naive model maximizes accuracy by biasing toward PNEUMONIA, producing high False Negative rates — missed pneumonia cases classified as healthy, which is the most dangerous error in a diagnostic setting.

**Solution:** Inverse-frequency class weights, computed directly from filesystem counts:

```
weight_i = total_samples / (n_classes × count_i)
```

These weights are passed to `model.fit(class_weight=...)`, scaling each sample's contribution to the loss function proportionally.

### 2. Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) produces a spatial heatmap showing which regions of the input X-ray most influenced the model's prediction.

**Implementation steps:**
1. Forward-pass through the MobileNetV2 backbone → last conv feature map (7×7×1280)
2. Record gradients of the predicted class score w.r.t. the feature map via `tf.GradientTape`
3. Global-average-pool the gradients → per-channel importance weights (1280 scalars)
4. Weighted sum of feature map channels → (7×7) raw heatmap
5. Apply ReLU (retain only positive activations) + normalize to [0, 1]
6. Bilinear upsample to (224×224) and overlay on the original image

**Clinical significance:** Allows radiologists to verify the model is attending to lung parenchyma rather than spurious artifacts (text labels, borders, pacemaker leads).

### 3. Test-Time Augmentation (TTA)

At inference time, instead of a single deterministic forward pass, TTA applies `N=10` stochastic augmentations (horizontal flip, ±10% rotation, ±10% zoom) per image and averages the resulting probability vectors.

**Why it works:** CNNs are sensitive to small spatial perturbations. Averaging over augmented views reduces this variance, acting as an implicit ensemble and typically improving accuracy by 1–3%.

**Implementation note:** The model's internal augmentation layers are inactive at inference (`training=False`). TTA uses a separate `keras.Sequential` augmentation pipeline applied explicitly before each forward pass.

---

## Data Pipeline

```python
tf.keras.utils.image_dataset_from_directory(...)
    .prefetch(tf.data.AUTOTUNE)     # No .cache() — prevents WSL2 OOM
```

`.cache()` is intentionally omitted. Caching the full training set into RAM causes `RESOURCE_EXHAUSTED` errors in WSL2 due to its strict memory limits. `.prefetch(AUTOTUNE)` achieves near-equivalent throughput by overlapping data I/O with GPU computation.
