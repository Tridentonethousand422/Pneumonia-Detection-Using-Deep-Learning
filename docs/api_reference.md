# 📖 API Reference

Full reference for all modules and public functions in the Pneumonia Detection system.

---

## `pneumonia_detection.py` — Training Pipeline

### `load_datasets(train_dir, val_dir, test_dir, img_size, batch_size)`
Loads the train/val/test splits using `image_dataset_from_directory`.

| Param | Type | Description |
|---|---|---|
| `train_dir` | `str` | Path to training images directory |
| `val_dir` | `str` | Path to validation images directory |
| `test_dir` | `str` | Path to test images directory |
| `img_size` | `tuple` | Target resize dimensions, e.g. `(224, 224)` |
| `batch_size` | `int` | Number of images per batch |

**Returns:** `(train_ds, val_ds, test_ds, class_names)`

---

### `compute_class_weights(train_dir)`
Computes inverse-frequency class weights from filesystem image counts.

**Returns:** `(class_weights: dict, class_counts: dict)`

---

### `build_augmentation_layer()`
Returns a `keras.Sequential` augmentation pipeline: `RandomFlip`, `RandomRotation`, `RandomZoom`. Active only when `training=True`.

---

### `build_model(img_size, augmentation_layer)`
Builds the full Functional API model with MobileNetV2 backbone.

**Returns:** `(model, base_model)`

---

### `compile_model(model, learning_rate)`
Compiles the model with Adam optimizer, BinaryCrossentropy loss, and Accuracy / Precision / Recall metrics.

---

### `build_callbacks(checkpoint_path)`
Returns a list of `[EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]` callbacks.

---

### `make_gradcam_heatmap(img_array, model, base_model)`
Generates a Grad-CAM heatmap for a single image.

| Param | Type | Description |
|---|---|---|
| `img_array` | `np.ndarray` | Shape `(1, 224, 224, 3)`, values `[0..255]` |
| `model` | `keras.Model` | Full trained model |
| `base_model` | `keras.Model` | MobileNetV2 sub-model |

**Returns:** `heatmap` — `np.ndarray` shape `(7, 7)`, values `[0..1]`

---

### `predict_with_tta(model, dataset, n_rounds=10)`
Runs Test-Time Augmentation over `n_rounds` stochastic forward passes and returns averaged probabilities.

**Returns:** `y_prob` — `np.ndarray` shape `(N,)`

---

## `predict.py` — CLI Inference

### Usage

```bash
python predict.py --image <path> [--model <path>]
```

| Flag | Default | Description |
|---|---|---|
| `--image` / `-i` | *(required)* | Path to input chest X-ray image |
| `--model` / `-m` | `pneumonia_model_best.keras` | Path to trained `.keras` model |

**Output:** Prints prediction + confidence to stdout; saves Grad-CAM overlay to `outputs/<name>_gradcam.png`.

---

### `load_and_preprocess_image(image_path)`
Loads an image, converts to RGB, resizes to `(224, 224)`.

**Returns:** `(img_display: np.ndarray, img_batch: np.ndarray)`

### `run_inference(model, img_batch)`
Single forward pass; returns predicted class, confidence, and raw sigmoid probability.

**Returns:** `(pred_class: str, confidence: float, raw_prob: float)`

### `make_gradcam_heatmap(img_batch, model)`
See training pipeline version — identical algorithm.

### `save_gradcam_figure(img, overlay, pred_class, confidence, save_path)`
Saves a side-by-side matplotlib figure: original image | Grad-CAM overlay.

---

## `report_generator.py` — LLM Integration

### `generate_report(prediction, confidence, model_name, ollama_url, timeout)`
Sends a structured prompt to a local Ollama instance and returns the generated report.

| Param | Type | Default | Description |
|---|---|---|---|
| `prediction` | `str` | *(required)* | `"NORMAL"` or `"PNEUMONIA"` |
| `confidence` | `float` | *(required)* | Model confidence in `[0, 1]` |
| `model_name` | `str` | `"qwen2.5"` | Ollama model identifier |
| `ollama_url` | `str` | auto-detected | Ollama `/api/generate` endpoint |
| `timeout` | `int` | `90` | HTTP timeout in seconds |

**Returns:** `str` — generated report, or a descriptive error message.

**WSL2 note:** The Ollama URL is auto-detected by reading the default gateway from `ip route`, which reliably points to the Windows host when Ollama runs in PowerShell.

---

## `app.py` — Streamlit Dashboard

### Cached Resources

```python
@st.cache_resource
def load_model() -> keras.Model
```
Loads the `.keras` model once into memory; persists across all Streamlit reruns and user sessions.

### Key Session State Keys

| Key | Type | Description |
|---|---|---|
| `results` | `dict` | Holds `pred_class`, `confidence`, `overlay` after analysis |
| `report` | `str` | Holds the LLM-generated report text |
| `_last_filename` | `str` | Tracks uploaded file name to reset state on new uploads |
