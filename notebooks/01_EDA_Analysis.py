# %% [markdown]
# # 🫁 Chest X-Ray Pneumonia Detection — Exploratory Data Analysis
#
# This notebook performs a thorough visual and statistical exploration
# of the Kaggle "Chest X-Ray Images (Pneumonia)" dataset before model training.
#
# **Key questions we answer:**
# 1. How imbalanced are the classes?
# 2. What do typical NORMAL vs PNEUMONIA X-rays look like?
# 3. Are there measurable pixel-intensity differences between the two classes?

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
import os
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict

# Resolve paths relative to this script (works from any working directory)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)                       # Resize target for intensity analysis
INTENSITY_SAMPLE_SIZE = 300                 # Images per class for intensity analysis
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Aesthetic defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

print(f"[INFO] Dataset root: {DATASET_DIR}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — DATASET OVERVIEW: Count images per class per split
# ═══════════════════════════════════════════════════════════════════════════════

counts = defaultdict(dict)

for split in SPLITS:
    for cls in CLASS_NAMES:
        class_dir = pathlib.Path(DATASET_DIR) / split / cls
        if class_dir.exists():
            n = sum(1 for f in class_dir.iterdir() if f.is_file())
            counts[split][cls] = n
        else:
            counts[split][cls] = 0

# Print summary table
print(f"\n{'Split':<10} {'NORMAL':>10} {'PNEUMONIA':>12} {'Total':>10} {'Ratio (P/N)':>14}")
print("─" * 58)
for split in SPLITS:
    n = counts[split]["NORMAL"]
    p = counts[split]["PNEUMONIA"]
    total = n + p
    ratio = f"{p / n:.2f}x" if n > 0 else "N/A"
    print(f"{split:<10} {n:>10,} {p:>12,} {total:>10,} {ratio:>14}")

total_n = sum(counts[s]["NORMAL"] for s in SPLITS)
total_p = sum(counts[s]["PNEUMONIA"] for s in SPLITS)
print("─" * 58)
print(f"{'TOTAL':<10} {total_n:>10,} {total_p:>12,} {total_n + total_p:>10,}")

# %% [markdown]
# ## 1. Class Distribution
#
# The training set is heavily imbalanced — roughly **3:1** PNEUMONIA to NORMAL.
# This motivates the use of **class weights** or oversampling during training.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — CLASS DISTRIBUTION BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 5), sharey=True)

palette = {"NORMAL": "#4CAF50", "PNEUMONIA": "#E53935"}

for ax, split in zip(axes, SPLITS):
    classes = list(counts[split].keys())
    values  = list(counts[split].values())
    bars = ax.bar(classes, values, color=[palette[c] for c in classes],
                  edgecolor="white", linewidth=1.5, width=0.6)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title(f"{split.upper()} Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Images" if split == "train" else "")
    ax.set_ylim(0, max(values) * 1.15)

fig.suptitle("Class Distribution Across Dataset Splits",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved → class_distribution.png")

# %% [markdown]
# ## 2. Sample X-Ray Images
#
# Visual inspection of 4 random samples from each class (training set).
# PNEUMONIA X-rays typically show **diffuse opacities or consolidation**
# in the lung fields, while NORMAL X-rays show **clear, well-aerated lungs**.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — SAMPLE IMAGE GRID: 4 NORMAL vs 4 PNEUMONIA (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════

n_samples = 4
fig, axes = plt.subplots(2, n_samples, figsize=(3.5 * n_samples, 7))

for row, cls in enumerate(CLASS_NAMES):
    class_dir = pathlib.Path(DATASET_DIR) / "train" / cls
    all_files = sorted([f for f in class_dir.iterdir() if f.is_file()])
    sampled = random.sample(all_files, min(n_samples, len(all_files)))

    for col, img_path in enumerate(sampled):
        img = Image.open(img_path).convert("L")          # Grayscale for X-ray
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"{cls}", fontsize=10,
                                 color=palette.get(cls, "black"),
                                 fontweight="bold")
        axes[row, col].axis("off")

fig.suptitle("Sample Chest X-Ray Images (Training Set)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "sample_images.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved → sample_images.png")

# %% [markdown]
# ## 3. Pixel Intensity Analysis
#
# We compare the **per-image mean pixel intensity** distributions between
# the two classes. Pneumonia cases are expected to show **higher mean
# intensity** due to lung opacification (white/cloudy areas on the X-ray).
#
# We also compute the **average composite X-ray** for each class to
# visualize global spatial differences.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — PIXEL INTENSITY DISTRIBUTIONS + AVERAGE COMPOSITE IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

def load_grayscale_samples(class_dir, n_samples, img_size):
    """Load and resize a random sample of images as grayscale numpy arrays."""
    all_files = [f for f in pathlib.Path(class_dir).iterdir() if f.is_file()]
    sampled = random.sample(all_files, min(n_samples, len(all_files)))
    images = []
    for f in sampled:
        try:
            img = Image.open(f).convert("L").resize(img_size)
            images.append(np.array(img, dtype=np.float32))
        except Exception:
            continue       # Skip corrupt files silently
    return np.array(images)


# Load samples from training set
normal_imgs    = load_grayscale_samples(
    os.path.join(DATASET_DIR, "train", "NORMAL"), INTENSITY_SAMPLE_SIZE, IMG_SIZE
)
pneumonia_imgs = load_grayscale_samples(
    os.path.join(DATASET_DIR, "train", "PNEUMONIA"), INTENSITY_SAMPLE_SIZE, IMG_SIZE
)

print(f"[INFO] Loaded {len(normal_imgs)} NORMAL and {len(pneumonia_imgs)} PNEUMONIA samples")

# ── 3a. Per-image mean intensity histogram ──────────────────────────────────
normal_means    = normal_imgs.mean(axis=(1, 2))
pneumonia_means = pneumonia_imgs.mean(axis=(1, 2))

fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(normal_means, label=f"NORMAL (μ={normal_means.mean():.1f})",
            color=palette["NORMAL"], fill=True, alpha=0.3, linewidth=2, ax=ax)
sns.kdeplot(pneumonia_means, label=f"PNEUMONIA (μ={pneumonia_means.mean():.1f})",
            color=palette["PNEUMONIA"], fill=True, alpha=0.3, linewidth=2, ax=ax)

ax.set_xlabel("Mean Pixel Intensity (0 = black, 255 = white)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Per-Image Mean Pixel Intensity Distribution",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "intensity_distribution.png"), dpi=150)
plt.show()
print("[INFO] Saved → intensity_distribution.png")

# ── 3b. Average composite X-ray for each class ─────────────────────────────
avg_normal    = normal_imgs.mean(axis=0)
avg_pneumonia = pneumonia_imgs.mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Average NORMAL
axes[0].imshow(avg_normal, cmap="gray")
axes[0].set_title("Average NORMAL X-Ray", fontsize=12, fontweight="bold",
                  color=palette["NORMAL"])
axes[0].axis("off")

# Average PNEUMONIA
axes[1].imshow(avg_pneumonia, cmap="gray")
axes[1].set_title("Average PNEUMONIA X-Ray", fontsize=12, fontweight="bold",
                  color=palette["PNEUMONIA"])
axes[1].axis("off")

# Difference map (highlights regions that differ between classes)
diff = avg_pneumonia - avg_normal
axes[2].imshow(diff, cmap="RdBu_r", vmin=-30, vmax=30)
axes[2].set_title("Difference Map\n(PNEUMONIA − NORMAL)", fontsize=12, fontweight="bold")
axes[2].axis("off")

fig.suptitle("Average Composite X-Rays & Class Difference",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "average_xrays.png"), dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved → average_xrays.png")

# %% [markdown]
# ## Key Takeaways
#
# | Finding | Implication |
# |---|---|
# | **3:1 class imbalance** | Must use class weights or oversampling to prevent bias toward PNEUMONIA |
# | **Pneumonia X-rays show diffuse opacities** | CNN must learn to detect subtle textural differences in lung fields |
# | **Higher mean pixel intensity in PNEUMONIA** | Confirms lung opacification is a measurable signal, not just visual |
# | **Difference map highlights central lung fields** | The model should attend to mediastinal and lower-lobe regions |
