"""
Task 2.1: Input Pipeline with tf.data
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Features:
- Reproducible 70/15/15 stratified train/val/test splitting
- tf.data input pipeline with safe decoding, RGB standardization, and aspect-ratio padding
- Training-only data augmentation (Flip, Rotation, Zoom, Contrast)
- Batching (batch_size=32) and AUTOTUNE prefetching
- Exports augmented batch figure and technical pipeline documentation
"""

import os
import sys

# Configure sandboxed runtime directories BEFORE third-party imports
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Constants
RANDOM_SEED = 42
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
MANIFEST_CSV = "evidence/task1/audit_manifest.csv"
PROCESSED_DIR = "data/processed"
EVIDENCE_DIR = "evidence/task2"

# Label encoding: 0 = Real, 1 = AI-Generated
LABEL_MAP = {"Real": 0, "AI-Generated": 1}
INV_LABEL_MAP = {0: "Real", 1: "AI-Generated"}

def create_stratified_splits(manifest_path=MANIFEST_CSV, output_dir=PROCESSED_DIR):
    """Generates and saves stratified 70% Train, 15% Validation, 15% Test splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(manifest_path)
    # Ensure only valid, decodable images are used
    df = df[~df["is_corrupt"]].copy()
    
    # 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["class"]
    )
    
    # Split the 30% temp into 15% Val and 15% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["class"]
    )
    
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[+] Dataset Splits Created (Seed={RANDOM_SEED}):")
    print(f"  - Train Set : {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"  - Val Set   : {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
    print(f"  - Test Set  : {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")
    
    # Print class distribution per split
    print("\n  Class Distribution Across Splits:")
    for name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = s_df["class"].value_counts().to_dict()
        print(f"    * {name:<5}: {counts}")
        
    return train_df, val_df, test_df

def load_and_preprocess_image(filepath, label):
    """
    Reads image from disk, decodes, standardizes to 3-channel RGB,
    and resizes with padding to preserve aspect ratio.
    """
    img_bytes = tf.io.read_file(filepath)
    # decode_image supports JPEG, PNG, GIF, BMP; expand_animations=False ensures 3D tensor
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False, dtype=tf.uint8)
    
    # Ensure static shape for TensorFlow graph
    img.set_shape([None, None, 3])
    
    # Resize with aspect ratio preservation (pads borders with zeros)
    img = tf.image.resize_with_pad(img, target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
    
    # Cast to float32 [0.0, 255.0]
    img = tf.cast(img, tf.float32)
    
    return img, label

def get_data_augmentation():
    """
    Constructs the data augmentation pipeline applied exclusively to training data.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.08, fill_mode="constant", seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.08, fill_mode="constant", seed=RANDOM_SEED),
        tf.keras.layers.RandomContrast(0.1, seed=RANDOM_SEED),
    ], name="data_augmentation")

def build_dataset(split_csv_path, is_training=False, batch_size=BATCH_SIZE, augment=False):
    """
    Builds an optimized tf.data pipeline from a split CSV file.
    """
    df = pd.read_csv(split_csv_path)
    filepaths = df["filepath"].values
    labels = df["class"].map(LABEL_MAP).values.astype(np.int32)
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        # Shuffle with a buffer size matching the training dataset length
        ds = ds.shuffle(buffer_size=len(filepaths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        
    # Parallel decoding and preprocessing
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch tensors
    ds = ds.batch(batch_size, drop_remainder=False)
    
    # Apply data augmentation ONLY if explicitly requested (training set only)
    if is_training and augment:
        augmentation_layer = get_data_augmentation()
        ds = ds.map(lambda x, y: (augmentation_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
    # Prefetch in memory for pipeline concurrency
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds

def plot_augmented_batch(train_ds, output_dir=EVIDENCE_DIR):
    """Extracts a batch from the training pipeline and visualizes augmented images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Take one batch from training dataset
    for images, labels in train_ds.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        break
        
    # Plot a 4x4 grid (16 samples)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle(f"VisionGuard (23L0710): One Augmented Training Batch (16 of {len(images_np)} Images)", 
                 fontsize=15, fontweight="bold")
    
    for i in range(16):
        ax = axes[i // 4, i % 4]
        # Images are in float32 [0, 255], convert to uint8 for plotting
        img_display = np.clip(images_np[i], 0, 255).astype(np.uint8)
        label_text = INV_LABEL_MAP.get(int(labels_np[i]), "Unknown")
        
        ax.imshow(img_display)
        color = "#e74c3c" if label_text == "AI-Generated" else "#2980b9"
        ax.set_title(f"Sample {i+1}: {label_text}", color=color, fontweight="bold", fontsize=11)
        ax.axis("off")
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "augmented_batch.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved augmented batch figure: {output_path}")

def generate_pipeline_documentation(output_dir=EVIDENCE_DIR):
    """Generates comprehensive markdown documentation explaining the pipeline."""
    doc = """# Task 2.1: Input Pipeline with tf.data Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

---

## 1. Dataset Partitioning Strategy
The dataset (973 valid images) is partitioned into three disjoint, stratified splits using a fixed random seed (`42`):
- **Training Set (70%):** 681 images (377 AI-Generated, 304 Real)
- **Validation Set (15%):** 146 images (81 AI-Generated, 65 Real)
- **Test Set (15%):** 146 images (80 AI-Generated, 66 Real)

The splits are permanently recorded in:
- `data/processed/train_split.csv`
- `data/processed/val_split.csv`
- `data/processed/test_split.csv`

---

## 2. tf.data Pipeline Architecture & Stages

```
[Split CSV File]
      │
      ▼
(filepath, label) Tensor Slices
      │
      ▼
tf.data.Dataset.shuffle (buffer=681, seed=42) [Training only]
      │
      ▼
tf.io.read_file (Loads raw file bytes from disk)
      │
      ▼
tf.io.decode_image (Safe decoding to 3-channel RGB uint8)
      │
      ▼
tf.image.resize_with_pad (224x224 px, preserves aspect ratio with zero padding)
      │
      ▼
tf.cast (Float32 tensor in range [0.0, 255.0])
      │
      ▼
batch(batch_size=32)
      │
      ▼
data_augmentation (HorizontalFlip, Rotation, Zoom, Contrast) [Training only]
      │
      ▼
prefetch(tf.data.AUTOTUNE) (Overlaps CPU preprocessing with model execution)
```

### Detailed Stage Breakdown:
1. **Tensor Slicing:**
   Filepaths and integer labels (`0 = Real`, `1 = AI-Generated`) are transformed into an asynchronous streaming dataset.
2. **Safe Decoding & 3-Channel RGB Standardization:**
   Our Task 1.2 audit detected 4 distinct color modes (`RGB`, `RGBA`, `P`, `L`). Using `channels=3` in `tf.io.decode_image` standardizes all formats (dropping alpha transparency and replicating grayscale across 3 channels), ensuring consistent `(224, 224, 3)` tensors without channel mismatch crashes.
3. **Choice of Image Size ($224 \\times 224$ pixels):**
   - Standard benchmark resolution for ImageNet pretrained architectures (`MobileNetV2`, `ResNet50`, `EfficientNetB0`).
   - Strikes an optimal balance between retaining high-frequency texture details (critical for detecting AI generative artifacts) and maintaining fast training throughput on CPU hardware.
4. **Aspect Ratio Preservation (`resize_with_pad`):**
   - Our Task 1.3 EDA proved that aspect ratios vary widely ($0.33$ to $2.5$, with AI peaking at 1:1 and Real peaking at 4:3 and 3:2).
   - Standard bilinear resizing without aspect ratio preservation squashes and stretches images, distorting natural circular contours and facial proportions.
   - `tf.image.resize_with_pad` scales the image along its longest dimension to 224 px while maintaining the true aspect ratio, symmetrically padding the remaining borders with black pixels (zeros).
5. **Data Augmentation (Training ONLY):**
   - Random horizontal flips ($p=0.5$).
   - Small random rotations (factor $\\pm 0.08$ rad, $\\approx \\pm 4.5^\\circ$).
   - Minor random zoom (factor $\\pm 0.08$).
   - Contrast perturbation (factor $0.1$).
   - Applied exclusively to the training dataset. Validation and test sets remain strictly un-augmented to preserve authentic evaluation metrics.
6. **Batching and AUTOTUNE Prefetching:**
   - Grouped into batches of $32$ samples.
   - `prefetch(buffer_size=tf.data.AUTOTUNE)` uses a background worker thread to load batch $N+1$ while batch $N$ is being processed by the model, eliminating CPU I/O stalls.

---

## 3. Evidence Figure
- **Figure:** `evidence/task2/augmented_batch.png`
- **Caption:** *Figure 2.1 — Visualization of 16 augmented training samples from a single batch, demonstrating horizontal flipping, slight zoom/rotation, and black border padding from aspect-ratio preservation.*
"""
    doc_path = os.path.join(output_dir, "input_pipeline_description.md")
    with open(doc_path, "w") as f:
        f.write(doc)
    print(f"[+] Saved pipeline technical documentation: {doc_path}")

def main():
    print("=" * 70)
    print("       TASK 2.1: TF.DATA INPUT PIPELINE & AUGMENTATION AUDIT       ")
    print("=" * 70)
    
    # 1. Create stratified splits
    train_df, val_df, test_df = create_stratified_splits()
    
    # 2. Build training dataset with augmentation
    train_path = os.path.join(PROCESSED_DIR, "train_split.csv")
    train_ds = build_dataset(train_path, is_training=True, augment=True)
    
    # 3. Build validation and test datasets (deterministic, no augmentation)
    val_path = os.path.join(PROCESSED_DIR, "val_split.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_split.csv")
    val_ds = build_dataset(val_path, is_training=False, augment=False)
    test_ds = build_dataset(test_path, is_training=False, augment=False)
    
    # 4. Verify shapes on a batch
    print("\n[*] Testing dataset pipeline throughput and batch shapes...")
    for images, labels in train_ds.take(1):
        print(f"  - Training Batch Image Tensor Shape : {images.shape} (dtype={images.dtype})")
        print(f"  - Training Batch Label Tensor Shape : {labels.shape} (dtype={labels.dtype})")
        print(f"  - Image Value Range                 : [{tf.reduce_min(images):.1f}, {tf.reduce_max(images):.1f}]")
        print(f"  - Unique labels in batch            : {np.unique(labels.numpy())}")
        
    for images, labels in val_ds.take(1):
        print(f"  - Validation Batch Image Shape      : {images.shape} (dtype={images.dtype})")
        
    # 5. Export augmented batch visualization
    plot_augmented_batch(train_ds)
    
    # 6. Generate pipeline documentation
    generate_pipeline_documentation()
    
    print("\n" + "=" * 70)
    print("                TASK 2.1 VERIFICATION COMPLETED                       ")
    print("=" * 70)

if __name__ == "__main__":
    main()
