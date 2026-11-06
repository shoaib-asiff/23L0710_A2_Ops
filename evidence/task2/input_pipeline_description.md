# Task 2.1: Input Pipeline with tf.data Technical Report
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
3. **Choice of Image Size ($224 \times 224$ pixels):**
   - Standard benchmark resolution for ImageNet pretrained architectures (`MobileNetV2`, `ResNet50`, `EfficientNetB0`).
   - Strikes an optimal balance between retaining high-frequency texture details (critical for detecting AI generative artifacts) and maintaining fast training throughput on CPU hardware.
4. **Aspect Ratio Preservation (`resize_with_pad`):**
   - Our Task 1.3 EDA proved that aspect ratios vary widely ($0.33$ to $2.5$, with AI peaking at 1:1 and Real peaking at 4:3 and 3:2).
   - Standard bilinear resizing without aspect ratio preservation squashes and stretches images, distorting natural circular contours and facial proportions.
   - `tf.image.resize_with_pad` scales the image along its longest dimension to 224 px while maintaining the true aspect ratio, symmetrically padding the remaining borders with black pixels (zeros).
5. **Data Augmentation (Training ONLY):**
   - Random horizontal flips ($p=0.5$).
   - Small random rotations (factor $\pm 0.08$ rad, $\approx \pm 4.5^\circ$).
   - Minor random zoom (factor $\pm 0.08$).
   - Contrast perturbation (factor $0.1$).
   - Applied exclusively to the training dataset. Validation and test sets remain strictly un-augmented to preserve authentic evaluation metrics.
6. **Batching and AUTOTUNE Prefetching:**
   - Grouped into batches of $32$ samples.
   - `prefetch(buffer_size=tf.data.AUTOTUNE)` uses a background worker thread to load batch $N+1$ while batch $N$ is being processed by the model, eliminating CPU I/O stalls.

---

## 3. Evidence Figure
- **Figure:** `evidence/task2/augmented_batch.png`
- **Caption:** *Figure 2.1 — Visualization of 16 augmented training samples from a single batch, demonstrating horizontal flipping, slight zoom/rotation, and black border padding from aspect-ratio preservation.*
