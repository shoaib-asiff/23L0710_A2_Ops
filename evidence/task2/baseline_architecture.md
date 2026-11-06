# Task 2.2: Baseline CNN Architecture & Design Rationale
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Model Name:** `VisionGuard_Baseline_CNN`  

---

## 1. Layer-by-Layer Architectural Specifications

| Layer Name             | Layer Type             | Output Shape         |   Param Count | Trainable   |
|:-----------------------|:-----------------------|:---------------------|--------------:|:------------|
| pixel_rescaling        | Rescaling              | (None, 224, 224, 3)  |             0 | True        |
| conv1                  | Conv2D                 | (None, 224, 224, 32) |           896 | True        |
| bn1                    | BatchNormalization     | (None, 224, 224, 32) |           128 | True        |
| pool1                  | MaxPooling2D           | (None, 112, 112, 32) |             0 | True        |
| conv2                  | Conv2D                 | (None, 112, 112, 64) |        18,496 | True        |
| bn2                    | BatchNormalization     | (None, 112, 112, 64) |           256 | True        |
| pool2                  | MaxPooling2D           | (None, 56, 56, 64)   |             0 | True        |
| conv3                  | Conv2D                 | (None, 56, 56, 128)  |        73,856 | True        |
| bn3                    | BatchNormalization     | (None, 56, 56, 128)  |           512 | True        |
| pool3                  | MaxPooling2D           | (None, 28, 28, 128)  |             0 | True        |
| conv4                  | Conv2D                 | (None, 28, 28, 128)  |       147,584 | True        |
| bn4                    | BatchNormalization     | (None, 28, 28, 128)  |           512 | True        |
| pool4                  | MaxPooling2D           | (None, 14, 14, 128)  |             0 | True        |
| global_avg_pool        | GlobalAveragePooling2D | (None, 128)          |             0 | True        |
| dense_features         | Dense                  | (None, 128)          |        16,512 | True        |
| bn_dense               | BatchNormalization     | (None, 128)          |           512 | True        |
| dropout_regularization | Dropout                | (None, 128)          |             0 | True        |
| output_probability     | Dense                  | (None, 1)            |           129 | True        |

### Parameter Summary:
- **Total Parameters:** 259,393 (0.99 MB float32 weights)
- **Trainable Parameters:** 258,433
- **Non-Trainable Parameters (BatchNorm running stats):** 960

---

## 2. Layer Design Rationales & Engineering Explanations

| Layer Name | Type | Purpose & Technical Rationale |
| :--- | :--- | :--- |
| `pixel_rescaling` | `Rescaling` | Converts raw pixel tensors [0.0, 255.0] to normalized [0.0, 1.0]. Prevents numerical instability and exploding gradients during backpropagation. |
| `conv1` (32 filters) | `Conv2D (3x3)` | Extracts low-level visual primitives, such as fine digital noise, gradient edges, and high-frequency pixel transitions characteristic of synthetic generators. |
| `bn1`, `bn2`, `bn3`, `bn4` | `BatchNormalization` | Normalizes intermediate activations across each mini-batch (zero mean, unit variance). Mitigates internal covariate shift, accelerates training convergence, and provides a regularizing effect. |
| `pool1`, `pool2`, `pool3`, `pool4` | `MaxPooling2D (2x2)` | Halves spatial dimensions at each stage (224 -> 112 -> 56 -> 28 -> 14), expanding the receptive field while providing translational invariance and reducing computational load. |
| `conv2` (64 filters) | `Conv2D (3x3)` | Captures intermediate visual patterns, localized textures, and synthetic skin/hair smoothing artifacts. |
| `conv3` (128 filters) | `Conv2D (3x3)` | Captures complex geometric shapes, lighting inconsistencies, and compositional structures. |
| `conv4` (128 filters) | `Conv2D (3x3)` | Synthesizes high-level semantic features across wide receptive fields to distinguish holistic generative patterns from photographic realism. |
| `global_avg_pool` | `GlobalAveragePooling2D` | Computes the spatial mean across each of the 128 feature maps (14x14 -> 1). Compared to Flatten(), this reduces parameter count from 25,088 to 128, eliminating spatial overfitting and enforcing global translation invariance. |
| `dense_features` (128 units) | `Dense (ReLU)` | Fully connected non-linear feature integration layer mapping spatial representations into class-discriminative decision boundaries. |
| `dropout_regularization` | `Dropout (0.4)` | Randomly deactivates 40% of feature activations during training passes, preventing co-adaptation of neurons and mitigating overfitting on small datasets. |
| `output_probability` | `Dense (Sigmoid)` | Outputs a single scalar probability in range [0.0, 1.0], representing the calibrated likelihood that the input image is AI-Generated (Label = 1). |

---

## 3. Justification of Loss, Optimizer, and Metrics

### 1. Loss Function: `BinaryCrossentropy`
- **Formula:** `L(y, y_hat) = - [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]`
- **Justification:** VisionGuard is a mutually exclusive two-class problem (Real vs. AI-Generated). Binary Crossentropy mathematically corresponds to the negative log-likelihood of the Bernoulli distribution, directly penalizing confident incorrect classifications with steep logarithmic loss slopes.

### 2. Optimizer: `Adam` (Learning Rate = 1e-3)
- **Justification:** Adaptive Moment Estimation (Adam) combines the advantages of AdaGrad (handling sparse gradients) and RMSProp (exponentially decaying average of squared gradients). It maintains per-parameter adaptive learning rates, providing fast convergence and robust optimization on convolutional feature maps without requiring manual momentum tuning.

### 3. Evaluation Metrics:
- **Binary Accuracy:** Measures overall correctness across the dataset.
- **Precision:** `TP / (TP + FP)`. Critical for measuring the proportion of flagged images that are genuinely AI-generated (guarding against false accusations on real artists).
- **Recall:** `TP / (TP + FN)`. Measures the detection rate of AI-generated content (guarding against synthetic media slipping through undetected).
- **ROC-AUC:** Measures the Area Under the Receiver Operating Characteristic curve across all possible decision thresholds, assessing the classifier's discriminative power independent of any specific threshold.
