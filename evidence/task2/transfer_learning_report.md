# Task 2.3: Pretrained Transfer Learning Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Backbone:** `MobileNetV2` (Pretrained on ImageNet-1k)  

---

## 1. Transfer Learning Methodology

### Two-Stage Training Paradigm:
1. **Phase 1 — Feature Extraction (Frozen Backbone):**
   - The entire `MobileNetV2` backbone (154 layers) is frozen (`trainable = False`).
   - A custom classification head is appended:
     `GlobalAveragePooling2D` -> `BatchNormalization` -> `Dense(128, ReLU)` -> `Dropout(0.4)` -> `Dense(1, Sigmoid)`.
   - The network is trained with `Adam(lr=1e-3)` for 6 epochs.
   - **Trainable Layers in Phase 1:** Only the top classification head (7 trainable layers).
   
2. **Phase 2 — Partial Fine-Tuning (Unfreezing Top Block):**
   - The top block of MobileNetV2 (from layer index 135 to 154, encompassing `block_16` and the final 1x1 projection convolution `Conv_1`) is unfrozen.
   - Lower convolutional layers (layers 0 to 134) remain strictly frozen to preserve fundamental edge and primitive feature detectors learned on ImageNet.
   - The model is recompiled with a drastically reduced learning rate: `Adam(lr=1e-5)`.
   - **Trainable Layers in Phase 2:** Top classification head + 19 backbone layers (26 total trainable layers).

---

## 2. Pretrained Model Preprocessing Justification

### Why `mobilenet_v2.preprocess_input` is Essential:
- **Mathematical Transformation:**
  `preprocess_input(x) = (x / 127.5) - 1.0`
- **Justification:**
  - `MobileNetV2` was pretrained on ImageNet with pixel values normalized symmetrically to the interval `[-1.0, 1.0]`.
  - Passing inputs scaled to `[0.0, 1.0]` (standard) or `[0, 255]` (raw) produces a severe input distribution shift (covariate shift).
  - Preserving the exact normalization range ensures that activations entering the first convolutional layer match the statistical expectations of the pretrained filter weights, preventing gradient degradation and negative transfer.

---

## 3. Trainable Layer Counts per Stage

| Training Stage | Status of Backbone | Trainable Parameters | Non-Trainable Parameters | Key Trainable Layers |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 1 (Feature Extraction)** | Fully Frozen (0/154 layers) | 165,121 | 2,258,240 | `head_bn1`, `head_dense128`, `head_output` |
| **Phase 2 (Fine-Tuning)** | Top 20 Layers Unfrozen | 412,417 | 2,010,944 | `block_16_*`, `Conv_1`, `head_*` |

---

## 4. Training Results & Performance Summary

- **Phase 1 Final Validation Accuracy:** 0.7055 (Val Loss: 0.6823)
- **Phase 2 Best Validation Accuracy:** 0.7329 (Best Val Loss: 0.6753)
- **Figure Deliverable:** `evidence/task2/transfer_learning_curves.png`
- **Phase 1 Layer List:** `evidence/task2/transfer_layers_phase1.txt`
- **Phase 2 Layer List:** `evidence/task2/transfer_layers_phase2.txt`
