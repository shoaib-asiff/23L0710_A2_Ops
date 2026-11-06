# Task 2.4: Training Discipline Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  

---

## 1. Reproducibility & Seed Discipline
To ensure end-to-end reproducibility across training runs, random number generators across all underlying libraries were locked to a fixed seed (`42`):
- `random.seed(42)` — Standard library Python randomization
- `np.random.seed(42)` — NumPy array shuffling and splits
- `tf.random.set_seed(42)` — TensorFlow graph operations, weight initializers, and dropout masks
- `os.environ['PYTHONHASHSEED'] = '42'` — Enforces deterministic dictionary and hash table ordering

---

## 2. Callback Architecture & Technical Justifications

| Callback | Configured Parameters | Purpose & Operational Rationale |
| :--- | :--- | :--- |
| `EarlyStopping` | `monitor='val_loss'`, `patience=5`, `restore_best_weights=True` | Continuously monitors validation loss. If `val_loss` fails to achieve a new minimum for 5 consecutive epochs, training terminates prematurely to prevent overfitting. Crucially, `restore_best_weights=True` ensures the model state is automatically rolled back to the exact best epoch. |
| `ModelCheckpoint` | `filepath='models/best_checkpoint_model.keras'`, `monitor='val_loss'`, `save_best_only=True` | Serializes the entire Keras model graph and weights strictly when validation loss improves. Guarantees that subsequent epochs with rising validation loss do not overwrite the optimal model. |
| `ReduceLROnPlateau` | `monitor='val_loss'`, `factor=0.5`, `patience=2`, `min_lr=1e-6` | Decays the learning rate by 50% whenever validation loss plateaus for 2 epochs. Allows coarse-grained optimization early on, followed by fine-grained gradient descent steps near the loss minimum. |

---

## 3. Best Epoch Analysis & Findings

- **Total Epochs Trained:** 8
- **Identified Best Epoch:** **Epoch 3**
- **Metrics at Best Epoch (Epoch 3):**
  - **Validation Loss:** 0.5884 (Global Minimum)
  - **Validation Accuracy:** 0.6781 (67.81%)
  - **Training Loss:** 0.5193
  - **Training Accuracy:** 0.7680 (76.80%)

### Convergence Commentary:
The model achieved rapid initial loss reduction over the first several epochs as the custom classification head adapted to MobileNetV2 representations. Validation loss reached its optimal minimum at **Epoch 3** with a validation loss of **0.5884** and validation accuracy of **67.81%**. When validation loss plateaued in subsequent epochs, `ReduceLROnPlateau` lowered the learning rate, stabilizing the gradient updates without severe oscillation. The `restore_best_weights=True` flag ensured that the optimal checkpoint from Epoch 3 was preserved as the final operational model.

---

## 4. Evidence Deliverables
- **Figure:** `evidence/task2/training_discipline_curves.png` (Curves with Epoch 3 marked)
- **Model Checkpoint:** `models/best_checkpoint_model.keras`
- **Technical Report:** `evidence/task2/training_discipline_report.md`
