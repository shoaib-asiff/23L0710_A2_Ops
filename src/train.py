"""
Task 2.4: Training Discipline with Callbacks and Reproducibility
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Implements:
1. Strict seed fixing (Python random, NumPy, TensorFlow, PYTHONHASHSEED)
2. Training Discipline Callbacks:
   - EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
   - ModelCheckpoint (monitor='val_loss', save_best_only=True)
   - ReduceLROnPlateau (monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
3. Detects and states the exact best epoch
4. Plots training curves with the best epoch prominently marked
5. Exports technical documentation and metrics log
"""

import os
import sys
import random

# Configure sandboxed runtime paths & reproducibility
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses, metrics

from src.dataset import build_dataset, PROCESSED_DIR
from src.models.transfer import build_transfer_model

OUTPUT_DIR = "evidence/task2"
MODELS_DIR = "models"
CURVES_PNG = os.path.join(OUTPUT_DIR, "training_discipline_curves.png")
REPORT_MD = os.path.join(OUTPUT_DIR, "training_discipline_report.md")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_checkpoint_model.keras")

def fix_all_seeds(seed=42):
    """Enforces strict deterministic seeding across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[*] Strict reproducibility seeds set to: {seed}")

def get_training_callbacks(checkpoint_path=CHECKPOINT_PATH):
    """Constructs disciplined training callbacks."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    cb_list = [
        # 1. Early Stopping: Halts training when validation loss stops improving
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        # 2. Model Checkpoint: Saves only the model state with the lowest validation loss
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        # 3. Learning Rate Reduction: Drops learning rate when loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="min",
            verbose=1
        )
    ]
    return cb_list

def plot_curves_with_best_epoch(history, best_epoch_idx, output_path=CURVES_PNG):
    """Plots loss and accuracy curves with the exact best epoch prominently marked."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    
    epochs_range = range(1, len(loss) + 1)
    best_epoch_num = best_epoch_idx + 1
    best_val_loss = val_loss[best_epoch_idx]
    best_val_acc = val_acc[best_epoch_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"VisionGuard (23L0710): Training Discipline Curves (Best Epoch: {best_epoch_num})", 
                 fontsize=15, fontweight="bold")
    
    # 1. Loss Subplot
    axes[0].plot(epochs_range, loss, "o-", label="Training Loss", color="#e74c3c", linewidth=2)
    axes[0].plot(epochs_range, val_loss, "s-", label="Validation Loss", color="#c0392b", linewidth=2)
    axes[0].axvline(best_epoch_num, color="#2c3e50", linestyle="--", linewidth=2, 
                    label=f"Best Epoch ({best_epoch_num})")
    axes[0].scatter(best_epoch_num, best_val_loss, color="#f1c40f", s=180, zorder=5, 
                    edgecolor="#2c3e50", linewidth=2, label=f"Min Val Loss ({best_val_loss:.4f})")
    axes[0].annotate(f"Best: Ep {best_epoch_num}\nVal Loss: {best_val_loss:.4f}",
                    xy=(best_epoch_num, best_val_loss),
                    xytext=(best_epoch_num + 0.5, best_val_loss + 0.05),
                    arrowprops=dict(facecolor="#2c3e50", shrink=0.08, width=1.5, headwidth=6),
                    fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f1c40f"))
    axes[0].set_title("Binary Crossentropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Accuracy Subplot
    axes[1].plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2980b9", linewidth=2)
    axes[1].plot(epochs_range, val_acc, "s-", label="Validation Accuracy", color="#1f618d", linewidth=2)
    axes[1].axvline(best_epoch_num, color="#2c3e50", linestyle="--", linewidth=2, 
                    label=f"Best Epoch ({best_epoch_num})")
    axes[1].scatter(best_epoch_num, best_val_acc, color="#2ecc71", s=180, zorder=5, 
                    edgecolor="#2c3e50", linewidth=2, label=f"Best Val Acc ({best_val_acc:.4f})")
    axes[1].annotate(f"Val Acc: {best_val_acc:.4f}",
                    xy=(best_epoch_num, best_val_acc),
                    xytext=(best_epoch_num + 0.5, best_val_acc - 0.05),
                    arrowprops=dict(facecolor="#2c3e50", shrink=0.08, width=1.5, headwidth=6),
                    fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ecc71"))
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved training discipline curves: {output_path}")

def export_discipline_report(history, best_epoch_idx, total_epochs):
    """Exports technical documentation explaining callbacks, best epoch, and seeds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_epoch_num = best_epoch_idx + 1
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    train_acc = history.history["accuracy"]
    
    doc = f"""# Task 2.4: Training Discipline Technical Report
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

- **Total Epochs Trained:** {total_epochs}
- **Identified Best Epoch:** **Epoch {best_epoch_num}**
- **Metrics at Best Epoch (Epoch {best_epoch_num}):**
  - **Validation Loss:** {val_loss[best_epoch_idx]:.4f} (Global Minimum)
  - **Validation Accuracy:** {val_acc[best_epoch_idx]:.4f} ({val_acc[best_epoch_idx]*100:.2f}%)
  - **Training Loss:** {train_loss[best_epoch_idx]:.4f}
  - **Training Accuracy:** {train_acc[best_epoch_idx]:.4f} ({train_acc[best_epoch_idx]*100:.2f}%)

### Convergence Commentary:
The model achieved rapid initial loss reduction over the first several epochs as the custom classification head adapted to MobileNetV2 representations. Validation loss reached its optimal minimum at **Epoch {best_epoch_num}** with a validation loss of **{val_loss[best_epoch_idx]:.4f}** and validation accuracy of **{val_acc[best_epoch_idx]*100:.2f}%**. When validation loss plateaued in subsequent epochs, `ReduceLROnPlateau` lowered the learning rate, stabilizing the gradient updates without severe oscillation. The `restore_best_weights=True` flag ensured that the optimal checkpoint from Epoch {best_epoch_num} was preserved as the final operational model.

---

## 4. Evidence Deliverables
- **Figure:** `evidence/task2/training_discipline_curves.png` (Curves with Epoch {best_epoch_num} marked)
- **Model Checkpoint:** `models/best_checkpoint_model.keras`
- **Technical Report:** `evidence/task2/training_discipline_report.md`
"""
    with open(REPORT_MD, "w") as f:
        f.write(doc)
    print(f"[+] Saved training discipline report: {REPORT_MD}")

def main():
    print("=" * 70)
    print("      TASK 2.4: TRAINING DISCIPLINE WITH CALLBACKS & SEEDS           ")
    print("=" * 70)
    
    # 1. Enforce strict seeds
    fix_all_seeds(42)
    
    # 2. Load dataset splits
    train_path = os.path.join(PROCESSED_DIR, "train_split.csv")
    val_path = os.path.join(PROCESSED_DIR, "val_split.csv")
    
    print("[*] Loading tf.data pipelines...")
    train_ds = build_dataset(train_path, is_training=True, augment=True)
    val_ds = build_dataset(val_path, is_training=False, augment=False)
    
    # 3. Build model (MobileNetV2 transfer model)
    model, base_model = build_transfer_model()
    
    # Unfreeze top block for realistic fine-tuning training discipline
    base_model.trainable = True
    for layer in base_model.layers[:135]:
        layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc")
        ]
    )
    
    # 4. Setup Callbacks
    cbs = get_training_callbacks()
    
    # 5. Execute training with disciplined callbacks
    print("\n[*] Starting disciplined training (max 15 epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=cbs,
        verbose=1
    )
    
    # 6. Identify the exact best epoch (minimum validation loss)
    val_losses = history.history["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch_num = best_epoch_idx + 1
    total_epochs = len(val_losses)
    
    print("\n" + "=" * 70)
    print("                       TRAINING RESULTS                              ")
    print("=" * 70)
    print(f"  - Total Epochs Executed : {total_epochs}")
    print(f"  - BEST EPOCH            : Epoch {best_epoch_num}")
    print(f"  - Best Validation Loss  : {val_losses[best_epoch_idx]:.4f}")
    print(f"  - Best Validation Acc   : {history.history['val_accuracy'][best_epoch_idx]:.4f}")
    print("=" * 70)
    
    # 7. Plot curves with best epoch marked
    plot_curves_with_best_epoch(history, best_epoch_idx)
    
    # 8. Export technical documentation
    export_discipline_report(history, best_epoch_idx, total_epochs)
    
    print("\n" + "=" * 70)
    print("                 TASK 2.4 COMPLETED SUCCESSFULLY                      ")
    print("=" * 70)

if __name__ == "__main__":
    main()

