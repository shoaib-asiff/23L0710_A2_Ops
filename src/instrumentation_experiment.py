"""
Task 3.3: Instrument VisionGuard Training (Manual Callback vs. TensorFlow Autologging)
Author: 23L0710
Project: 23L0710_A2_Ops - VisionGuard

Features:
- Instruments VisionGuard classification training using two distinct approaches:
  1. Manual Callback Logging: Custom Keras callback explicitly logging parameters,
     per-epoch dynamics, custom threshold test metrics (F1, Precision, Recall),
     and MLOps tags (Git commit, DVC data MD5, random seed).
  2. Automated Logging: Using mlflow.tensorflow.autolog().
- Streams both runs directly to the remote DagsHub MLflow server (Experiment: 23L0710_VisionGuard).
- Compiles an in-depth comparative report analyzing what each approach captures vs. misses.
"""

import os
import sys
import subprocess
import yaml

# Configure sandbox-friendly paths BEFORE third-party imports
os.environ["KERAS_HOME"] = "/tmp/.keras"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mlflow
import mlflow.tensorflow

from src.dataset import build_dataset, PROCESSED_DIR

# Constants & Paths
EXPERIMENT_NAME = "23L0710_VisionGuard"
EVIDENCE_DIR = "evidence/task3"
REPORT_MD = os.path.join(EVIDENCE_DIR, "instrumentation_comparison_report.md")
MANUAL_CURVES_PNG = os.path.join(EVIDENCE_DIR, "manual_run_curves.png")
AUTOLOG_CURVES_PNG = os.path.join(EVIDENCE_DIR, "autolog_run_curves.png")

RANDOM_SEED = 42
TAU_STAR = 0.23  # Tuned decision threshold from Task 2.5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_git_commit_hash():
    """Retrieves current Git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown_commit"


def get_dvc_data_version():
    """Extracts DVC MD5 hash from archive.dvc pointer file."""
    try:
        with open("archive.dvc", "r") as f:
            dvc_data = yaml.safe_load(f)
        return dvc_data["outs"][0]["md5"]
    except Exception:
        return "unknown_dvc_hash"


class VisionGuardManualCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback that logs per-epoch train and validation metrics explicitly."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step = epoch + 1
        mlflow.log_metric("train_loss", float(logs.get("loss", 0.0)), step=step)
        mlflow.log_metric("train_accuracy", float(logs.get("accuracy", 0.0)), step=step)
        mlflow.log_metric("train_auc", float(logs.get("auc", 0.0)), step=step)
        mlflow.log_metric("val_loss", float(logs.get("val_loss", 0.0)), step=step)
        mlflow.log_metric("val_accuracy", float(logs.get("val_accuracy", 0.0)), step=step)
        mlflow.log_metric("val_auc", float(logs.get("val_auc", 0.0)), step=step)


def build_mobilenet_classifier(learning_rate=1e-4, unfrozen_layers=20, dropout_rate=0.3):
    """Constructs MobileNetV2 with partially unfrozen top layers and classification head."""
    input_shape = (224, 224, 3)
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Lambda(preprocess_input, name="mobilenetv2_preprocess")(inputs)
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model._name = "mobilenetv2_backbone"
    base_model.trainable = True
    
    cutoff = len(base_model.layers) - unfrozen_layers
    for i, layer in enumerate(base_model.layers):
        if i < cutoff:
            layer.trainable = False
        else:
            layer.trainable = True
            
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dense(128, activation="relu", name="dense128")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="VisionGuard_MobileNetV2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.AUC(name="auc")]
    )
    return model


def evaluate_test_set(model, test_ds):
    """Evaluates model on test dataset and computes standard and custom threshold metrics."""
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        probs = model(images, training=False).numpy().flatten()
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy().flatten())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs >= TAU_STAR).astype(int)
    
    metrics_dict = {
        "test_accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "test_precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true, y_pred_probs))
    }
    return metrics_dict


def plot_training_curves(history, title, output_path):
    """Renders loss and accuracy curve figure for run artifact upload."""
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    
    ax1.plot(epochs, history.history["loss"], "o-", label="Train Loss", color="#2980b9", lw=2)
    ax1.plot(epochs, history.history["val_loss"], "s--", label="Val Loss", color="#c0392b", lw=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    ax2.plot(epochs, history.history["accuracy"], "o-", label="Train Accuracy", color="#27ae60", lw=2)
    ax2.plot(epochs, history.history["val_accuracy"], "s--", label="Val Accuracy", color="#f39c12", lw=2)
    ax2.set_title("Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    return output_path


def run_instrumentation_experiment():
    print("="*70)
    print("      TASK 3.3: INSTRUMENT VISIONGUARD TRAINING (MANUAL VS AUTOLOG)   ")
    print("="*70)
    
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    git_commit = get_git_commit_hash()
    dvc_hash = get_dvc_data_version()
    
    print(f"[*] MLflow Tracking URI: {tracking_uri}")
    print(f"[*] DagsHub User: {username}")
    print(f"[*] Git Commit Hash: {git_commit}")
    print(f"[*] DVC Data Version (MD5): {dvc_hash}")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load VisionGuard Data
    print("\n[*] Loading VisionGuard Train, Val, and Test pipelines...")
    train_csv = os.path.join(PROCESSED_DIR, "train_split.csv")
    val_csv = os.path.join(PROCESSED_DIR, "val_split.csv")
    test_csv = os.path.join(PROCESSED_DIR, "test_split.csv")
    
    train_ds = build_dataset(train_csv, batch_size=32, is_training=True, augment=True)
    val_ds = build_dataset(val_csv, batch_size=32, is_training=False, augment=False)
    test_ds = build_dataset(test_csv, batch_size=32, is_training=False, augment=False)
    
    run_records = []
    
    # ==========================================================================
    # APPROACH A: MANUAL CALLBACK LOGGING
    # ==========================================================================
    print("\n" + "-"*70)
    print("    RUN 1: APPROACH A — MANUAL INSTRUMENTATION VIA KERAS CALLBACK     ")
    print("-"*70)
    
    run1_name = "visionguard_manual_callback_run"
    with mlflow.start_run(run_name=run1_name) as run1:
        run1_id = run1.info.run_id
        
        # 1. Log explicit hyperparameters
        params_manual = {
            "model_architecture": "MobileNetV2",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 3,
            "unfrozen_layers": 20,
            "dropout_rate": 0.3,
            "optimizer": "Adam",
            "loss_function": "binary_crossentropy",
            "decision_threshold_tau": TAU_STAR
        }
        mlflow.log_params(params_manual)
        
        # 2. Log explicit MLOps tags
        tags_manual = {
            "author": "23L0710",
            "roll_number": "23L0710",
            "project": "23L0710_A2_Ops",
            "stage": "task3.3_instrumentation",
            "logging_method": "manual_callback",
            "git_commit": git_commit,
            "dvc_data_version": dvc_hash,
            "random_seed": str(RANDOM_SEED)
        }
        mlflow.set_tags(tags_manual)
        
        # 3. Train model with custom callback
        model_manual = build_mobilenet_classifier(learning_rate=1e-4, unfrozen_layers=20, dropout_rate=0.3)
        history_manual = model_manual.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            callbacks=[VisionGuardManualCallback()],
            verbose=1
        )
        
        # 4. Evaluate on test set and log test metrics
        test_metrics_manual = evaluate_test_set(model_manual, test_ds)
        mlflow.log_metrics(test_metrics_manual)
        
        # 5. Log curve plot and model artifact
        plot_training_curves(history_manual, "VisionGuard: Manual Callback Run Curves", MANUAL_CURVES_PNG)
        mlflow.log_artifact(MANUAL_CURVES_PNG, artifact_path="figures")
        mlflow.tensorflow.log_model(model_manual, artifact_path="model")
        
        run1_url = f"{tracking_uri}/#/experiments/2/runs/{run1_id}"
        print(f"[+] Manual Run Completed! Run ID: {run1_id}")
        print(f"[+] Test Accuracy: {test_metrics_manual['test_accuracy']*100:.2f}% | Test F1: {test_metrics_manual['test_f1']:.4f}")
        print(f"[+] Live URL: {run1_url}")
        
        run_records.append({
            "Approach": "Manual Logging (Callback)",
            "Run Name": run1_name,
            "Run ID": run1_id,
            "Logged Params Count": len(params_manual),
            "Logged Tags": "git_commit, dvc_data_version, seed, author, method",
            "Test Accuracy": f"{test_metrics_manual['test_accuracy']*100:.2f}%",
            "Test F1": f"{test_metrics_manual['test_f1']:.4f}",
            "Test ROC-AUC": f"{test_metrics_manual['test_roc_auc']:.4f}",
            "Run URL": run1_url
        })
        
    # ==========================================================================
    # APPROACH B: AUTOMATED LOGGING (mlflow.tensorflow.autolog)
    # ==========================================================================
    print("\n" + "-"*70)
    print("    RUN 2: APPROACH B — MLFLOW TENSORFLOW AUTOLOGGING                 ")
    print("-"*70)
    
    # Enable TensorFlow autologging
    mlflow.tensorflow.autolog(
        log_models=True,
        log_datasets=False,
        registered_model_name=None
    )
    
    run2_name = "visionguard_autolog_run"
    with mlflow.start_run(run_name=run2_name) as run2:
        run2_id = run2.info.run_id
        
        # Set required tags
        tags_autolog = {
            "author": "23L0710",
            "roll_number": "23L0710",
            "project": "23L0710_A2_Ops",
            "stage": "task3.3_instrumentation",
            "logging_method": "tensorflow_autolog",
            "git_commit": git_commit,
            "dvc_data_version": dvc_hash,
            "random_seed": str(RANDOM_SEED)
        }
        mlflow.set_tags(tags_autolog)
        
        # Build identical model
        model_autolog = build_mobilenet_classifier(learning_rate=1e-4, unfrozen_layers=20, dropout_rate=0.3)
        
        # Fit model (autologging automatically tracks params, metrics, model)
        history_autolog = model_autolog.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            verbose=1
        )
        
        # Evaluate test set
        test_eval = model_autolog.evaluate(test_ds, verbose=0)
        mlflow.log_metric("test_eval_loss", float(test_eval[0]))
        mlflow.log_metric("test_eval_accuracy", float(test_eval[1]))
        mlflow.log_metric("test_eval_auc", float(test_eval[2]))
        
        plot_training_curves(history_autolog, "VisionGuard: Autolog Run Curves", AUTOLOG_CURVES_PNG)
        mlflow.log_artifact(AUTOLOG_CURVES_PNG, artifact_path="figures")
        
        run2_url = f"{tracking_uri}/#/experiments/2/runs/{run2_id}"
        print(f"[+] Autolog Run Completed! Run ID: {run2_id}")
        print(f"[+] Test Accuracy: {test_eval[1]*100:.2f}% | Test AUC: {test_eval[2]:.4f}")
        print(f"[+] Live URL: {run2_url}")
        
        run_records.append({
            "Approach": "Automated (TF Autolog)",
            "Run Name": run2_name,
            "Run ID": run2_id,
            "Logged Params Count": "Automatic (optimizer, steps, epochs)",
            "Logged Tags": "git_commit, dvc_data_version, seed, author, method",
            "Test Accuracy": f"{test_eval[1]*100:.2f}%",
            "Test F1": "N/A (Autolog only evaluates built-in metrics)",
            "Test ROC-AUC": f"{test_eval[2]:.4f}",
            "Run URL": run2_url
        })
        
    # Disable autologging to prevent interference with subsequent scripts
    mlflow.tensorflow.autolog(disable=True)
    
    # Generate comprehensive comparative documentation
    df_comp = pd.DataFrame(run_records)
    generate_comparison_report(df_comp, tracking_uri)
    
    print("\n" + "="*70)
    print("           TASK 3.3 COMPLETED SUCCESSFULLY!           ")
    print("="*70)


def generate_comparison_report(df_comp, tracking_uri):
    """Compiles the Task 3.3 comprehensive comparison report."""
    report_content = f"""# Task 3.3: Instrument VisionGuard Training Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**DagsHub Remote Tracking URI:** `{tracking_uri}`  
**Experiment Name:** `{EXPERIMENT_NAME}`  

---

## 1. Experimental Overview & Remote Run Summary

To rigorously compare experiment instrumentation paradigms, we executed full VisionGuard classification runs on the remote DagsHub MLflow tracking server using two distinct methodologies:

| Approach | Run Name | Run ID | Logged Parameters | Test Accuracy | Test F1 Score | Test ROC-AUC | Live DagsHub Run Link |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Manual Callback** | `{df_comp.loc[0, 'Run Name']}` | `{df_comp.loc[0, 'Run ID'][:8]}...` | {df_comp.loc[0, 'Logged Params Count']} explicit | **{df_comp.loc[0, 'Test Accuracy']}** | **{df_comp.loc[0, 'Test F1']}** | **{df_comp.loc[0, 'Test ROC-AUC']}** | [View Run on DagsHub]({df_comp.loc[0, 'Run URL']}) |
| **TF Autologging** | `{df_comp.loc[1, 'Run Name']}` | `{df_comp.loc[1, 'Run ID'][:8]}...` | Automatic (15+) | **{df_comp.loc[1, 'Test Accuracy']}** | N/A (Standard) | **{df_comp.loc[1, 'Test ROC-AUC']}** | [View Run on DagsHub]({df_comp.loc[1, 'Run URL']}) |

Both runs logged the required audit tags:
- `git_commit`: `{get_git_commit_hash()}`
- `dvc_data_version`: `{get_dvc_data_version()}`
- `random_seed`: `42`
- `author`: `23L0710`

---

## 2. In-Depth Comparative Analysis: What Each Approach Captures vs. Misses

### 1. Manual Instrumentation (Custom Keras Callback)
- **What It Captures:**
  - **Domain & Business-Specific Hyperparameters:** Explicit logging of custom architectural decisions (e.g., number of unfrozen backbone layers, dropout rates, and optimal decision threshold $\\tau^* = 0.23$).
  - **Custom Thresholded Test Metrics:** Computes and logs threshold-dependent evaluation metrics (Precision, Recall, F1 score at $\\tau^* = 0.23$) on the held-out test set that Keras standard `.evaluate()` does not report out-of-the-box.
  - **Full MLOps Metadata Control:** Seamless injection of external provenance tags linking the run to the exact Git commit SHA and DVC dataset hash.
  - **Clean, Predictable Namespace:** Metric keys are human-readable (`train_loss`, `val_accuracy`, `test_f1`), preventing cluttered dashboards.
- **What It Misses:**
  - **Boilerplate Overhead:** Requires manually creating and maintaining a `tf.keras.callbacks.Callback` subclass and writing explicit logging statements.
  - **Deep Framework Telemetry:** Does not automatically log internal optimizer hyperparameters (such as $\\beta_1, \\beta_2, \\epsilon$, amsgrad) or model graph layer parameter summaries unless explicitly extracted via introspection.

---

### 2. Automated Instrumentation (`mlflow.tensorflow.autolog()`)
- **What It Captures:**
  - **Zero-Boilerplate Convenience:** A single call to `mlflow.tensorflow.autolog()` intercepts `.fit()` calls without altering model training code.
  - **Comprehensive Framework & Optimizer State:** Automatically captures granular Adam parameters (learning rate, $\\beta_1, \\beta_2, \\epsilon$, decay schedule), batch sizes, training steps per epoch, and full Keras layer parameter tables.
  - **Automatic Model Serialization & Signature Inference:** Logs the complete saved model, environment requirements, and input/output tensor signature automatically upon completion.
- **What It Misses:**
  - **Blind to Domain-Specific MLOps Metadata:** Cannot automatically infer external MLOps state (e.g., DVC pointer hashes, Git commit SHAs, random seeds, or business roll numbers) unless manually passed via `mlflow.set_tags()`.
  - **Inability to Evaluate Custom Decision Thresholds:** Only captures scalar metrics compiled in `model.compile()`. It cannot automatically compute Precision, Recall, and F1 at our custom validation-tuned threshold $\\tau^* = 0.23$ on the held-out test set.
  - **Metric Clutter:** Generates automated metric namespaces that can produce excessive or duplicate entries in multi-phase transfer learning workflows.

---

## 3. Engineering Conclusion & Best Practice Recommendation

In a production MLOps pipeline, **a hybrid strategy is optimal**:
1. Leverage `mlflow.tensorflow.autolog()` to capture underlying optimizer telemetry, model architecture graphs, and execution environments automatically.
2. Augment autologging with explicit manual logging for **provenance tags** (`git_commit`, `dvc_data_version`, `random_seed`), **custom hyperparameters** (`unfrozen_layers`, `confidence_threshold`), and **business-critical threshold metrics** (`test_f1`, `test_precision`, `test_recall`).
"""
    with open(REPORT_MD, "w") as f:
        f.write(report_content)
    print(f"[+] Saved comparative instrumentation report to: {REPORT_MD}")


if __name__ == "__main__":
    run_instrumentation_experiment()

