# Task 3.3: Instrument VisionGuard Training Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**DagsHub Remote Tracking URI:** `https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow`  
**Experiment Name:** `23L0710_VisionGuard`  

---

## 1. Experimental Overview & Remote Run Summary

To rigorously compare experiment instrumentation paradigms, we executed full VisionGuard classification runs on the remote DagsHub MLflow tracking server using two distinct methodologies:

| Approach | Run Name | Run ID | Logged Parameters | Test Accuracy | Test F1 Score | Test ROC-AUC | Live DagsHub Run Link |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Manual Callback** | `visionguard_manual_callback_run` | `b6a6bad4...` | 9 explicit | **74.66%** | **0.7299** | **0.8467** | [View Run on DagsHub](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow/#/experiments/2/runs/b6a6bad43cd947b5932ae65a1f4d812c) |
| **TF Autologging** | `visionguard_autolog_run` | `273a68a3...` | Automatic (15+) | **79.45%** | N/A (Standard) | **0.8615** | [View Run on DagsHub](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow/#/experiments/2/runs/273a68a3ef234c9db43bf4951a1519cf) |

Both runs logged the required audit tags:
- `git_commit`: `bd2193a7c2caa89b2ae3b0df708bbf5342d5ce09`
- `dvc_data_version`: `21fa43c3eb533c1613e51cd0085035d5.dir`
- `random_seed`: `42`
- `author`: `23L0710`

---

## 2. In-Depth Comparative Analysis: What Each Approach Captures vs. Misses

### 1. Manual Instrumentation (Custom Keras Callback)
- **What It Captures:**
  - **Domain & Business-Specific Hyperparameters:** Explicit logging of custom architectural decisions (e.g., number of unfrozen backbone layers, dropout rates, and optimal decision threshold $\tau^* = 0.23$).
  - **Custom Thresholded Test Metrics:** Computes and logs threshold-dependent evaluation metrics (Precision, Recall, F1 score at $\tau^* = 0.23$) on the held-out test set that Keras standard `.evaluate()` does not report out-of-the-box.
  - **Full MLOps Metadata Control:** Seamless injection of external provenance tags linking the run to the exact Git commit SHA and DVC dataset hash.
  - **Clean, Predictable Namespace:** Metric keys are human-readable (`train_loss`, `val_accuracy`, `test_f1`), preventing cluttered dashboards.
- **What It Misses:**
  - **Boilerplate Overhead:** Requires manually creating and maintaining a `tf.keras.callbacks.Callback` subclass and writing explicit logging statements.
  - **Deep Framework Telemetry:** Does not automatically log internal optimizer hyperparameters (such as $\beta_1, \beta_2, \epsilon$, amsgrad) or model graph layer parameter summaries unless explicitly extracted via introspection.

---

### 2. Automated Instrumentation (`mlflow.tensorflow.autolog()`)
- **What It Captures:**
  - **Zero-Boilerplate Convenience:** A single call to `mlflow.tensorflow.autolog()` intercepts `.fit()` calls without altering model training code.
  - **Comprehensive Framework & Optimizer State:** Automatically captures granular Adam parameters (learning rate, $\beta_1, \beta_2, \epsilon$, decay schedule), batch sizes, training steps per epoch, and full Keras layer parameter tables.
  - **Automatic Model Serialization & Signature Inference:** Logs the complete saved model, environment requirements, and input/output tensor signature automatically upon completion.
- **What It Misses:**
  - **Blind to Domain-Specific MLOps Metadata:** Cannot automatically infer external MLOps state (e.g., DVC pointer hashes, Git commit SHAs, random seeds, or business roll numbers) unless manually passed via `mlflow.set_tags()`.
  - **Inability to Evaluate Custom Decision Thresholds:** Only captures scalar metrics compiled in `model.compile()`. It cannot automatically compute Precision, Recall, and F1 at our custom validation-tuned threshold $\tau^* = 0.23$ on the held-out test set.
  - **Metric Clutter:** Generates automated metric namespaces that can produce excessive or duplicate entries in multi-phase transfer learning workflows.

---

## 3. Engineering Conclusion & Best Practice Recommendation

In a production MLOps pipeline, **a hybrid strategy is optimal**:
1. Leverage `mlflow.tensorflow.autolog()` to capture underlying optimizer telemetry, model architecture graphs, and execution environments automatically.
2. Augment autologging with explicit manual logging for **provenance tags** (`git_commit`, `dvc_data_version`, `random_seed`), **custom hyperparameters** (`unfrozen_layers`, `confidence_threshold`), and **business-critical threshold metrics** (`test_f1`, `test_precision`, `test_recall`).
