# Task 3.4: MLflow Artifacts Logging Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Champion Run:** `exp3_champion_unfreeze20_lr5e5` (Run ID: `bb6d4446ea174baf8ea6f433772a4291`)  
**DagsHub Live Artifacts Tab:** [https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow/#/experiments/2/runs/bb6d4446ea174baf8ea6f433772a4291/artifacts](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow/#/experiments/2/runs/bb6d4446ea174baf8ea6f433772a4291/artifacts)  

---

## 1. Required Artifacts Attached to Champion Run

In compliance with Task 3.4, the following 5 distinct artifacts were logged directly to DagsHub remote MLflow storage:

| # | Artifact Name | Remote Artifact Path | Description |
| :---: | :--- | :--- | :--- |
| **1** | `confusion_matrix.png` | `evaluation/confusion_matrix.png` | Annotated $2 \times 2$ confusion matrix evaluated on held-out test set at $\tau^* = 0.23$. |
| **2** | `roc_curve.png` | `evaluation/roc_curve.png` | Full ROC curve plot with test ROC-AUC score (0.8401) and marked operating threshold. |
| **3** | `splits/` directory | `data/splits/` | Contains reproducible split manifests: `train_split.csv` (681), `val_split.csv` (146), and `test_split.csv` (146). |
| **4** | `requirements.txt` | `environment/requirements.txt` | Exact frozen Python dependency package list ensuring environment parity. |
| **5** | `model/` directory | `model/` | Fully serialized Keras Champion Model containing architecture, weights, and MLmodel signature metadata. |

---

## 2. Local Model Artifact Preservation
- Local champion model serialized to: `models/champion.keras`
- This ensures offline availability for Flask web serving (Task 4) and containerized deployment (Task 6) without requiring network calls to DagsHub for every inference request.
