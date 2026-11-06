# VisionGuard MLOps Assignment 2 — Progress Tracker

**Project:** VisionGuard — AI vs Real Image Classification and Deployment  
**Standard Naming:** `23L0710_A2_Ops`  
**Current Status:** Task 1.1 In Progress  
**Last Updated:** 2026-11-06  

---

## Overall Task Status Summary

| Task | Description | Marks | Status | Deliverables / Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **Task 1** | Data Understanding and Versioning | 7 | Completed (All 1.1-1.4) | 7/7 Marks Verified (Audit, EDA, DVC Toy & Applied) |
| **Task 2** | Model Development with TensorFlow/Keras | 14 | In progress (2.1 Done) | 2.1 Verified (tf.data Pipeline & Augmented Batch) |
| **Task 2 Bonus** | Explainable AI (Grad-CAM) | +1 | Not started | Grad-CAM heatmaps for 4 correct & 4 wrong predictions |
| **Task 3** | Experiment Tracking with MLflow & DagsHub | 12 | Not started | Local MLflow toy, DagsHub setup, 6 VisionGuard runs, Model Registry |
| **Task 4** | Flask Web Application | 12 | Not started | Inference module, Input validation, HTML/CSS UI, /health route |
| **Task 5** | Database Trail | 8 | Not started | Database model, /predict logging hook, /history UI table |
| **Task 6** | Containerization with Docker & CI | 14 | Not started | Docker warm-up, Web Dockerfile, Compose stack, CI workflow |
| **Task 7** | Cloud API Deployment | 15 | Not started | Cloud-hosted API, /health & /predict, Postman tests, Live URL |
| **Task 8** | Integrating Flask & Cloud API | 15 | Not started | Localhost Flask $\to$ Cloud API, DB logging, End-to-end evidence |
| **Report** | PDF Report Compilation | — | Not started | `<ROLL_NUMBER>_VisionGuard.pdf` with all required evidence |
| **Total** | **100 Marks (+ 1 Bonus)** | **100** | **0% Completed** | |

---

## Detailed Subtask Checklist

### Task 1: Data Understanding and Versioning (7 marks)
- [x] **1.1 Environment Setup (1 mark)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `.venv` (Python 3.10.19), `requirements.txt`, `.gitignore`, `src/verify_env.py`
  - Evidence collected: `evidence/task1/env_versions.txt` (Full environment, dependency, and GPU detection audit)
- [x] **1.2 Dataset Audit (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/audit.py`, `evidence/task1/audit_manifest.csv`, `evidence/task1/audit_summary.md`, `evidence/task1/audit_table.txt`
  - Evidence collected: Class counts, min/median/max dims and sizes table, corrupt & duplicate analysis
- [x] **1.3 Visual EDA (1.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/eda.py`, `evidence/task1/eda_sample_grid_comparison.png`, `evidence/task1/eda_dimensions_hist.png`, `evidence/task1/eda_filesize_hist.png`, `evidence/task1/eda_aspect_ratio_hist.png`
  - Evidence collected: 5 figures with full captions and 2-3 sentence interpretations in `evidence/task1/eda_interpretations.md`
- [x] **1.4 DVC Versioning (1.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/run_dvc_toy.sh`, `toy_dvc_demo/`, `src/run_dvc_applied.sh`, `archive.dvc`, `evidence/task1/dvc_explanation.md`
  - Evidence collected: `evidence/task1/dvc_toy_demo_log.txt`, `evidence/task1/dvc_applied_log.txt`, initial git commits, pointer file breakdown

---

### Task 2: Model Development with TensorFlow/Keras (14 marks)
- [x] **2.1 Input Pipeline (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/dataset.py`, `data/processed/train_split.csv`, `data/processed/val_split.csv`, `data/processed/test_split.csv`, `evidence/task2/augmented_batch.png`, `evidence/task2/input_pipeline_description.md`
  - Evidence collected: Pipeline architecture description, stratified 70/15/15 splits, augmented batch figure
- [ ] **2.2 Baseline CNN (3 marks)**
  - Status: Not started
  - Dependencies: 2.1
  - Evidence needed: Model summary screenshot, parameter count & output shape table, layer rationale
- [ ] **2.3 Transfer Learning (3 marks)**
  - Status: Not started
  - Dependencies: 2.1
  - Evidence needed: Two sets of training curves, list of trainable layers per stage
- [ ] **2.4 Training Discipline (1 mark)**
  - Status: Not started
  - Dependencies: 2.2, 2.3
  - Evidence needed: Loss & accuracy curves with best epoch marked, callbacks configured
- [ ] **2.5 Evaluation & Threshold Selection (2 marks)**
  - Status: Not started
  - Dependencies: 2.4
  - Evidence needed: Metric comparison table, confusion matrices, ROC curves, FP vs FN trade-off analysis
- [ ] **2.6 Error Analysis (1 mark)**
  - Status: Not started
  - Dependencies: 2.5
  - Evidence needed: Grid of 10 misclassified test images with failure hypotheses
- [ ] **2.7 Keras Tuner Tool Lab (2 marks)**
  - Status: Not started
  - Dependencies: 2.3
  - Evidence needed: Toy Fashion-MNIST results table, 8-trial VisionGuard tuning table, validation overfitting write-up
- [ ] **Task 2 Bonus: Grad-CAM (+1 mark)**
  - Status: Not started
  - Dependencies: 2.3, 2.5
  - Evidence needed: Heatmap overlays for 4 correct & 4 wrong predictions with commentary

---

### Task 3: Experiment Tracking with MLflow and DagsHub (12 marks)
- [ ] **3.1 MLflow Local Tool Lab (2 marks)**
  - Status: Not started
  - Dependencies: MLflow installation
  - Evidence needed: Screenshots of runs, comparison view, definitions of experiment/run/param/metric/artifact/tag
- [ ] **3.2 DagsHub Setup (2 marks)**
  - Status: Not started
  - Dependencies: 3.1, GitHub repo created, DagsHub account linked
  - Evidence needed: DagsHub project screenshot, remote toy run screenshot, proof of gitignored .env
- [ ] **3.3 Instrument VisionGuard Training (3 marks)**
  - Status: Not started
  - Dependencies: 2.4, 3.2
  - Evidence needed: DagsHub run page showing parameters, curves, tags for manual vs autologging
- [ ] **3.4 Artifact Logging (1.5 marks)**
  - Status: Not started
  - Dependencies: 2.5, 3.3
  - Evidence needed: Artifacts tab screenshot (confusion matrix, ROC, split lists, packages, model)
- [ ] **3.5 Experiment Comparison (2 marks)**
  - Status: Not started
  - Dependencies: 3.3
  - Evidence needed: Comparison screenshot of at least 6 runs, written justification of best model
- [ ] **3.6 Model Registry & Champion Loading (1.5 marks)**
  - Status: Not started
  - Dependencies: 3.5
  - Evidence needed: Registry screenshot with `champion` alias, terminal output of loading and classifying image

---

### Task 4: Flask Web Application (12 marks)
- [ ] **4.1 Inference Module (2 marks)**
  - Status: Not started
  - Dependencies: 3.6 (Champion model)
  - Evidence needed: Technical explanation, console test output for two known images
- [ ] **4.2 Routes & Validation (3 marks)**
  - Status: Not started
  - Dependencies: 4.1
  - Evidence needed: Screenshots of the three failure cases (wrong extension, file size, undecodable image)
- [ ] **4.3 HTML and CSS Frontend (2.5 marks)**
  - Status: Not started
  - Dependencies: 4.2
  - Evidence needed: Screenshots of home upload form, prediction result card, honest caveat banner
- [ ] **4.4 Configuration & Production Server (1 mark)**
  - Status: Not started
  - Dependencies: 4.2
  - Evidence needed: Health-check response screenshot, explanation of debug=False in production

---

### Task 5: Database Trail (8 marks)
- [ ] **5.1 Database Models & Views (8 marks)**
  - Status: Not started
  - Dependencies: 4.2
  - Evidence needed: Database schema, /predict logging verification, /history menu screenshot showing logged records

---

### Task 6: Containerization with Docker and Docker Compose (14 marks)
- [ ] **6.1 Docker Warm-Up (2 marks)**
  - Status: Not started
  - Dependencies: Docker Desktop running
  - Evidence needed: Screenshots of container run, port mapping, volume mount, logs, exec, definitions
- [ ] **6.2 Web App Dockerfile (3 marks)**
  - Status: Not started
  - Dependencies: 4.4, 5.1
  - Evidence needed: Dockerfile text, explanation of layer caching, non-root user verification
- [ ] **6.3 Compose Stack (4 marks)**
  - Status: Not started
  - Dependencies: 6.2, PostgreSQL
  - Evidence needed: Compose file, service health status, proof of database port closed from outside
- [ ] **6.4 Image Analysis & Optimization (1.5 marks)**
  - Status: Not started
  - Dependencies: 6.2
  - Evidence needed: Before vs after size comparison numbers and analytical conclusion
- [ ] **6.5 Operations & Troubleshooting (1.5 marks)**
  - Status: Not started
  - Dependencies: 6.3
  - Evidence needed: Service lifecycle commands, logs of intentional failure, diagnosis, and fix
- [ ] **6.6 GitHub Actions CI Tool Lab (2 marks)**
  - Status: Not started
  - Dependencies: Git repository on GitHub
  - Evidence needed: Red & green toy runs, passing VisionGuard CI run, README status badge

---

### Task 7: Cloud API Deployment (15 marks)
- [ ] **7.1 Cloud Service Setup (15 marks)**
  - Status: Not started
  - Dependencies: Cloud provider account, champion model
  - Evidence needed: Live public API URL, deployment step screenshots, Postman test evidence, external network test

---

### Task 8: Flask & Hosted API Integration (15 marks)
- [ ] **8.1 End-to-End System Demonstration (15 marks)**
  - Status: Not started
  - Dependencies: 7.1, 4.3, 5.1
  - Evidence needed: Localhost Flask calling Cloud API, returned predictions, database history updated, screenshots

---

### Final Submission Deliverables
- [ ] PDF Report: `23L0710_VisionGuard.pdf`
- [ ] GitHub Repository: `https://github.com/<username>/23L0710_A2_Ops`
- [ ] DagsHub Project: `https://dagshub.com/<username>/23L0710_A2_Ops`
- [ ] Live Cloud API URL
- [ ] Graded Commit Hash

