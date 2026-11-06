# VisionGuard MLOps Assignment 2 — Progress Tracker

**Project:** VisionGuard — AI vs Real Image Classification and Deployment  
**Standard Naming:** `23L0710_A2_Ops`  
**Current Status:** Task 6.3 Complete (62 / 100 Marks Verified)  
**Last Updated:** 2026-11-07  

---

## Overall Task Status Summary

| Task | Description | Marks | Status | Deliverables / Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **Task 1** | Data Understanding and Versioning | 7 | Completed (All 1.1-1.4) | 7/7 Marks Verified (Audit, EDA, DVC Toy & Applied) |
| **Task 2** | Model Development with TensorFlow/Keras | 14 | Completed (All 2.1-2.7) | 14/14 Marks Verified (Input Pipeline, Baseline CNN, Transfer Learning, Training Discipline, Evaluation & Threshold, Error Analysis, Keras Tuner) |
| **Task 2 Bonus** | Explainable AI (Grad-CAM) | +1 | Completed | +1 Bonus Verified (Heatmap overlays for 4 correct & 4 wrong predictions with commentary) |
| **Task 3** | Experiment Tracking with MLflow & DagsHub | 12 | Completed (All 3.1-3.6) | 12/12 Marks Verified (Local SQLite & Remote DagsHub MLflow, Manual vs Autolog, 6 Comparison Runs, Champion Artifacts, Model Registry @champion & Loader) |
| **Task 4** | Flask Web Application | 12 | Completed (All 4.1-4.4) | 12/12 Marks Verified (Singleton Inference Engine, Preprocessing Parity, Defensive Routes & 3 Failure Screenshots, Responsive UI with Honest Caveat, /health endpoint & debug=False report) |
| **Task 5** | Database Trail | 8 | Completed | 8/8 Marks Verified (SQLAlchemy ORM Model, Schema DDL, Automatic Persistence on /predict, /history UI Screenshot, Database Records Dump) |
| **Task 6** | Containerization with Docker & CI | 14 | In progress (6.1, 6.2 & 6.3 Done) | 6.1-6.3 Verified (Warm-Up Lab, Production Dockerfile, Compose Stack web+db, Proof of Closed Port 5432, Service Health, Model Delivery Strategy) |
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
- [x] **2.2 Baseline CNN (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/models/baseline.py`, `src/models/__init__.py`, `evidence/task2/baseline_summary.txt`, `evidence/task2/baseline_architecture.md`
  - Evidence collected: Model summary text, parameter count and output shape table, layer-by-layer justification
- [x] **2.3 Transfer Learning (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/models/transfer.py`, `models/mobilenetv2_finetuned.keras`, `evidence/task2/transfer_layers_phase1.txt`, `evidence/task2/transfer_layers_phase2.txt`, `evidence/task2/transfer_learning_curves.png`, `evidence/task2/transfer_learning_report.md`
  - Evidence collected: Two sets of training curves (Phase 1 & Phase 2), trainable layer listings per stage, preprocessing justification
- [x] **2.4 Training Discipline (1 mark)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/train.py`, `models/best_checkpoint_model.keras`, `evidence/task2/training_discipline_curves.png`, `evidence/task2/training_discipline_report.md`
  - Evidence collected: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau execution, Best Epoch identified (Epoch 3, val_loss=0.5884), curves with best epoch marked
- [x] **2.5 Evaluation & Threshold Selection (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-06
  - Deliverables: `src/evaluate.py`, `models/baseline_cnn.keras`, `evidence/task2/evaluation_metrics_table.md`, `evidence/task2/evaluation_report.md`, `evidence/task2/roc_curves_comparison.png`, `evidence/task2/confusion_matrices.png`, `evidence/task2/threshold_tuning_val.png`
  - Evidence collected: Single test set evaluation (both models), validation threshold sweep (tau*=0.23), ROC curves, confusion matrices, written FP vs FN risk analysis
- [x] **2.6 Error Analysis (1 mark)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/error_analysis.py`, `evidence/task2/misclassified_grid.png`, `evidence/task2/error_analysis_report.md`
  - Evidence collected: Analysis of 39 test set misclassifications (13 FP, 26 FN), 2x5 visual grid of 10 sampled failure cases, individual failure hypotheses, and 4 systemic error themes with mitigation strategies
- [x] **2.7 Keras Tuner Tool Lab (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/tuner.py`, `evidence/task2/tuner_toy_table.md`, `evidence/task2/tuner_summary_table.md`, `evidence/task2/tuner_results.md`
  - Evidence collected: Fashion-MNIST toy lab results table (5 trials), VisionGuard MobileNetV2 applied tuning table (8 trials ranked), champion parameter identification (LR=5e-5, Dropout=0.3, Unfrozen=20), and written discussion analyzing hyperparameter sensitivity and validation set overfitting
- [x] **Task 2 Bonus: Grad-CAM (+1 mark)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/models/gradcam.py`, `evidence/task2/gradcam_grid.png`, `evidence/task2/gradcam_report.md`
  - Evidence collected: Grad-CAM heatmap overlays for 4 correct predictions (2 TP, 2 TN) and 4 wrong predictions (2 FP, 2 FN), mathematical formulation of Grad-CAM, and case-by-case diagnostic attention commentary

---

### Task 3: Experiment Tracking with MLflow and DagsHub (12 marks)
- [x] **3.1 MLflow Local Tool Lab (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/mlflow_toy.py`, `evidence/task3/toy_run_comparison.md`, `evidence/task3/mlflow_toy_definitions.md`, `evidence/task3/toy_run1_curves.png`, `evidence/task3/toy_run2_curves.png`, `evidence/task3/toy_comparison_curves.png`
  - Evidence collected: Two local MLflow runs (`run_1_lr_0.001` vs `run_2_lr_0.010`), logged hyperparameters, dynamic per-epoch metrics, serialized Keras models, comparative curves, and student-authored definitions for Experiment, Run, Parameter, Metric, Artifact, and Tag
- [x] **3.2 DagsHub Setup (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/dagshub_toy.py`, `.env.example`, `evidence/task3/dagshub_setup_report.md`, `evidence/task3/dagshub_toy_curves.png`
  - Evidence collected: Live DagsHub project (`https://dagshub.com/shoaib-asiff/23L0710_A2_Ops`), live remote toy run on DagsHub MLflow server (Run ID: `b474f87c071f49ad8df1f0ad26ae19e3`), and formal verification that `.env` is strictly excluded in `.gitignore`
- [x] **3.3 Instrument VisionGuard Training (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/instrumentation_experiment.py`, `evidence/task3/manual_run_curves.png`, `evidence/task3/autolog_run_curves.png`, `evidence/task3/instrumentation_comparison_report.md`
  - Evidence collected: Two live runs on DagsHub MLflow server (`visionguard_manual_callback_run` vs `visionguard_autolog_run`), logged hyperparameters, dynamic per-epoch metrics, final test metrics (Accuracy, Precision, Recall, F1, ROC-AUC), provenance tags (Git commit, DVC data MD5, seed 42), and comprehensive technical write-up analyzing what each approach captures vs misses
- [x] **3.4 Artifact Logging (1.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `evidence/task3/run_artifacts/confusion_matrix.png`, `evidence/task3/run_artifacts/roc_curve.png`, `evidence/task3/run_artifacts/splits/`, `evidence/task3/run_artifacts/requirements.txt`, `evidence/task3/artifacts_logging_report.md`
  - Evidence collected: All 5 required artifacts attached to Champion run on DagsHub MLflow remote server (`confusion_matrix.png`, `roc_curve.png`, `splits/`, `requirements.txt`, `model/`) and saved locally to `models/champion.keras`
- [x] **3.5 Experiment Comparison (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/experiment_comparison.py`, `evidence/task3/experiment_comparison_table.md`, `evidence/task3/experiment_comparison_chart.png`, `evidence/task3/experiment_comparison_report.md`
  - Evidence collected: 6 distinct experimental runs executed and logged on DagsHub MLflow varying learning rate, backbone unfreezing depth, augmentation, dropout, and architecture; comparative matrix table and comparative bar chart; written justification selecting `exp3_champion_unfreeze20_lr5e5` based on ROC-AUC, F1, and stability
- [x] **3.6 Model Registry & Champion Loading (1.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/registry_loader.py`, `evidence/task3/registry_loader_output.txt`, `evidence/task3/model_registry_report.md`
  - Evidence collected: DagsHub Model Registry `VisionGuard_Classifier` configured with Version 2 tagged with alias `@champion`; standalone inference script `src/registry_loader.py` dynamically resolving and loading remote model strictly by alias over HTTPS, executing inference with complete preprocessing parity, and documenting decoupled production serving architecture

---

### Task 4: Flask Web Application (12 marks)
- [x] **4.1 Inference Module (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `webapp/inference.py`, `src/test_inference_module.py`, `evidence/task4/task4_1_console_output.txt`, `evidence/task4/task4_1_inference_report.md`
  - Evidence collected: Decoupled singleton inference engine loading `models/champion.keras` once at startup; full preprocessing parity (RGB conversion, aspect-ratio preserving padding to 224x224, float32 scaling, zero disk writes); console evaluation on two known ground-truth images (AI-Generated: 50.00% PASS, Real: 83.97% PASS); 35.6x throughput speedup benchmark (267.53ms in-memory vs 9529.78ms per-request reload); complete write-up on per-request loading unacceptability and training-serving preprocessing parity
- [x] **4.2 Routes & Validation (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `webapp/app.py`, `src/test_webapp_routes.py`, `evidence/task4/failure_case1_wrong_extension.png`, `evidence/task4/failure_case2_size_limit.png`, `evidence/task4/failure_case3_corrupt_image.png`, `evidence/task4/task4_2_routes_validation_report.md`
  - Evidence collected: Defensive routes implemented (`/`, `/predict`, `/history`, `/health`); filename sanitization via `secure_filename`; zero disk writes (in-memory `io.BytesIO` + base64 Data URI); three mandatory failure cases validated and screenshotted via headless Chrome (Wrong Extension $\to$ 400, Size Limit $>5$MB $\to$ 413, Undecodable/Corrupt Bytes $\to$ 400)
- [x] **4.3 HTML and CSS Frontend (2.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `webapp/templates/base.html`, `webapp/templates/index.html`, `webapp/templates/result.html`, `webapp/templates/history.html`, `webapp/templates/error.html`, `webapp/static/css/style.css`, `evidence/task4/frontend_home_upload.png`, `evidence/task4/frontend_prediction_result.png`, `evidence/task4/frontend_audit_history.png`, `evidence/task4/task4_3_frontend_report.md`
  - Evidence collected: Responsive card-based UI with drag-and-drop file upload, live feedback, classification verdict badges, animated confidence bar, detailed metadata table, and prominently placed mandatory Honest Engineering Caveat Banner
- [x] **4.4 Configuration & Production Server (1 mark)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `evidence/task4/health_check.json`, `evidence/task4/health_check_response.png`, `evidence/task4/task4_4_production_config_report.md`
  - Evidence collected: Environment-driven settings (`MODEL_PATH`, `DECISION_THRESHOLD`, `INFERENCE_MODE`, `MAX_CONTENT_LENGTH`); dedicated `/health` route returning JSON diagnostics; production execution with `debug=False` and comprehensive write-up on arbitrary code execution and DoS risks of debug mode in production

---

### Task 5: Database Trail (8 marks)
- [x] **5.1 Database Models & Views (8 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/database.py`, `src/test_database.py`, `evidence/task5/schema.sql`, `evidence/task5/database_dump.txt`, `evidence/task5/history_database_ui.png`, `evidence/task5/database_report.md`
  - Evidence collected: SQLAlchemy 2.0 ORM schema `prediction_records` with indexed `timestamp` and `prediction`; dynamic SQLite/PostgreSQL URL switching; automatic persistence hook on `POST /predict`; chronological queryable history on `GET /history`; formatted table dump of 5 logged records; headless Chrome screenshot of populated Audit Trail UI; comprehensive write-up on indexing, forensic auditability, and Docker PostgreSQL migration plan

---

### Task 6: Containerization with Docker and Docker Compose (14 marks)
- [x] **6.1 Docker Warm-Up (2 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/docker_warmup.py`, `evidence/task6/docker_warmup_log.txt`, `evidence/task6/docker_warmup_terminal.png`, `evidence/task6/docker_warmup_report.md`
  - Evidence collected: 10 lifecycle actions verified on Docker Desktop 4.98.0 (Engine 29.9.2): pulled `nginx:alpine`, detached run with port mapping (`-p 8085:80`) and host volume mount (`-v`), curl HTTP 200 OK proof, container logs, `docker exec` inspecting volume note and processes, `docker ps -a`, `docker images`, clean stop, removal, and rmi; terminal screenshot captured; 5 core student definitions authored in own words (Image, Container, Layer, Volume, Port Mapping)
- [x] **6.2 Web App Dockerfile (3 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `Dockerfile`, `.dockerignore`, `requirements-docker.txt`, `src/test_dockerfile.py`, `evidence/task6/dockerfile_user_check.txt`, `evidence/task6/dockerfile_report.md`
  - Evidence collected: Minimal `python:3.10-slim` base image; layer caching optimization (dependencies copied & installed before source code); non-root user `appuser:appgroup` verified via `whoami` and `id`; native Docker `HEALTHCHECK` targeting `/health` (Status: healthy, 200 OK); `.dockerignore` excluding `.env`, `.git`, caches and training data; technical justification for `tensorflow-cpu` choice saving >1.2GB over GPU TensorFlow
- [x] **6.3 Compose Stack (4 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `docker-compose.yml`, `.env.example`, `src/test_compose_stack.py`, `evidence/task6/compose_status.png`, `evidence/task6/compose_output.txt`, `evidence/task6/compose_report.md`
  - Evidence collected: Multi-service Compose stack (`web` + `db` PostgreSQL 15); internal network `visionguard-net`; named volume `postgres_data`; startup readiness ordering via `depends_on: db: condition: service_healthy`; mandatory check proving port 5432 is strictly closed and refused from host socket probes; live end-to-end classification ingestion persisting records into containerized PostgreSQL; terminal screenshot captured; full write-up on model delivery strategy and secrets management
- [x] **6.4 Image Analysis & Optimization (1.5 marks)**
  - Status: Completed — verified
  - Completed on: 2026-11-07
  - Deliverables: `src/image_analysis.py`, `evidence/task6/image_analysis_layers.txt`, `evidence/task6/image_analysis_report.md`
  - Evidence collected: Layer-by-layer breakdown via `docker history visionguard-web:latest`; two mandatory rubric size numbers reported (Before Optimization: 3.84 GB, After Optimization: 1.94 GB, Net Reduction: 1.90 GB / 49.5%); written technical justification detailing base image selection (`python:3.10-slim`), CPU TensorFlow (`tensorflow-cpu`), pip cache pruning (`--no-cache-dir`), apt metadata cleanup, and production impact on cold-start latency and attack surface
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

