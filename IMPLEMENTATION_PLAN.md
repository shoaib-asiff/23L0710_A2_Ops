# Implementation Plan: VisionGuard MLOps Assignment 2

**Project:** VisionGuard — AI vs Real Image Classification and Deployment  
**Standard Naming Convention:** `23L0710_A2_Ops`  
**Report Output:** `23L0710_VisionGuard.pdf`  
**Total Marks:** 100 Marks (+ 1 Bonus Mark)  
**Execution Mode:** Sequential, Task-by-Task, Evidence-Driven  

---

## 1. System & Environment Baseline Inspection

Based on automated environment audits performed on the current workstation:
- **Operating System:** macOS 15.8.1 (x86_64 Darwin Kernel 24.6.0)
- **CPU:** Intel Core i7-8559U CPU @ 2.70GHz (4 Cores, 8 Threads)
- **GPU Availability:** None (Apple Mac Intel hardware lacks CUDA support; TensorFlow will execute on CPU using optimized oneDNN operations. This will be explicitly recorded for Task 1.1).
- **Available Python Versions:**
  - System default: Python 3.13.7 (Incompatible with current TensorFlow wheels)
  - Pre-installed Homebrew: **Python 3.10.19** located at `/usr/local/Cellar/python@3.10/3.10.19_1/bin/python3.10` (**Recommended & Selected**)
- **Disk Storage:** 466 GiB total, **13 GiB available** (98% capacity utilized). *Careful resource management required: no duplicate dataset copies, prune dangling Docker images, manage pip caches.*
- **Git:** Version 2.50.1 (Apple Git-155) installed; workspace is currently not initialized as a git repository.
- **Docker:** Docker version 29.4.2 & Docker Compose v5.1.3 installed (Desktop app needs to be launched when Docker tasks begin).
- **Dataset Structure:** `./archive`
  - Two class directories: `archive/AiArtData/AiArtData` (~540 images) and `archive/RealArt/RealArt` (~436 images).
  - Total files: 978 files (972 valid image formats: JPG, PNG, JPEG, WEBP, GIF; 6 system files).
  - Total size: ~505 MB.
  - Raw images will remain strictly untouched in `./archive`.

---

## 2. Technical Stack & Architectural Decisions

### 2.1 Python Environment & Dependencies
- **Python Version:** 3.10.19 via dedicated virtual environment (`.venv`).
- **Core ML Framework:** TensorFlow 2.15.0 / 2.16.x (CPU-optimized for macOS x86_64 and Linux Docker containers).
- **Tracking & Versioning:** DVC (with local / DagsHub storage remote), MLflow (integrated with DagsHub tracking URI).
- **Hyperparameter Optimization:** Keras Tuner.
- **Serving & Web Application:** Flask, Gunicorn (for production WSGI in Docker/Cloud), Werkzeug, Pillow, NumPy, Requests.
- **Database:** PostgreSQL (`psycopg2-binary`, `SQLAlchemy`) for Docker Compose and production; SQLite option for lightweight offline fallback.
- **Containerization:** Docker multi-stage / slim build (`python:3.10-slim`), Docker Compose v2.
- **CI/CD:** GitHub Actions (matrix testing, flake8/black linting, pytest unit tests, Docker build verification).

### 2.2 Reconciling Ambiguities & Critical Assignment Requirements
1. **Local Flask vs Cloud Hosted API (Tasks 4, 6, 7, and 8):**
   - *Rubric Requirement:* Task 4 requires a local Flask web application with a standalone `inference.py` loading the model once at startup. Task 6 requires containerizing this app and database using Docker Compose. Task 7 requires deploying an inference API on the cloud. Task 8 requires Flask running on localhost to send inference requests to the cloud-hosted API and display the result + DB history.
   - *Architectural Solution:* We will design the Flask application with an environment-driven inference engine:
     - `INFERENCE_MODE=local`: loads the champion model directly from `models/champion.keras` (used in Task 4, Task 5, Task 6).
     - `INFERENCE_MODE=remote`: forwards the uploaded image payload via HTTP POST to `REMOTE_INFERENCE_API_URL` (used in Task 7 and Task 8).
     - This guarantees that Task 4, Task 6, Task 7, and Task 8 are all 100% satisfied without rewriting or breaking existing code.
2. **Database Choice for Task 5 and Task 6.3:**
   - Task 6.3 explicitly requires two services in `docker-compose.yml` (`web` and `database`), where services communicate over the internal Compose network by service name, the database uses a named volume, has a readiness healthcheck before `web` starts, and publishes no ports externally.
   - PostgreSQL (`postgres:15-alpine`) perfectly matches these requirements. We will use SQLAlchemy to model the inference history table, allowing seamless operation across PostgreSQL in Docker Compose and SQLite for isolated local unit testing.
3. **DagsHub Model Registry & Champion Alias (Task 3.6):**
   - We will implement model registration in MLflow. We will verify DagsHub's model registry alias support (`models:/<model_name>@champion`). If DagsHub MLflow server currently supports stages (`Production`) or version tags rather than aliases, we will document the exact REST API response, apply the alias or stage, and verify model loading via script.
4. **Dataset Directory Structure:**
   - The archive contains nested subdirectories: `archive/AiArtData/AiArtData` and `archive/RealArt/RealArt`. Our audit script and tf.data loader will dynamically locate image files while preserving class labels.

---

## 3. Directory Layout & Repository Structure

```text
23L0710_A2_Ops/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Task 6.6 VisionGuard CI pipeline
│       └── toy_ci.yml             # Task 6.6 Toy CI matrix workflow
├── archive/                       # Original raw dataset (gitignored, DVC tracked)
├── data/
│   ├── processed/                 # Train / val / test split metadata & manifests
│   └── toy/                       # Task 1.4 DVC toy demonstration data
├── src/
│   ├── __init__.py
│   ├── audit.py                   # Task 1.2 Full dataset audit & hash check
│   ├── eda.py                     # Task 1.3 Visual EDA & figure generator
│   ├── dataset.py                 # Task 2.1 tf.data input pipeline & augmentations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py            # Task 2.2 Custom CNN architecture
│   │   ├── transfer.py            # Task 2.3 MobileNetV2 / EfficientNet transfer learning
│   │   └── gradcam.py             # Task 2 Bonus Grad-CAM visualizer
│   ├── train.py                   # Task 2.4 & Task 3.3 Training with callbacks & MLflow
│   ├── evaluate.py                # Task 2.5 Metrics, threshold tuning, ROC & confusion matrix
│   ├── error_analysis.py          # Task 2.6 Misclassification analysis (10 samples)
│   ├── tuner.py                   # Task 2.7 Keras Tuner script (toy & VisionGuard)
│   ├── registry_loader.py         # Task 3.6 Model registry champion loader & tester
│   └── database.py                # Task 5 Database models and query interface
├── webapp/
│   ├── __init__.py
│   ├── app.py                     # Task 4 & 8 Flask application routes
│   ├── inference.py               # Task 4.1 Inference engine (dual local/remote mode)
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html             # Upload form & caveat banner
│   │   ├── result.html            # Prediction verdict & confidence
│   │   ├── history.html           # Database query table
│   │   └── error.html             # Friendly error page
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── api/                           # Task 7 Standalone Cloud API service
│   ├── app.py                     # Cloud-hosted inference endpoint
│   ├── requirements.txt
│   └── Dockerfile
├── docker/
│   ├── Dockerfile.web             # Task 6.2 Slim, non-root, layer-cached Dockerfile
│   └── Dockerfile.web.optimized   # Task 6.4 Optimized comparison Dockerfile
├── tests/
│   ├── test_pipeline.py
│   ├── test_inference.py
│   └── test_app.py
├── evidence/                      # Organized evidence repository for report
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   ├── task5/
│   ├── task6/
│   ├── task7/
│   └── task8/
├── docker-compose.yml             # Task 6.3 Compose stack (web + postgres db)
├── .dockerignore
├── .gitignore
├── .dvcignore
├── .env.example
├── requirements.txt
├── requirements-cpu.txt           # Lightweight requirements for Docker serving
├── IMPLEMENTATION_PLAN.md
├── PROGRESS.md
└── README.md
```

---

## 4. Detailed Task-by-Task Implementation Plan

### TASK 1: Data Understanding and Versioning (7 marks)
- **Subtask 1.1: Environment Setup (1 mark)**
  - Create isolated virtual environment using Python 3.10.19 (`.venv`).
  - Install dependencies: `tensorflow==2.15.0`, `pillow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `dvc`, `mlflow`.
  - Execute automated environment verification script checking package versions, CPU features, and `tf.config.list_physical_devices('GPU')`.
  - *Evidence:* Terminal screenshot showing Python, OS, pip packages, and GPU detection output (`[]` on macOS Intel).
- **Subtask 1.2: Dataset Audit (3 marks)**
  - Write `src/audit.py`: walks `archive`, reads every file, extracts width, height, color mode, format, file size, and computes SHA-256 hash.
  - Detect corrupt/unreadable files (using PIL `verify()` and decoding check) and duplicate files (duplicate SHA-256 hashes).
  - Generate a structured summary table: class counts, min/median/max dimensions and file sizes, counts of corrupt and duplicate files.
  - *Evidence:* Formatted markdown/CSV audit table and console output.
- **Subtask 1.3: Visual EDA (1.5 marks)**
  - Write `src/eda.py`: generates a $3 \times 3$ sample image grid per class (real vs AI-generated).
  - Generate distribution histograms: image width, height, aspect ratio, and file size per class.
  - Save all figures to `evidence/task1/` with descriptive captions and 2–3 sentence analytical interpretations.
  - *Evidence:* Saved visual figures (`sample_grid.png`, `dimension_distribution.png`, `filesize_distribution.png`) and written interpretations.
- **Subtask 1.4: DVC Tracking & Toy Demonstration (1.5 marks)**
  - **Toy Lab:** Create `data/toy/`, create `data_v1.csv`, initialize Git and DVC, track CSV with DVC, commit v1 to Git. Update to `data_v2.csv`, commit v2. Switch Git branches/tags to restore v1. Configure remote storage (local/DagsHub), push data, delete local file, and execute `dvc pull` to restore.
  - **Applied DVC:** Track `archive/` or curated dataset via DVC (`dvc add archive`), configure DagsHub storage remote, push to remote, and commit `.dvc` pointer files to Git.
  - Explain the internals of the small `.dvc` pointer file (hash, size, path) and why large binary datasets must never be committed directly to Git.
  - *Evidence:* Terminal logs and screenshots of commits, remote push/pull, and pointer file contents.

---

### TASK 2: Model Development with TensorFlow/Keras (14 marks)
- **Subtask 2.1: Input Pipeline (2 marks)**
  - Write `src/dataset.py`: partition dataset into 70% Train, 15% Validation, 15% Test using stratified splitting with fixed random seed (42). Save split file lists to `data/processed/`.
  - Build `tf.data` pipeline: `from_tensor_slices` $\to$ decode $\to$ resize ($224 \times 224$ with aspect ratio handling via padding/smart resize) $\to$ batch (batch size 32) $\to$ prefetch (`tf.data.AUTOTUNE`).
  - Add data augmentation pipeline applied exclusively to the training set (RandomFlip, RandomRotation, RandomZoom, RandomContrast).
  - *Evidence:* Written architectural description and a generated plot of an augmented batch (`evidence/task2/augmented_batch.png`).
- **Subtask 2.2: Baseline CNN from Scratch (3 marks)**
  - Write `src/models/baseline.py`: custom CNN with 3–4 convolutional blocks (Conv2D $\to$ BatchNormalization $\to$ MaxPooling2D), Dropout (0.3–0.5), GlobalAveragePooling2D, Dense(128, relu), Dense(1, sigmoid).
  - Compile with `BinaryCrossentropy`, `Adam(learning_rate=1e-3)`, and metrics (`accuracy`, `AUC`, `Precision`, `Recall`).
  - Document layer-by-layer justification, parameter counts, and design rationale.
  - *Evidence:* Model summary terminal capture and layer parameter table.
- **Subtask 2.3: Pretrained Transfer Learning (3 marks)**
  - Write `src/models/transfer.py`: Load `MobileNetV2` (or `EfficientNetB0`) pretrained on ImageNet without top classifier.
  - Apply exact model-specific preprocessing (`mobilenet_v2.preprocess_input`).
  - **Phase 1 (Feature Extraction):** Freeze backbone (`base_model.trainable = False`), add custom classification head, train top layers.
  - **Phase 2 (Fine-Tuning):** Unfreeze the top block of the backbone (e.g. last 20–30 layers), compile with small learning rate (`1e-5`), train for fine-tuning.
  - *Evidence:* Trainable layers listing per stage and two sets of training curves (`loss` and `accuracy` across both phases).
- **Subtask 2.4: Training Discipline (1 mark)**
  - Implement training discipline in `src/train.py`:
    - Seed setting (`random.seed`, `np.random.seed`, `tf.random.set_seed`).
    - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
    - `ModelCheckpoint(filepath='models/best_model.keras', monitor='val_loss', save_best_only=True)`.
    - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)`.
  - Plot learning curves marking the exact best epoch.
  - *Evidence:* Saved loss/accuracy curves with best epoch marker and terminal epoch summary.
- **Subtask 2.5: Test Set Evaluation & Threshold Selection (2 marks)**
  - Write `src/evaluate.py`: Evaluate models on the held-out test set **only once**.
  - Compute Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and ROC Curve.
  - Threshold Optimization: Sweep decision thresholds on the **validation set** to select the optimal operating point.
  - Business impact analysis: Thoroughly discuss and justify whether False Positives (labeling a real photo as AI fake) or False Negatives (labeling an AI fake as real) represent the worse risk for a trust & safety verification product.
  - *Evidence:* Comprehensive metric comparison table, ROC curves, confusion matrices, and threshold trade-off analysis.
- **Subtask 2.6: Error Analysis (1 mark)**
  - Write `src/error_analysis.py`: Identify misclassified test images.
  - Extract 10 misclassified samples (both False Positives and False Negatives).
  - Generate a formatted visual grid and formulate concrete hypotheses for each failure (e.g., high-frequency digital textures, heavy blur, stylistic painterly art, lighting artifacts).
  - *Evidence:* $2 \times 5$ grid of misclassified images with individual diagnostic captions.
- **Subtask 2.7: Keras Tuner Tool Lab (2 marks)**
  - Write `src/tuner.py`:
    - **Toy Lab:** Tune a small dense/conv network on Fashion-MNIST over learning rate ($1e-2, 1e-3, 1e-4$) and dense units ($32, 64, 128$) using `RandomSearch` or `Hyperband`.
    - **VisionGuard Tuning:** Run at least 8 trials searching learning rate, dropout rate, and number of unfrozen layers on the transfer learning model.
    - Discuss hyperparameter sensitivity (which setting impacted performance most) and explain the risk of overfitting the validation set during extensive hyperparameter searches.
  - *Evidence:* Toy tuner summary table, 8-trial VisionGuard trial results, and validation overfitting discussion.
- **Bonus Task: Explainable AI with Grad-CAM (+1 mark)**
  - Write `src/models/gradcam.py`: Compute Grad-CAM heatmaps from the final convolutional layer of the fine-tuned model.
  - Generate visual overlays for 4 correct predictions (2 real, 2 AI) and 4 misclassified predictions (2 FP, 2 FN).
  - Provide analytical commentary on model attention patterns.

---

### TASK 3: Experiment Tracking with MLflow and DagsHub (12 marks)
- **Subtask 3.1: Local MLflow Tool Lab (2 marks)**
  - Train a tiny model on built-in dataset; log hyperparameters (`learning_rate`, `batch_size`), per-epoch metrics (`loss`, `accuracy`), an artifact plot, and the saved model.
  - Execute two runs with different learning rates (`lr=0.01` vs `lr=0.001`).
  - Open local MLflow UI (`mlflow ui`), compare the two runs.
  - Provide student-authored definitions in own words for: *Experiment*, *Run*, *Parameter*, *Metric*, *Artifact*, and *Tag*.
  - *Evidence:* Screenshots of run comparison table, metric curves, and definitions.
- **Subtask 3.2: DagsHub Setup & Integration (2 marks)**
  - Create DagsHub project linked to the GitHub repository: `23L0710_A2_Ops`.
  - Securely retrieve DagsHub MLflow tracking URI and authentication token.
  - Store credentials in local `.env` (gitignored) and load via `python-dotenv`.
  - Re-run the toy experiment logging directly to DagsHub remote MLflow server.
  - *Evidence:* Screenshot of DagsHub MLflow dashboard showing the toy runs, and verification that `.env` is absent from git tracking.
- **Subtask 3.3: Instrument VisionGuard Training (3 marks)**
  - Implement dual logging approaches:
    1. **Manual logging:** Custom Keras Callback logging per-epoch metrics, validation metrics, test metrics, hyperparameters, Git commit hash (`git rev-parse HEAD`), DVC data version hash, and random seed.
    2. **Autologging:** Using `mlflow.tensorflow.autolog()`.
  - Analyze and contrast what manual logging captures vs what autologging captures or misses.
  - *Evidence:* Run overview page on DagsHub showing parameters, metric plots, tags, and comparison write-up.
- **Subtask 3.4: Artifact Logging (1.5 marks)**
  - Log complete evaluation artifacts to the primary MLflow run:
    - Confusion Matrix plot (`confusion_matrix.png`)
    - ROC Curve plot (`roc_curve.png`)
    - Split list CSVs (`train_split.csv`, `val_split.csv`, `test_split.csv`)
    - Environment package list (`requirements.txt` / `pip freeze`)
    - Trained Keras model artifact (`model/`)
  - *Evidence:* Screenshot of MLflow Run Artifacts tab showing all attached files.
- **Subtask 3.5: Experiment Comparison & Model Selection (2 marks)**
  - Execute and log **at least 6 meaningfully distinct VisionGuard training runs**:
    1. *Run 1:* Baseline CNN (scratch)
    2. *Run 2:* MobileNetV2 Frozen Backbone ($lr=1e-3$)
    3. *Run 3:* MobileNetV2 Fine-Tuned ($lr=1e-5$, unfreeze 20 layers)
    4. *Run 4:* MobileNetV2 Fine-Tuned + Heavy Data Augmentation
    5. *Run 5:* MobileNetV2 Fine-Tuned with Different Image Size ($160 \times 160$ vs $224 \times 224$)
    6. *Run 6:* EfficientNetB0 (Alternative Backbone comparison)
  - Generate a structured comparative matrix and chart; provide reasoned justification for selecting the champion model.
  - *Evidence:* Screenshot of DagsHub experiment comparison table and written model selection rationale.
- **Subtask 3.6: Model Registry & Champion Loading (1.5 marks)**
  - Register the top two models in MLflow Model Registry (`VisionGuard_Classifier`).
  - Assign the alias `champion` (or stage `Production`) to the winning version.
  - Write `src/registry_loader.py`: standalone script that connects to the registry, loads the model using the `champion` alias, and runs inference on a test image.
  - Document the operational strategy: explain why the serving application must **never** download the model from DagsHub on every incoming HTTP request, and detail how the champion model is packaged or mounted into the serving container.
  - *Evidence:* Screenshot of DagsHub Model Registry and terminal output of `registry_loader.py`.

---

### TASK 4: Flask Web Application (12 marks)
- **Subtask 4.1: Standalone Inference Module (2 marks)**
  - Write `webapp/inference.py`: encapsulates model loading once at process initialization (`load_model_at_startup()`), preprocessing identical to training pipeline (`preprocess_image()`), and prediction formatting.
  - Write technical justification: explain why per-request model loading causes catastrophic latency, memory leaks, and CPU thrashing, and why training/serving preprocessing parity is strictly mandatory.
  - *Evidence:* Terminal console output showing model startup load time and inference results on two known test images (one Real, one AI).
- **Subtask 4.2: Routes, Validation, and Security (3 marks)**
  - Write `webapp/app.py`: implement routes `/`, `/predict`, `/history`, and `/health`.
  - Comprehensive input validation:
    - Allowed file extensions (`.png`, `.jpg`, `.jpeg`, `.webp`).
    - File size limit enforcement (e.g. 5 MB max payload) returning HTTP 413.
    - Pillow decoding verification (reject corrupted files, disguised binaries, or scripts named `.jpg`).
    - Filename sanitization using `werkzeug.utils.secure_filename`.
    - Zero disk persistence: process image directly from memory buffer (`BytesIO`), leaving no temporary files on disk.
    - Custom friendly error pages (`400`, `404`, `413`, `500`).
  - *Evidence:* Browser screenshots demonstrating error handling for: (1) invalid extension (.txt/pdf), (2) file exceeding size limit, and (3) corrupt/unreadable image file.
- **Subtask 4.3: Frontend UI with HTML & CSS (2.5 marks)**
  - Build clean, modern responsive UI (`webapp/templates/` and `webapp/static/css/style.css`):
    - Clean drag-and-drop / file picker upload form.
    - Clear prediction result card displaying verdict (AI-Generated vs Real), confidence percentage meter, and prominent honest caveat banner ("*Automated classification is probabilistic; inspect subtle artifacts before relying on results*").
    - Navigation bar with one-click access to the History log.
  - *Evidence:* Full-page browser screenshots of the home upload view and prediction result view.
- **Subtask 4.4: Configuration & Production Readiness (1 mark)**
  - Manage configuration strictly via environment variables: `MODEL_PATH`, `DECISION_THRESHOLD`, `DATABASE_URL`, `FLASK_ENV`.
  - Expose `/health` endpoint returning JSON (`{"status": "healthy", "model_loaded": true, "database_connected": true, "timestamp": ...}`).
  - Run server with `debug=False` and explain risks of enabling debug mode in production (arbitrary code execution via interactive PIN debugger, secret exposure).
  - *Evidence:* Screenshot of `/health` JSON response in browser/curl and explanation.

---

### TASK 5: Database Trail (8 marks)
- **Database Engine Selection & Justification:**
  - Selected: **PostgreSQL** for Docker Compose and Cloud deployment (using SQLite as local offline fallback).
  - *Justification:* Task 6.3 mandates a multi-container Compose stack where the web service reaches the database service via Compose DNS over an internal network, waits for database readiness via healthcheck, and uses a named persistent volume without exposing ports externally. SQLite is an embedded single-file library and cannot run as an independent network service. PostgreSQL satisfies all criteria of Tasks 5 and 6.3.
- **Schema & Implementation:**
  - Write `src/database.py`: Define `InferenceRecord` model using SQLAlchemy:
    - `id` (Primary Key, Integer / UUID)
    - `timestamp` (DateTime, UTC default)
    - `filename` (Sanitized string)
    - `predicted_label` (String: "AI-Generated" or "Real")
    - `confidence` (Float)
    - `raw_score` (Float)
    - `threshold_used` (Float)
    - `model_version` (String, e.g. "MobileNetV2-v1.0")
    - `client_ip` (String, optional metadata)
  - Attach database logging hook to the `/predict` route: every successful classification automatically commits an audit record.
  - Add `/history` route and template displaying historical queries in a paginated, filterable table.
  - *Evidence:* Screenshot of the History UI table populated with real query logs, and database query output.

---

### TASK 6: Containerization with Docker, Docker Compose, and CI (14 marks)
- **Subtask 6.1: Docker Warm-Up (2 marks)**
  - Execute foundational Docker CLI operations using an official image (e.g. `python:3.10-slim` or `nginx:alpine`):
    - Run container, map port (`-p 8080:80`).
    - Mount volume (`-v $(pwd)/test_mount:/app/test`).
    - Inspect logs (`docker logs`).
    - Interactive shell access (`docker exec -it <container> sh`).
    - List and remove containers and images (`docker ps -a`, `docker rm`, `docker rmi`).
  - Student definitions in own words for: *Image*, *Container*, *Layer*, *Volume*, and *Port Mapping*.
  - *Evidence:* Terminal screenshots of each action and written definitions.
- **Subtask 6.2: Web App Dockerfile (3 marks)**
  - Write `docker/Dockerfile.web`:
    - Base image: `python:3.10-slim`.
    - Layer caching: copy `requirements-cpu.txt` and install via `pip install --no-cache-dir` *before* copying application code.
    - Security: create non-root user `appuser` (`useradd -m appuser`), set ownership, switch to `USER appuser`.
    - Health check instruction (`HEALTHCHECK CMD curl -f http://localhost:5000/health || exit 1`).
    - Package selection: Use CPU-only TensorFlow (`tensorflow-cpu==2.15.0`) to avoid massive CUDA/cuDNN binaries.
    - Write `.dockerignore` to exclude `.git`, `archive`, `.venv`, `.env`, `evidence`.
  - *Evidence:* Dockerfile text, explanation of instruction ordering, terminal output of `whoami` and `id` showing non-root user inside container.
- **Subtask 6.3: Docker Compose Stack (4 marks)**
  - Write `docker-compose.yml`:
    - `web`: builds `docker/Dockerfile.web`, depends on `db` with condition `service_healthy`, internal port 5000 mapped to host 5000, environment loaded from `.env`.
    - `db`: `postgres:15-alpine`, named persistent volume `postgres_data`, readiness healthcheck (`pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB`), **no published host ports** (closed to host/outside).
  - Supply `.env.example` with template variables; keep `.env` untracked.
  - Model file delivery: Package the frozen champion model inside the container image under `/app/models/` (or via mounted volume) with documented trade-offs.
  - *Evidence:* Link to Compose file, `docker compose ps` showing healthy status, `nmap` or `nc` proof that port 5432 is closed to outside.
- **Subtask 6.4: Image Analysis & Optimization (1.5 marks)**
  - Inspect image layer sizes using `docker history` and `docker images`.
  - Apply optimizations: compare standard build vs optimized build (using slim base, `--no-cache-dir`, removal of build tools, and `.dockerignore`).
  - Report exact sizes (before vs after) and document conclusions.
  - *Evidence:* Two size numbers with terminal output and analytical conclusion.
- **Subtask 6.5: Operations & Intentional Failure Troubleshooting (1.5 marks)**
  - Demonstrate operational commands: `docker compose up -d`, `docker compose stop`, `docker compose start`, `docker compose restart`, `docker compose logs`, and `docker compose exec`.
  - Controlled failure drill: inject intentional misconfiguration (e.g., incorrect database password in `.env` or missing model path), attempt startup, extract failure logs, diagnose root cause solely from log output, apply fix, and verify recovery.
  - *Evidence:* Terminal log excerpts showing failure trace, diagnosis notes, and recovery logs.
- **Subtask 6.6: GitHub Actions CI Tool Lab (2 marks)**
  - **Toy Lab:** Setup `.github/workflows/toy_ci.yml` in a throwaway repo (or isolated branch) with Python matrix (`3.10`, `3.11`), run a basic test. Inject intentional failure to obtain a RED run, then fix for a GREEN run.
  - **VisionGuard CI:** Setup `.github/workflows/ci.yml`:
    - Triggers on push and pull request.
    - Installs dependencies.
    - Runs linting (`flake8` / `black --check`).
    - Runs test suite (`pytest tests/`).
    - Builds Docker image to verify container construction.
  - Embed GitHub Actions status badge in `README.md`.
  - Explain where CI secrets belong (GitHub Repository Secrets) and why they must never be hardcoded into workflow YAML files.
  - *Evidence:* Screenshots of Red and Green toy runs, passing VisionGuard CI run, and README badge.

---

### TASK 7: Cloud API Deployment (15 marks)
- **Deployment Pathway Selection & Justification:**
  - We evaluate the permitted pathways:
    - *Option A: Oracle Cloud Infrastructure (OCI Always Free VM):* Excellent for running Compose, but requires credit-card verification and frequently suffers from "Out of host capacity" errors in common regions.
    - *Option B: Cloud Container / PaaS (Render.com / Hugging Face Spaces / Railway):* Completely free tier, native Docker support, instant HTTPS URL, no credit card required, reliable CPU resources for TensorFlow inference.
    - *Option C: AWS / GCP / Azure VM:* Free credit tiers available for students.
  - We will provide step-by-step guidance for the user's preferred option (e.g. Render / Hugging Face Spaces or OCI VM) ensuring zero unexpected costs.
- **Inference API Architecture:**
  - Build standalone lightweight inference service in `api/` exposing:
    - `GET /health`: service status and model metadata.
    - `POST /predict`: accepts multipart image upload, decodes, resizes, runs inference with champion model, returns JSON:
      ```json
      {
        "status": "success",
        "prediction": "AI-Generated",
        "confidence": 0.9421,
        "model_version": "MobileNetV2-Champion-v1",
        "timestamp": "2026-11-06T22:00:00Z"
      }
      ```
- **Testing & Verification:**
  - Verify public accessibility from an external network (e.g. mobile hotspot or curl from remote).
  - Execute comprehensive Postman API test suite: test with sample AI image, sample Real image, corrupt image, and invalid payload. Save Postman collection and response screenshots.
  - *Evidence:* Live public URL, Postman screenshots with headers/response payloads, deployment setup screenshots, and external network test logs.

---

### TASK 8: Flask & Hosted API Integration (15 marks)
- **Architecture & Integration:**
  - Configure the local Flask web application (`webapp/app.py`) to operate in remote mode:
    - Set `INFERENCE_MODE=remote` and `REMOTE_INFERENCE_API_URL=https://<your-cloud-api>/predict`.
  - When a user uploads an image to the local Flask web interface:
    1. Flask validates the file in memory.
    2. Flask streams the image payload via HTTP POST to the cloud-hosted API.
    3. The cloud API returns prediction verdict and confidence.
    4. Flask logs the complete transaction (including cloud model version) to the database (Task 5).
    5. Flask renders the prediction result on `result.html` with caveat warning.
- **Robust Error Handling:**
  - Implement timeout handling, connection retries, and friendly fallback UI if cloud service is waking up or temporarily unreachable.
- **Verification & Demonstration:**
  - Classify multiple test images across both classes through the local web interface talking to the remote cloud API.
  - Open `/history` menu to verify that every cloud-backed inference has been logged into the database.
  - *Evidence:* Side-by-side screenshots showing: local web UI upload $\to$ cloud API response $\to$ database history table updated.

---

## 5. Evidence Collection Strategy & Report Structure

### 5.1 Evidence Directory Structure
Every single subtask will save its raw outputs immediately into `evidence/task<ID>/`:
- `evidence/task1/`: `env_versions.txt`, `audit_table.md`, `sample_grid.png`, `distributions.png`, `dvc_commits.txt`
- `evidence/task2/`: `augmented_batch.png`, `baseline_summary.txt`, `transfer_curves.png`, `best_epoch.txt`, `confusion_matrix.png`, `roc_curve.png`, `misclassified_grid.png`, `tuner_results.csv`, `gradcam_grid.png`
- `evidence/task3/`: `mlflow_toy_runs.png`, `dagshub_setup.png`, `manual_vs_auto_curves.png`, `mlflow_artifacts.png`, `model_comparison_table.png`, `registry_champion.png`
- `evidence/task4/`: `inference_startup.txt`, `error_bad_ext.png`, `error_size.png`, `error_corrupt.png`, `frontend_home.png`, `frontend_result.png`, `health_check.png`
- `evidence/task5/`: `db_schema.sql`, `db_history_ui.png`, `db_query_output.txt`
- `evidence/task6/`: `docker_warmup.png`, `dockerfile_user.txt`, `compose_ps.png`, `port_closed_proof.txt`, `image_sizes.txt`, `troubleshooting_log.txt`, `toy_ci_red_green.png`, `ci_passing.png`
- `evidence/task7/`: `cloud_deployment_steps.png`, `postman_test_real.png`, `postman_test_ai.png`, `external_network_test.txt`
- `evidence/task8/`: `integrated_inference_flow.png`, `integrated_history_log.png`

### 5.2 Final Report Compilation (`23L0710_VisionGuard.pdf`)
The report will strictly follow Section 11 of the assignment PDF:
1. **Front Matter:** Student Name, Roll Number, GitHub repo link, DagsHub project link, live Cloud API URL, graded Git commit hash, AI assistant disclosure.
2. **Chapters 1 through 8:** Dedicated chapter for each assignment task detailing *what was done*, *why it was done*, *how it works*, *problems encountered and fixes*, and embedded high-resolution screenshots with contextual captions.
3. **Submission Checklist:** Verification against all 100 marks + bonus criteria.

---

## 6. Prerequisites & Manual User Actions Checklist

To ensure smooth sequential execution, the following external accounts and actions will be guided step-by-step when their respective subtasks are reached:
1. **Roll Number Configured:** `23L0710` (Used for repository naming, project variables, and report naming).
2. **GitHub Account:** Create empty repository named `23L0710_A2_Ops`.
3. **DagsHub Account:** Connect GitHub repo to DagsHub, generate DagsHub access token.
4. **Docker Desktop:** Launch Docker Desktop on macOS prior to Task 6.
5. **Cloud Provider Account:** Free-tier account on chosen host (Render / Hugging Face / Oracle Cloud) prior to Task 7.
6. **Postman:** Desktop app or web client to test the cloud API.

