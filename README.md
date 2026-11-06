# VisionGuard: AI vs. Real Image Detection & MLOps Platform

[![VisionGuard CI](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/workflows/ci_visionguard.yml/badge.svg)](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/workflows/ci_visionguard.yml)
[![DagsHub Project](https://img.shields.io/badge/DagsHub-23L0710__A2__Ops-0066FF?logo=dagshub&logoColor=white)](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/Docker_Compose-v2-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops.mlflow)

**Student Name:** Shoaib Asif  
**Roll Number:** 23L0710  
**Course:** MLOps (CS / AI) — Assignment 2  
**Standard Naming:** `23L0710_A2_Ops`  
**GitHub Repository:** [https://github.com/shoaib-asiff/23L0710_A2_Ops](https://github.com/shoaib-asiff/23L0710_A2_Ops)  
**DagsHub Project:** [https://dagshub.com/shoaib-asiff/23L0710_A2_Ops](https://dagshub.com/shoaib-asiff/23L0710_A2_Ops)  

---

## 1. Project Overview

**VisionGuard** is an end-to-end production machine learning system engineered to classify images as either **Authentic Human Photography (Real)** or **AI-Generated Synthetic Media**. 

The platform implements complete MLOps lifecycle best practices:
- **Data Engineering & Versioning:** Stratified input pipelines, aspect-preserving zero-padding to 224x224, and DVC pointer tracking.
- **Deep Learning Model Development:** Transfer learning using MobileNetV2 with two-phase fine-tuning, ROC-AUC threshold tuning ($\tau^* = 0.23$), error analysis, Keras Tuner hyperparameter exploration, and Explainable AI (Grad-CAM).
- **Remote Experiment Tracking & Model Registry:** Comprehensive instrumentation logging parameters, dynamic epoch metrics, confusion matrices, ROC curves, and artifacts directly to remote DagsHub MLflow server. Model versions registered with `@champion` aliases.
- **Production Web Application:** Decoupled Flask application served via Gunicorn WSGI with in-memory image validation, zero disk leak, verdict badges, and honest engineering caveat disclosures.
- **Audit Persistence Trail:** SQLAlchemy ORM logging classification transactions into PostgreSQL / SQLite with queryable historical inspection.
- **Containerization & Orchestration:** Production multi-container Docker Compose stack (`web` + `db`) on an internal bridge network, hardened with non-root security (`appuser:appgroup`), native health probes, and layer optimization (reducing image footprint by 49.5%).
- **Continuous Integration (CI):** GitHub Actions workflows enforcing linting (`flake8`), automated unit testing, and Docker container build verification on every push and pull request.

---

## 2. System Architecture

```
                       [ Incoming Client Request ]
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Host Network (Port 5001 / Cloud)           │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             VisionGuard Web Container (Flask)          │
       │  • Gunicorn WSGI Server (Non-root appuser:appgroup)    │
       │  • Singleton Inference Engine (In-Memory Preprocessing)│
       │  • Health Probe (/health) & Web UI (/)                 │
       └──────────────┬───────────────────────────┬─────────────┘
                      │                           │
          Inference   │                           │ Audit Trail
          Forwarding  ▼                           ▼ Logging
       ┌────────────────────────┐      ┌────────────────────────┐
       │ Champion MobileNetV2   │      │ Internal Bridge Network│
       │ (models/champion.keras)│      │ visionguard_db:5432    │
       │ Decision Threshold=0.23│      │ (PostgreSQL 15 Alpine) │
       │ Grad-CAM Explainability│      │ Named Volume Storage   │
       └────────────────────────┘      └────────────────────────┘
```

---

## 3. Directory Structure

```text
23L0710_A2_Ops/
├── .github/
│   └── workflows/
│       ├── ci_toy.yml              # Matrix workflow testing across Python 3.10 and 3.11
│       └── ci_visionguard.yml      # Production CI (flake8, unit tests, docker build)
├── archive/                        # Raw image dataset (DVC managed)
├── docker-compose.yml              # Multi-container orchestration (web + db)
├── Dockerfile                      # Production slim container specification
├── evidence/                       # Audit artifacts, logs, and screenshots (Tasks 1-8)
│   ├── task1/                      # EDA distributions, environment logs, DVC manifests
│   ├── task2/                      # Curves, confusion matrices, Grad-CAM overlays
│   ├── task3/                      # MLflow comparison tables, model registry proof
│   ├── task4/                      # Webapp failure modes, UI snapshots, console outputs
│   ├── task5/                      # Database schema SQL, query logs, history UI
│   └── task6/                      # Docker warmup, image optimization, troubleshooting logs
├── models/
│   └── champion.keras              # Serialized Champion MobileNetV2 model
├── requirements.txt                # Full development dependencies
├── requirements-docker.txt         # Container dependencies (tensorflow-cpu)
├── src/
│   ├── audit.py                    # Dataset validation and integrity audit
│   ├── database.py                 # SQLAlchemy ORM persistence module
│   ├── dataset.py                  # tf.data streaming and preprocessing pipeline
│   ├── evaluate.py                 # Evaluation metrics and ROC-AUC threshold tuning
│   ├── experiment_comparison.py    # 6-run MLflow comparison suite
│   ├── image_analysis.py           # Docker layer size inspection script
│   ├── registry_loader.py          # Dynamic champion model loader from DagsHub
│   ├── test_compose_stack.py       # Compose validation and closed-port verification
│   ├── test_compose_troubleshooting.py # Operations & fault drill automation
│   ├── train.py                    # Transfer learning training pipeline
│   └── tuner.py                    # Hyperparameter optimization lab
├── tests/
│   ├── test_toy.py                 # Toy matrix CI regression test
│   └── test_unit.py                # Preprocessing, validation, and health unit tests
└── webapp/
    ├── app.py                      # Flask application factory and HTTP routes
    ├── inference.py                # Decoupled singleton inference engine
    ├── static/                     # CSS stylesheets and UI assets
    └── templates/                  # Jinja2 HTML templates (Home, Result, History)
```

---

## 4. Quickstart Guide

### 4.1 Running with Docker Compose (Recommended)

Ensure Docker Desktop is running, then execute:

```bash
# 1. Copy sample environment file
cp .env.example .env

# 2. Build and launch the multi-container stack in detached mode
docker compose up -d --build

# 3. Verify services and health status
docker compose ps

# 4. Open the Web Application
# Access via browser: http://localhost:5001
```

To stop the services:
```bash
docker compose down
```

### 4.2 Local Python Environment Setup

```bash
# 1. Create and activate Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements-docker.txt

# 3. Run unit tests
python -m unittest tests/test_unit.py

# 4. Launch Flask development server
python webapp/app.py
# Web application available at http://localhost:5000
```

---

## 5. API Endpoints

| HTTP Method | Route | Description | Response / Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web portal home interface. | HTML (Drag & drop upload form) |
| `POST` | `/predict` | Multipart image upload for AI vs Real classification. | HTML (Prediction Verdict & Confidence) |
| `GET` | `/history` | Chronological audit trail of past classifications. | HTML (Filterable audit log table) |
| `GET` | `/health` | Container & engine health telemetry. | `application/json` (Status & Model metadata) |

---

## 6. Continuous Integration & Secrets Policy

1. **GitHub Actions CI Pipeline:**
   Every push to `main` and all pull requests trigger `.github/workflows/ci_visionguard.yml`. The pipeline enforces:
   - Python code quality and syntax verification via `flake8`.
   - Comprehensive unit testing validating aspect-ratio padding, byte decoding, and health endpoints.
   - Isolated Docker container build and runtime smoke testing.

2. **Security & Secrets Best Practice:**
   - Secrets such as `DAGSHUB_USER_TOKEN`, database passwords, or cloud credentials **MUST NEVER** be hardcoded into YAML files or committed to Git.
   - All runtime credentials are encrypted and stored within **GitHub Repository Secrets** (`Settings -> Secrets and variables -> Actions`) and injected at runtime via environment variables.
   - `.env` files are strictly excluded from version control via `.gitignore`.

