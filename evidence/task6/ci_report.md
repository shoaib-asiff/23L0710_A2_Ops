# Task 6.6: Continuous Integration with GitHub Actions Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Repository:** [https://github.com/shoaib-asiff/23L0710_A2_Ops](https://github.com/shoaib-asiff/23L0710_A2_Ops)  
**CI Badge:** [![VisionGuard CI](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/workflows/ci_visionguard.yml/badge.svg)](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/workflows/ci_visionguard.yml)  

---

## 1. Executive Summary & Objective

Continuous Integration (CI) is a foundational pillar of MLOps that automates the verification of code quality, model preprocessing integrity, unit functionality, and container buildability on every code push and pull request. This report documents the implementation and verification of **Task 6.6**:
1. **Toy CI Matrix Workflow (`ci_toy.yml`):** A dual-version Python test matrix (`3.10` and `3.11`) demonstrating both a controlled **Red Run (Failure Drill)** and a remediated **Green Run (Passing Build)**.
2. **Production VisionGuard CI Pipeline (`ci_visionguard.yml`):** An automated multi-stage pipeline executing static code analysis (`flake8`), domain-specific unit testing (`tests/test_unit.py`), and full Docker container construction with in-container smoke testing.
3. **Repository Status Badge:** Integration of a dynamic SVG build status badge directly into the root `README.md`.
4. **Secrets Management Governance:** Comprehensive architectural justification analyzing why credentials must reside strictly in GitHub Repository Secrets and never appear in version-controlled workflow YAML files.

---

## 2. Toy CI Matrix Workflow: Red vs. Green Demonstration

The Toy CI workflow (`.github/workflows/ci_toy.yml`) targets two parallel runner environments: **Python 3.10** and **Python 3.11** on `ubuntu-latest`.

### 2.1 The Red Run (Controlled Failure Drill)
- **Workflow Name:** `Toy CI Matrix Workflow`
- **GitHub Run ID:** `42312786844`
- **Run URL:** [https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/runs/42312786844](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/runs/42312786844)
- **Trigger Commit:** `600cf25` (*"feat(ci): add toy matrix and visionguard ci workflows with deliberate failure drill"*)
- **Failure Drill Procedure:** An intentional assertion failure was injected into `tests/test_toy.py`:
  ```python
  def test_intentional_failure_drill(self):
      expected = "EXPECTED_GREEN_STATUS"
      actual = "INTENTIONAL_RED_FAILURE_DRILL"
      self.assertEqual(actual, expected)
  ```
- **Observed CI Behavior:** Both matrix runners (`Python 3.10` and `Python 3.11`) intercepted the assertion error, exited with code 1, halted the pipeline, and correctly displayed a **RED** failure state on GitHub.

### 2.2 The Green Run (Remediation & Pass)
- **Workflow Name:** `Toy CI Matrix Workflow`
- **GitHub Run ID:** `42313178806`
- **Run URL:** [https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/runs/42313178806](https://github.com/shoaib-asiff/23L0710_A2_Ops/actions/runs/42313178806)
- **Trigger Commit:** `f5802c2` (*"fix(ci): remediate toy matrix test assertion to achieve green build"*)
- **Remediation Applied:** The assertion was updated to verify parity:
  ```python
  def test_intentional_failure_drill(self):
      expected = "EXPECTED_GREEN_STATUS"
      actual = "EXPECTED_GREEN_STATUS"
      self.assertEqual(actual, expected)
  ```
- **Observed CI Behavior:** Both matrix jobs completed in under 12 seconds with exit code 0, turning the build status to **GREEN (SUCCESS)**.

| Run Characteristic | Red Run (Failure Drill) | Green Run (Remediated) |
| :--- | :---: | :---: |
| **Run ID** | `42312786844` | `42313178806` |
| **Commit Hash** | `600cf25` | `f5802c2` |
| **Python 3.10 Job** | **FAILED** (AssertionError) | **SUCCESS** (12s) |
| **Python 3.11 Job** | **FAILED** (AssertionError) | **SUCCESS** (10s) |
| **Overall Status** | `failure` | `success` |

---

## 3. Production VisionGuard CI Pipeline Architecture

The primary CI pipeline (`.github/workflows/ci_visionguard.yml`) runs on every push and pull request to `main`. It consists of two sequential jobs:

```
[ Git Push / PR ] ──▶ [ Job 1: Lint & Unit Tests ] ──▶ [ Job 2: Build & Verify Docker Image ] ──▶ [ PASS ]
```

### 3.1 Job 1: Linting & Unit Testing (`lint-and-test`)
- **Environment:** `ubuntu-latest`, Python `3.10`.
- **Dependency Caching:** Pip wheel caching enabled via `actions/setup-python@v5` using `cache: 'pip'`.
- **Static Analysis (Flake8):**  
  Checks for critical syntax errors, unresolved symbols, and undefined variables across all application modules:
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,archive,warmup_data,evidence
  ```
- **Automated Unit Testing (`tests/test_unit.py`):**
  1. `test_file_extension_whitelist`: Validates image upload security policy (permits `.jpg`, `.png`, `.webp`, `.bmp`; rejects `.exe`, `.py`, `.sh`).
  2. `test_preprocessing_aspect_ratio_and_shape`: Verifies training-serving preprocessing parity. Confirms that asymmetric inputs are zero-padded to `(1, 224, 224, 3)` with float values in `[0.0, 255.0]`.
  3. `test_health_check_endpoint`: Queries `GET /health` to confirm HTTP 200 and JSON telemetry structure (`status: "healthy"`, `model_loaded: true`, `student_roll_number: "23L0710"`).
  4. `test_database_orm_record_structure`: Confirms SQLAlchemy `PredictionRecord` entity properties and audit logging readiness.
- **Result:** 4/4 tests passed in 0.051s. Overall job completed in 38s.

### 3.2 Job 2: Container Construction & Smoke Testing (`docker-build`)
- **Dependency:** Runs strictly if `lint-and-test` passes (`needs: lint-and-test`).
- **Build Engine:** Uses `docker/setup-buildx-action@v3` and `docker/build-push-action@v5`.
- **Build Target:** Builds production image from `Dockerfile` (`python:3.10-slim`, `tensorflow-cpu==2.15.0`, non-root user `appuser:appgroup`).
- **Container Smoke Test:** Executes an in-container Python command within the built image to confirm TensorFlow runtime integrity before deployment:
  ```bash
  docker run --rm visionguard-web:ci python -c "import tensorflow as tf; print(f'CI Container Verification: TensorFlow {tf.__version__} OK')"
  ```
- **Result:** Container build and smoke test succeeded in 1m 15s.

---

## 4. CI Secrets Management Policy & Security Analysis

A critical requirement of enterprise MLOps engineering is the strict isolation of secrets from version control systems.

### 4.1 Where CI Secrets Belong: GitHub Repository Secrets
Secrets required by CI/CD workflows—such as `DAGSHUB_USER_TOKEN`, Docker Hub credentials, cloud deployment API keys (e.g. Render, AWS, GCP), and database production passwords—**must reside exclusively in encrypted GitHub Repository Secrets** (`Settings -> Secrets and variables -> Actions`).

Workflows consume these secrets securely using environment mapping:
```yaml
env:
  DAGSHUB_USER_TOKEN: ${{ secrets.DAGSHUB_USER_TOKEN }}
  DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
```

### 4.2 Why Secrets Must NEVER Be Hardcoded in Workflow YAML Files
1. **Public Repository Exposure:**  
   Workflow YAML files are plain-text files located in `.github/workflows/`. In open-source or shared academic repositories, any hardcoded token is immediately visible to anyone inspecting the code.
2. **Permanent Git Reflog Retention:**  
   Even if a developer commits a secret and deletes it in the very next commit, the secret remains permanently etched in the Git commit object database and reflogs. Attackers routinely clone repositories and extract historical secrets using automated scanners like `TruffleHog` or `GitGuardian`.
3. **Malicious PR Injection Attacks:**  
   Hardcoded secrets in YAML are accessible to untrusted pull requests. In contrast, GitHub Actions enforces that `secrets.*` are strictly redacted and hidden from pull requests originating from repository forks.
4. **Centralized Secret Rotation:**  
   When secrets are managed via GitHub Repository Secrets, tokens can be rotated, expired, or replaced instantly in the web dashboard without modifying source code, drafting new Git commits, or redeploying software.

---

## 5. Verification & Summary Table

| CI Component | File Path | Trigger / Strategy | Status | Evidence File |
| :--- | :--- | :--- | :---: | :--- |
| **Toy Matrix (Red)** | `.github/workflows/ci_toy.yml` | Python 3.10 & 3.11 matrix | **FAILURE** (Caught) | `evidence/task6/ci_toy_red_green.png` |
| **Toy Matrix (Green)** | `.github/workflows/ci_toy.yml` | Python 3.10 & 3.11 matrix | **SUCCESS** | `evidence/task6/ci_toy_red_green.png` |
| **Production CI** | `.github/workflows/ci_visionguard.yml` | Push & PR to `main` | **SUCCESS** | `evidence/task6/ci_passing.png` |
| **Unit Test Suite** | `tests/test_unit.py` | 4 unit test classes | **4/4 PASSED** | `evidence/task6/ci_report.md` |
| **README Status Badge** | `README.md` | Dynamic SVG badge | **ACTIVE** | [Repo README](file:///Users/shoaib/Desktop/University/7th Semester/MLOPS/A2/README.md) |

