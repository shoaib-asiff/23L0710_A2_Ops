# Task 6.5: Operations & Intentional Failure Troubleshooting Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Execution Timestamp:** 2026-11-06 23:25:57 UTC  

---

## 1. Executive Summary & Objective

In production MLOps environments, operational reliability demands proficiency in container lifecycle management, real-time log analysis, and incident remediation. This report documents the execution of **Task 6.5**, demonstrating:
1. Routine multi-container operational commands across both the `web` application and `db` (PostgreSQL) services.
2. Two controlled **fault injection drills**:
   - **Fault Drill A:** Database authentication mismatch (`OperationalError: password authentication failed`).
   - **Fault Drill B:** Model artifact path misconfiguration (`FileNotFoundError: model file not found`).
3. Root cause diagnostics derived strictly from container log excerpts.
4. Corrective action execution and verification of full service recovery back to `healthy` state.

---

## 2. Operational Lifecycle Commands Demonstration

The following core Docker Compose commands were executed and verified against the running multi-container stack:

| Operational Command | Target Service | Function & Observed System Behavior | Verification Status |
| :--- | :---: | :--- | :---: |
| `docker compose ps` | Stack | Enumerates running containers, mapped ports, and health status (`healthy`). | **PASS** |
| `docker compose logs --tail=15 web` | `web` | Streams application stdout/stderr, Gunicorn worker boots, and health probe queries. | **PASS** |
| `docker compose logs --tail=15 db` | `db` | Streams PostgreSQL engine logs, connection handshakes, and checkpoint activity. | **PASS** |
| `docker compose exec web python -c "..."` | `web` | Executes an in-container Python command, validating `Python 3.10` and `TensorFlow 2.15.0`. | **PASS** |
| `docker compose exec db pg_isready ...` | `db` | Queries PostgreSQL daemon socket directly inside container (`5432 - accepting connections`). | **PASS** |
| `docker compose stop web` | `web` | Sends `SIGTERM` followed by graceful shutdown of Gunicorn processes; container state becomes `Exited (0)`. | **PASS** |
| `docker compose start web` | `web` | Re-initializes existing stopped container; restarts Gunicorn processes; status transitions back to `healthy`. | **PASS** |
| `docker compose restart web` | `web` | Performs atomic stop-and-start sequence without recreating container filesystem. | **PASS** |
| `curl http://localhost:5001/health` | `web` | Validates end-to-end HTTP routing and application health (`200 OK`, `status: "healthy"`). | **PASS** |

---

## 3. Controlled Fault Injection Drill A: Database Authentication Failure

### 3.1 Failure Injection Procedure
To simulate an environment configuration drift incident, the `.env` file was modified by replacing the valid database password with an invalid credential:
```text
# Injected Fault:
DATABASE_URL=postgresql://visionguard_user:INJECTED_BAD_PASSWORD@db:5432/visionguard_db
```
The web service was reloaded (`docker compose up -d web`), and a database connection was requested.

### 3.2 Observed Container Failure Log
```text
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "db" (172.18.0.2), port 5432 failed: FATAL: password authentication failed for user "visionguard_user"
(Background on this error at: https://sqlalche.me/e/20/e3q8)
```

### 3.3 Diagnostic Analysis (Derivation Solely from Logs)
- **Error Class:** `psycopg2.OperationalError` wrapped by `sqlalchemy.exc.OperationalError`.
- **Target Host & Port:** `db:5432` at internal bridge IP `172.18.0.2`.
- **Root Cause Deduction:** The PostgreSQL daemon is online and listening on port 5432, proving network reachability across the bridge. However, the database server rejected the client handshake with `FATAL: password authentication failed for user "visionguard_user"`. This isolates the defect specifically to credential mismatch between the client's injected `DATABASE_URL` and the database's initialized `POSTGRES_PASSWORD`.

### 3.4 Remediation & Verification
1. **Fix Applied:** Restored original valid credentials from `.env.troubleshoot_bak` (`POSTGRES_PASSWORD=visionguard_secret_pwd`).
2. **Re-deployment:** Executed `docker compose up -d web`.
3. **Recovery Verification:** Executed `from src.database import engine; conn = engine.connect()`.
4. **Result:** Connection succeeded (`DB Connection Successful: OK`), confirming complete restoration of database persistence.

---

## 4. Controlled Fault Injection Drill B: Missing Model Artifact Path

### 4.1 Failure Injection Procedure
To simulate a container deployment where the machine learning model weight artifact path is misconfigured or corrupted during a pipeline migration, the following variable was injected:
```text
MODEL_PATH=/app/models/missing_model_v9.keras
```
The web service was restarted, and a sample inference POST request was submitted.

### 4.2 Observed Container Failure Log
```text
FileNotFoundError: Model file /app/models/missing_model_v9.keras not found
[2026-11-07 04:25:00,123] ERROR in app: Exception on /predict [POST]
Traceback (most recent call last):
  File ".../webapp/inference.py", line 105, in get_inference_engine
    raise FileNotFoundError(f"Model file {model_path} not found")
```

### 4.3 Diagnostic Analysis (Derivation Solely from Logs)
- **Error Class:** `FileNotFoundError`.
- **Target File:** `/app/models/missing_model_v9.keras`.
- **Root Cause Deduction:** The inference engine initialization function `get_inference_engine()` checked for file existence via `os.path.exists()` before loading weights. The filesystem path `/app/models/missing_model_v9.keras` does not exist in the container image. Because the model artifact is a critical dependency, the prediction pipeline halted immediately.

### 4.4 Remediation & Verification
1. **Fix Applied:** Removed the invalid override from `.env`, defaulting `MODEL_PATH` back to the bundled champion model artifact `models/champion.keras`.
2. **Re-deployment:** Executed `docker compose up -d web`.
3. **Recovery Verification:** Queried `GET /health` via curl:
   ```json
   {
     "inference_mode": "local",
     "model_loaded": true,
     "model_name": "VisionGuard_MobileNetV2_Champion",
     "model_path": "models/champion.keras",
     "status": "healthy"
   }
   ```
4. **Result:** HTTP 200 OK returned with `"status": "healthy"` and `"model_loaded": true`.

---

## 5. Conclusion & Operational Best Practices

The troubleshooting exercise demonstrated key principles of resilient MLOps container management:
1. **Structured Log Telemetry:** Standardizing stdout/stderr logging allows issues to be diagnosed within seconds without interactive container debugging.
2. **Graceful Signal Handling:** Implementing non-blocking health checks and decoupling database initialization prevents container crash loops while maintaining visibility.
3. **Configuration Defense:** Managing sensitive parameters via clean environment variables (`.env`) enables rapid remediation of configuration drift without requiring full image rebuilds.
