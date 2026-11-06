# Task 4.4: Configuration & Production Server Hardening Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Endpoint:** `GET /health`  
**Server Execution Mode:** Flask Server with `debug=False`  

---

## 1. Environment-Driven Configuration

All operational parameters are decoupled from source code and read dynamically from environment variables (`.env`):
- `MODEL_PATH`: Path to serialized Keras model (`models/champion.keras`).
- `DECISION_THRESHOLD`: Validation-tuned operating threshold ($0.23$).
- `INFERENCE_MODE`: Local in-process (`local`) or remote cloud forwarding (`remote`).
- `MAX_CONTENT_LENGTH`: Hard limit on upload payload ($5\text{ MB}$).
- `SECRET_KEY`: Secure cryptographic key for session cookies.

---

## 2. Production Justification: Why `debug=False` is Mandatory in Production

Running Flask with `debug=True` in production is a critical security vulnerability and operational hazard:
1. **Arbitrary Code Execution via Interactive Werkzeug Debugger:**  
   When an unhandled exception occurs in debug mode, Flask exposes the interactive Werkzeug pin-protected debugger in the browser. An attacker can execute arbitrary Python commands directly on the host server shell, resulting in complete system compromise and data exfiltration.
2. **Denial of Service via Auto-Reloading:**  
   Debug mode activates the stat-based auto-reloader (Werkzeug watchdog), which polls the filesystem continuously. Under heavy web concurrency, this creates massive disk I/O, process restarts, and dropped TCP connections.
3. **Information Disclosure:**  
   Tracebacks reveal sensitive environment variables, internal directory paths, database connection strings, and model architecture details to end users.

---

## 3. Health-Check Endpoint (`GET /health`)

The health check endpoint provides a lightweight, automated mechanism for container orchestrators (Docker Compose, Kubernetes) and load balancers to determine service viability:

```json
{
  "decision_threshold": 0.23,
  "inference_mode": "local",
  "max_upload_size_mb": 5,
  "model_loaded": true,
  "model_name": "VisionGuard_MobileNetV2",
  "model_path": "models/champion.keras",
  "project_name": "23L0710_A2_Ops",
  "status": "healthy",
  "student_roll_number": "23L0710",
  "timestamp": "2026-11-07T03:15:00.000000"
}
```

- Live JSON output saved to: `evidence/task4/health_check.json`
- Screenshot capture saved to: `evidence/task4/health_check_response.png`
