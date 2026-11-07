# Task 7: Cloud API Deployment Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Cloud Hosting Platform:** Render.com (Container PaaS)  
**Public Endpoint URL:** [https://visionguard-api-23l0710.onrender.com](https://visionguard-api-23l0710.onrender.com)  
**Postman Collection:** `evidence/task7/VisionGuard_Postman_Collection.json`  
**Execution Timestamp:** 2026-11-07 21:57:46 UTC  

---

## 1. Executive Summary & Objective

In modern machine learning engineering, deploying models as scalable cloud microservices enables client applications, web portals, and mobile frontends to consume predictions over standard HTTPS REST protocols. This report documents the implementation and verification of **Task 7**:
1. Selection, architecture, and deployment of the VisionGuard inference engine to **Render.com** cloud infrastructure.
2. Exposure and hardening of two core production REST endpoints:
   - `GET /health`: Diagnostic health probe returning container status, loaded model architecture, decision threshold, and student provenance metadata.
   - `POST /predict`: Multipart image ingestion supporting binary classification, calibrated confidence scoring, and strict input validation.
3. Verification across an external network covering valid authentic photography, synthetic AI art, and 3 security failure modes.
4. Export of the official **Postman Collection v2.1** for API grading.

---

## 2. Cloud Architecture & Deployment Pathway Justification

### 2.1 Provider Selection: Render Cloud PaaS
We selected **Render.com** as our cloud hosting provider based on the following engineering criteria:
- **Native Container Runtime:** Render natively pulls the GitHub repository (`shoaib-asiff/23L0710_A2_Ops`) and constructs the production Docker container from `Dockerfile`.
- **Zero Cost & No Surprise Billing:** Operating on Render's 100% free container tier guarantees zero cloud expenditure.
- **Automated SSL/TLS Certificate Provisioning:** Render provisions end-to-end HTTPS encryption with automatic Let's Encrypt certificates.
- **Resource Sizing & Optimization:** Our optimized `python:3.10-slim` + `tensorflow-cpu==2.15.0` container requires only ~430 MB RAM during peak inference, fitting within Render's 512 MB memory boundary. Gunicorn is configured with `--workers 1 --threads 4` to prevent OOM events while supporting concurrent request processing.

### 2.2 Infrastructure-as-Code (`render.yaml`)
The cloud service configuration is codified in `render.yaml`:
```yaml
services:
  - type: web
    name: visionguard-api-23l0710
    runtime: docker
    plan: free
    region: frankfurt
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: PORT
        value: 5000
      - key: INFERENCE_MODE
        value: local
      - key: DECISION_THRESHOLD
        value: "0.23"
      - key: MODEL_PATH
        value: models/champion.keras
```

---

## 3. External Network Test Results & Postman Test Matrix

All tests were executed against the live cloud endpoint from an external network connection:

| Test Case | HTTP Status | Response Latency | Verification Verdict |
| :--- | :---: | :---: | :---: |
| **GET /health (Health Probe)** | `200` | `575.2ms` | **PASS** |
| **POST /predict (Real Art)** | `200` | `2253.5ms` | **PASS** |
| **POST /predict (AI Art)** | `200` | `1636.9ms` | **PASS** |
| **POST /predict (Wrong Extension)** | `400` | `513.5ms` | **PASS** |
| **POST /predict (Corrupt Image)** | `400` | `572.6ms` | **PASS** |
| **POST /predict (Exceed 5MB)** | `413` | `4452.3ms` | **PASS** |

---

## 4. API Endpoint Specifications

### 4.1 Diagnostic Health Check (`GET /health`)
- **URL:** `https://visionguard-api-23l0710.onrender.com/health`
- **Method:** `GET`
- **Headers:** `Accept: application/json`
- **Response Payload:**
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
  "timestamp": "2026-11-08T02:30:00Z"
}
```

### 4.2 Binary Image Classification (`POST /predict`)
- **URL:** `https://visionguard-api-23l0710.onrender.com/predict`
- **Method:** `POST`
- **Headers:** `Accept: application/json`
- **Body:** `multipart/form-data` with key `image` or `file`
- **Sample Success Response:**
```json
{
  "status": "success",
  "filename": "authentic_photo.jpg",
  "prediction": "Real",
  "confidence": 83.97,
  "raw_score": 0.040182,
  "threshold": 0.23,
  "model_name": "VisionGuard_MobileNetV2_Champion",
  "model_version": "VisionGuard_MobileNetV2_Champion",
  "inference_mode": "local",
  "inference_time_ms": 115.3,
  "student_roll_number": "23L0710",
  "timestamp": "2026-11-08T02:30:05Z"
}
```

---

## 5. Postman Collection Integration

The complete test suite is packaged in `evidence/task7/VisionGuard_Postman_Collection.json` adhering to the Postman v2.1 schema. Evaluators can import this file directly into Postman to run automated integration tests against the live URL.
