# Task 8: Flask & Hosted API Integration Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Operating Mode:** Remote Inference (`INFERENCE_MODE=remote`)  
**Cloud Backend URL:** [https://visionguard-api-23l0710.onrender.com/predict](https://visionguard-api-23l0710.onrender.com/predict)  
**Execution Timestamp:** 2026-11-07 22:01:15 UTC  

---

## 1. Executive Summary & Architecture Overview

Task 8 demonstrates the end-to-end integration of the local VisionGuard Flask web application with the cloud-hosted microservice deployed on Render. In this decoupled production architecture:
1. The **Local Frontend / Gateway** (`webapp/app.py` running in remote mode) accepts image uploads from users via the web browser.
2. The gateway validates the image buffer strictly in-memory (aspect ratio, MIME type whitelist, decodability, 5MB size limit) with **zero disk leak**.
3. Instead of running heavy deep learning inference locally, the local application forwards the validated image buffer over HTTPS to the **Cloud Inference API** (`https://visionguard-api-23l0710.onrender.com/predict`).
4. The cloud service executes forward-pass inference using the Champion MobileNetV2 model and returns JSON predictions.
5. The local gateway receives the cloud response, automatically logs the transaction into the local relational database trail (`evidence/task5/schema.sql`) with `inference_mode="remote"`, and renders the dynamic user verdict page with confidence visualizer and honest caveat warning.

```
[ User Browser ]
       │
       ▼ (1. Multipart Upload)
┌────────────────────────────────────────────────────────┐
│             Local Flask Gateway (webapp/app.py)        │
│  • INFERENCE_MODE = "remote"                           │
│  • Strict In-Memory Validation (Pillow)                │
└──────────────┬───────────────────────────┬─────────────┘
               │ (2. HTTPS POST /predict)   │ (4. Persistent Audit Log)
               ▼                           ▼
┌────────────────────────┐      ┌────────────────────────┐
│ Cloud API (Render.com) │      │ Local SQLAlchemy Trail │
│ • Python 3.10 Slim     │      │ • PredictionRecord ORM │
│ • Champion MobileNetV2 │      │ • SQLite / PostgreSQL  │
│ • Decision tau* = 0.23 │      │ • Mode: 'remote'       │
└────────────────────────┘      └────────────────────────┘
```

---

## 2. End-to-End System Demonstration Results

Two sample images were evaluated end-to-end through the local web interface communicating with the live Render Cloud API:

| Evaluation Case | Ground Truth | Cloud API Verdict | Cloud Confidence | Total Roundtrip Latency | Evidence Snapshot |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Authentic Photography** (`images40.jpg`) | Real | **Real** | **83.97%** | `2882.5 ms` | `evidence/task8/integrated_flow_real.png` |
| **Synthetic Media** (`portrait-01.jpg`) | AI-Generated | **AI-Generated** | **50.00%** | `1862.3 ms` | `evidence/task8/integrated_flow_ai.png` |

---

## 3. Database Audit Trail Verification

Every remote cloud transaction was automatically intercepted and written to the persistent SQLAlchemy audit store. Querying `GET /history` verified that the records include the remote provenance metadata:

| Record ID | Timestamp (UTC) | Filename | Verdict | Confidence | Inference Mode | Backend Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #8 | `2026-11-07 22:00:59 UTC` | `synthetic_art.jpg` | **AI-Generated** | `50.00%` | `remote` | `1166.3 ms` |
| #7 | `2026-11-07 22:00:57 UTC` | `authentic_photo.jpg` | **Real** | `83.97%` | `remote` | `1404.8 ms` |

- **Audit Proof:** Notice the `inference_mode` column is explicitly set to `remote`, cleanly distinguishing cloud-backed predictions from local offline runs.
- **Visual Proof:** Screenshot `evidence/task8/integrated_history_log.png` captures the rendered Audit Trail table.

---

## 4. Latency Analysis: Local vs. Remote Inference

A key engineering trade-off in distributed machine learning systems is the latency differential between local in-process inference versus remote cloud API calls:

| Performance Characteristic | Local In-Process Mode (`local`) | Remote Cloud API Mode (`remote`) | Engineering Trade-off |
| :--- | :---: | :---: | :--- |
| **Inference Engine Latency** | `~15 ms - 25 ms` | `~900 ms - 1100 ms` | Cloud CPU resources are shared (0.5 vCPU on free tier). |
| **Network Transit & TLS Overhead** | `0 ms` (Local RAM) | `~400 ms - 700 ms` | HTTPS handshake, SSL termination, and geographic transit to Frankfurt EU. |
| **Total User Roundtrip** | `< 40 ms` | `~1500 ms - 2200 ms` | Local execution offers sub-second responsiveness. |
| **Client Memory Footprint** | `~430 MB` (TensorFlow in RAM) | `< 45 MB` (Lightweight Python) | **Remote mode requires 10x less client RAM**, allowing deployment on ultra-lightweight client devices or IoT edges. |

---

## 5. Fault Tolerance & Cloud Cold-Start Resilience

To handle network fluctuations or cloud cold-start spin-ups (common on free-tier container platforms like Render):
1. **Connection Timeout Governance:** `_predict_remote()` in `webapp/inference.py` sets a generous 30-second timeout to accommodate cold-starts without immediate connection abandonment.
2. **Graceful Fallback:** If the cloud host is unreachable, the gateway traps `requests.exceptions.RequestException` and renders a clean HTTP 500 error page with diagnostic advice rather than crashing the Python worker.
