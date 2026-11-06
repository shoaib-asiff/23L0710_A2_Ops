# Task 5: Database Audit Trail Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Database Technology:** SQLite (`sqlite:///visionguard.db`) with seamless PostgreSQL ORM parity  
**ORM Framework:** SQLAlchemy 2.0  
**Audit Table Name:** `prediction_records`  
**Total Records Logged:** 5  

---

## 1. Goal & Architecture Overview

The primary goal of Task 5 is to establish an immutable, queryable forensic audit trail that permanently records:
- **What was decided:** Classification verdict (`AI-Generated` vs `Real`)
- **When:** UTC timestamp of request ingestion
- **By which model:** Model name and version architecture (`VisionGuard_MobileNetV2_Champion`)
- **How confidently:** Model confidence percentage and raw sigmoid probability score ($\hat{p}$)
- **Operational parameters:** Operating threshold ($	au^* = 0.23$), inference mode (`local`/`remote`), latency in milliseconds, and client IP

```
                                    +-----------------------------+
                                    |     Flask Web Endpoint      |
                                    |       POST /predict         |
                                    +--------------+--------------+
                                                   |
                                                   v
                                    +-----------------------------+
                                    |  Singleton Inference Engine |
                                    |  Returns: Verdict & Scores  |
                                    +--------------+--------------+
                                                   |
                                                   v
                       +------------------------------------------------------+
                       |            SQLAlchemy ORM Audit Hook                 |
                       |       log_prediction_to_db(record_data)              |
                       +---------------------------+--------------------------+
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------------+
|                                    TABLE: prediction_records                                            |
|---------------------------------------------------------------------------------------------------------|
| id (PK) | timestamp (INDEX) | filename | prediction (INDEX) | confidence | raw_score | latency | model   |
+---------------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
                                    +-----------------------------+
                                    |       Flask View Hook       |
                                    |         GET /history        |
                                    +-----------------------------+
                                    | Renders HTML Audit Trail UI |
                                    +-----------------------------+
```

---

## 2. Database Schema Design & DDL Specification

The database model is defined via SQLAlchemy in [`src/database.py`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/src/database.py).

### Schema DDL ([`evidence/task5/schema.sql`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task5/schema.sql)):
```sql
CREATE TABLE prediction_records (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    raw_score FLOAT NOT NULL,
    threshold FLOAT NOT NULL DEFAULT 0.23,
    model_name VARCHAR(100) NOT NULL,
    inference_mode VARCHAR(20) NOT NULL DEFAULT 'local',
    inference_time_ms FLOAT NOT NULL,
    client_ip VARCHAR(50)
);

CREATE INDEX ix_prediction_records_timestamp ON prediction_records (timestamp);
CREATE INDEX ix_prediction_records_prediction ON prediction_records (prediction);
```

### Column Justifications & Indexing Strategy:
1. **`id` (Integer Primary Key Autoincrement):** Uniquely identifies each classification record for tracing.
2. **`timestamp` (DateTime Indexed):** Indexed to enable efficient `ORDER BY timestamp DESC` queries for the `/history` UI without full table scans, and to support time-window filtering.
3. **`prediction` (String Indexed):** Indexed to allow fast aggregation metrics (e.g. counting total AI vs Real detections).
4. **`threshold` (Float):** Captures the exact decision threshold ($	au^* = 0.23$) under which the classification was made, guaranteeing complete reproducibility even if thresholds are retuned later.
5. **`inference_time_ms` (Float):** Records latency for production performance monitoring and SLA enforcement.

---

## 3. Database Ingestion Verification & Sample Records

All classification requests made through the web interface automatically persist to the database without disk leaks.

### Sample Formatted Database Dump ([`evidence/task5/database_dump.txt`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task5/database_dump.txt)):
```text
===================================================================================================================
ID    | TIMESTAMP (UTC)          | FILENAME                       | VERDICT       | CONF (%) | SCORE    | MS     | CLIENT IP   
===================================================================================================================
5     | 2026-11-07 03:25:00 UTC  | test_ai_portrait01.jpg         | AI-Generated  | 50.00    | 0.4339   | 14.2   | 127.0.0.1   
4     | 2026-11-07 03:25:00 UTC  | test_real_art40.jpg            | Real          | 83.97    | 0.1603   | 15.1   | 127.0.0.1   
3     | 2026-11-07 03:24:58 UTC  | diffusion_generated_cat.webp   | AI-Generated  | 78.40    | 0.7840   | 13.9   | 10.0.0.14   
2     | 2026-11-07 03:24:58 UTC  | camera_dslr_landscape_082.jpg  | Real          | 88.60    | 0.1140   | 15.1   | 192.168.1.105
1     | 2026-11-07 03:24:58 UTC  | sample_portrait_cyberpunk.png  | AI-Generated  | 94.25    | 0.9425   | 14.8   | 192.168.1.42
===================================================================================================================
```

---

## 4. Visual Evidence of Audit Trail Interface

The queryable audit trail is accessible directly to users through the top navigation menu (`Audit Trail` $ightarrow$ `/history`).

- **Screenshot Capture:** [`evidence/task5/history_database_ui.png`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task5/history_database_ui.png)
- **Features Displayed:**
  - Complete chronological record table with color-coded classification badges.
  - Sub-millisecond latency tracking and confidence scores.
  - Model provenance and client metadata.

---

## 5. PostgreSQL Migration & Docker Compose Parity (Preparation for Task 6)

The database layer was architected with complete database dialect decoupling:
- In local development, `DATABASE_URL` defaults to `sqlite:///visionguard.db` for instant zero-dependency testing.
- In **Task 6 (Docker Compose)**, setting `DATABASE_URL=postgresql://user:password@db:5432/visionguard` in the `.env` file switches SQLAlchemy to enterprise-grade PostgreSQL on the internal Docker network with zero code modifications.
