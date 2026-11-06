# Task 6.3: Multi-Service Docker Compose Stack Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Compose File:** [`docker-compose.yml`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/docker-compose.yml)  
**Services:** `web` (VisionGuard Flask), `db` (PostgreSQL 15-alpine)  
**Network:** `visionguard-net` (internal bridge)  
**Named Volume:** `postgres_data`  

---

## 1. Multi-Service Architecture & Internal Networking

The VisionGuard platform is containerized as a decoupled, multi-service architecture using Docker Compose:

```
+---------------------------------------------------------------------------------------------------------+
|                                  PHYSICAL HOST (macOS / Developer Machine)                               |
|                                                                                                         |
|   HTTP Request (Browser / API Client)                                                                   |
|   http://localhost:5001 ------------------------------------+                                           |
|                                                             |                                           |
|   Direct DB Access Attempt                                  |                                           |
|   tcp://localhost:5432 ---- [CONNECTION REFUSED]            |                                           |
|                             (Port 5432 Closed to Host)      |                                           |
+-------------------------------------------------------------|-------------------------------------------+
                                                              |
                                                              v Port Mapping 5001:5000
+---------------------------------------------------------------------------------------------------------+
|                        DOCKER COMPOSE INTERNAL BRIDGE NETWORK ('visionguard-net')                       |
|                                                                                                         |
|    +------------------------------------------+        +------------------------------------------+     |
|    |               SERVICE: web               |        |               SERVICE: db                |     |
|    |      (visionguard-web:latest)            |        |           (postgres:15-alpine)           |     |
|    |------------------------------------------|        |------------------------------------------|     |
|    | - Flask Web Application                  |        | - PostgreSQL 15 Relational Database      |     |
|    | - Gunicorn WSGI Server                   |        | - Internal Port: 5432 (expose only)      |     |
|    | - Non-root user: appuser                 |        | - External Port: NONE (Closed to Host)   |     |
|    | - Model: models/champion.keras           |        | - Native Healthcheck: pg_isready         |     |
|    | - Healthcheck: curl /health              |        +--------------------+---------------------+     |
|    +--------------------+---------------------+                             |                          |
|                         |                                                    |                          |
|                         +--- Resolves 'db:5432' via Compose DNS ------------+                          |
|                              DATABASE_URL=postgresql://...                                              |
+------------------------------------------------------------------------------|--------------------------+
                                                                               |
                                                                               v Mount
                                                              +----------------------------------+
                                                              |   NAMED VOLUME: postgres_data    |
                                                              |  (/var/lib/postgresql/data)      |
                                                              +----------------------------------+
```

---

## 2. Key Architecture & Design Choices (Addressing Rubric Prompts)

### 2.1 Internal DNS Resolution by Service Name
Services communicate strictly over the internal isolated bridge network (`visionguard-net`) using automatic Docker DNS resolution:
- The `web` service targets `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}`.
- Docker resolves the hostname `db` directly to the database container's private internal IP address, eliminating hardcoded IP dependencies.

### 2.2 Service Dependency & Readiness Ordering (`depends_on`)
A common failure mode in multi-container stacks is "startup race condition", where the web application boots before the database engine is accepting TCP sockets.
In `docker-compose.yml`, this is resolved via:
```yaml
depends_on:
  db:
    condition: service_healthy
```
The `web` container startup is strictly blocked until PostgreSQL passes its internal readiness check (`pg_isready -U ... -d ...`).

### 2.3 Proof that Database Port 5432 is Closed from Outside
To enforce enterprise defense-in-depth, the database service uses the `expose:` directive rather than `ports:`:
- `expose: - "5432"` allows other containers on `visionguard-net` to reach port 5432.
- No host port mapping is declared.

**Verification Probe:**
```bash
$ nc -zv 127.0.0.1 5432
nc: connectx to 127.0.0.1 port 5432 (tcp) failed: Connection refused
```
Any attempt to query PostgreSQL from the host machine or outside network is actively refused, preventing database compromise.

### 2.4 Model File Delivery Strategy: Built-in Packaging vs Volume Mount Justification
**Decision:** We adopted a hybrid architecture: **Built-in by default, with optional read-only volume mounting**.
1. **Built-In Image Packaging (`COPY models/champion.keras ./models/champion.keras`):**
   - *Justification:* In continuous delivery and cloud deployments, shipping a self-contained image guarantees that the exact model weights tested in staging are executed in production. It eliminates external volume mounting dependencies on cloud providers (e.g. AWS ECS, Render, Vercel).
2. **Volume Mount (`./models:/app/models:ro`):**
   - *Justification:* During local development, mounting the models folder read-only allows engineers to hot-swap candidate models from DagsHub without rebuilding the entire 1.94GB Docker image.

### 2.5 Secrets Management & `.env` Hygiene
- Passwords (`POSTGRES_PASSWORD`), usernames, and secret keys are dynamically injected via environment variables from an untracked `.env` file (verified gitignored).
- A documented template [`.env.example`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/.env.example) is committed to the repository with clear placeholder keys, guaranteeing zero credentials leak into version control.

---

## 3. Service Health & Execution Proofs

### Compose Service Table (`docker compose ps`):
```text
time="2026-11-07T04:15:05+05:00" level=warning msg="/Users/shoaib/Desktop/University/7th Semester/MLOPS/A2/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion"
NAME              IMAGE                    COMMAND                  SERVICE   CREATED          STATUS                    PORTS
visionguard_db    postgres:15-alpine       "docker-entrypoint.s…"   db        25 seconds ago   Up 24 seconds (healthy)   5432/tcp
visionguard_web   visionguard-web:latest   "gunicorn --bind 0.0…"   web       25 seconds ago   Up 19 seconds (healthy)   0.0.0.0:5001->5000/tcp, [::]:5001->5000/tcp
```

### PostgreSQL Live Data Persistence Proof:
```text
id |       filename        |  prediction  | confidence | inference_time_ms |         timestamp          
----+-----------------------+--------------+------------+-------------------+----------------------------
  2 | compose_test_ai.jpg   | AI-Generated |         50 |            129.24 | 2026-11-06 23:15:06.007265
  1 | compose_test_real.jpg | Real         |      83.97 |            134.18 | 2026-11-06 23:15:05.845577
(2 rows)
```

- **Compose File Link:** [`docker-compose.yml`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/docker-compose.yml)
- **Status Screenshot:** [`evidence/task6/compose_status.png`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task6/compose_status.png)
- **Raw Execution Log:** [`evidence/task6/compose_output.txt`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task6/compose_output.txt)
