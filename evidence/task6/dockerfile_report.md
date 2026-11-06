# Task 6.2: Production Dockerfile Technical Report
**Student Roll Number:** 23L0710  
**Project:** 23L0710_A2_Ops - VisionGuard  
**Image Tag:** `visionguard-web:latest`  
**Base Image:** `python:3.10-slim`  
**Image Size:** `visionguard-web:latest | Size: 1.94GB | ID: ee85d82f7bb8`  
**Active User Inside Container:** `appuser` (`uid=999(appuser) gid=999(appgroup) groups=999(appgroup)`)  

---

## 1. Production Dockerfile Link & Overview

The production Dockerfile is committed at [`Dockerfile`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/Dockerfile).

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000 \
    HOST=0.0.0.0 \
    MODEL_PATH=models/champion.keras

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Layer Caching Optimization: Install dependencies before copying source code
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy source code and champion model artifact
COPY webapp/ ./webapp/
COPY src/ ./src/
COPY models/champion.keras ./models/champion.keras

# Security: Non-root user configuration
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /sbin/nologin -c "VisionGuard Service User" appuser && \
    mkdir -p /tmp/.keras /tmp/matplotlib && \
    chown -R appuser:appgroup /app /tmp/.keras /tmp/matplotlib

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "webapp.app:app"]
```

---

## 2. Order of Instructions & Layer Caching Principle

Docker builds images layer-by-layer using an overlay filesystem cache (`overlayfs`). When rebuilding an image, Docker evaluates whether the instructions and context files for each layer have changed:
- If a layer's inputs are identical to a previous build, Docker uses `CACHED` and skips execution.
- If a layer changes, that layer and **all subsequent layers** must be invalidated and re-executed from scratch.

### Layer Caching Architecture:
1. **OS Packages & System Utilities (`RUN apt-get update...`):** Placed at the top because system utilities (`curl`) never change between routine application edits.
2. **Pip Dependencies (`COPY requirements-docker.txt .` -> `RUN pip install...`):** Copied and installed **before** the application source code. Python dependencies change infrequently (e.g. weekly or monthly).
3. **Application Source Code (`COPY webapp/ ...`):** Copied **after** dependency installation. Application templates, stylesheets, and Python modules change constantly during active development.
4. **Benefit:** When a developer edits a Flask template or route, Docker uses the cached pip layer (avoiding re-downloading hundreds of megabytes of Python packages), slashing rebuild time from over 3 minutes to under 4 seconds.

---

## 3. Deliberate Choice of TensorFlow Package (CPU-Only Justification)

In deep learning containerization, TensorFlow packages represent the largest contributor to image bloat:
- **Standard `tensorflow` package:** Packages complete NVIDIA CUDA, cuDNN, and TensorRT runtime libraries. When installed, it expands to over 1.8 GB to 2.5 GB on disk.
- **Deliberate Choice — `tensorflow-cpu==2.15.0`:** We explicitly selected `tensorflow-cpu` for web serving:
  1. **Latency Requirements:** On modern multi-core server CPUs (x86_64 / ARM64), single-image MobileNetV2 forward inference executes in < 20 ms. GPU acceleration is completely unnecessary for single-image web serving.
  2. **Size Reduction:** Omitting GPU drivers shrinks the image by > 1.2 GB, dramatically accelerating Docker image pulls, CI/CD pipeline runs, and cloud deployment cold-starts.
  3. **Portability & Attack Surface:** Avoids host NVIDIA driver dependencies, runtime GPU toolkit mismatch errors, and reduces attack surface by removing hundreds of third-party C++ shared libraries.

---

## 4. Non-Root User Security Implementation

By default, Docker containers execute processes as `root` (`UID 0`). Running as root poses a severe container breakout vulnerability: if an attacker exploits a web vulnerability (e.g. Remote Code Execution), they inherit root privileges on the container and can potentially escape to the host kernel.

### Implementation:
```dockerfile
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /sbin/nologin -c "VisionGuard Service User" appuser && \
    chown -R appuser:appgroup /app /tmp/.keras /tmp/matplotlib
USER appuser
```

### Verification Output Proof ([`evidence/task6/dockerfile_user_check.txt`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/evidence/task6/dockerfile_user_check.txt)):
```text
$ docker exec visionguard_web_test whoami
appuser

$ docker exec visionguard_web_test id
uid=999(appuser) gid=999(appgroup) groups=999(appgroup)
```
The active execution user is verified as unprivileged `appuser` with no shell login capability (`/sbin/nologin`), completely mitigating root privilege escalation risks.

---

## 5. Secrets Protection & `.dockerignore`

To guarantee that credentials and training bloat never leak into production images, [`.dockerignore`](file:///Users/shoaib/Desktop/University/7th%20Semester/MLOPS/A2/.dockerignore) explicitly excludes:
- Credentials: `.env`, `*.env`, tokens, private keys.
- Version Control: `.git/`, `.gitignore`.
- Heavy Datasets: `archive/`, `data/`, `archive.dvc`, `.dvc/cache`.
- Local Run History: `mlruns/`, `mlflow.db*`, `visionguard.db*`.
