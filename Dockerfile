# ==============================================================================
# Production Dockerfile for VisionGuard Web Application
# Student Roll Number: 23L0710
# Standard Naming: 23L0710_A2_Ops
# ==============================================================================

# 1. Base Image: Minimal, secure, and lightweight Debian-based Python runtime
FROM python:3.10-slim

# 2. Environment Variables: Optimize Python execution and configure paths
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    KERAS_HOME=/tmp/.keras \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=5000 \
    HOST=0.0.0.0 \
    INFERENCE_MODE=local \
    MODEL_PATH=models/champion.keras \
    DECISION_THRESHOLD=0.23

# 3. System Dependencies: Install curl for health check and cleanup apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set application working directory
WORKDIR /app

# 5. LAYER CACHING OPTIMIZATION:
# Dependencies are copied and installed BEFORE application source code.
# Since dependencies change far less frequently than business logic or UI code,
# Docker reuses this cached intermediate layer across code edits, dramatically
# slashing rebuild times from minutes to seconds.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-docker.txt

# 6. Copy Application Code & Champion Model Artifact
COPY webapp/ ./webapp/
COPY src/ ./src/
COPY models/champion.keras ./models/champion.keras

# 7. SECURITY: Create non-root system group and user
# Running as root inside containers exposes the host to container breakout attacks.
# We create an unprivileged user 'appuser' belonging to 'appgroup'.
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /sbin/nologin -c "VisionGuard Service User" appuser && \
    mkdir -p /tmp/.keras /tmp/matplotlib && \
    chown -R appuser:appgroup /app /tmp/.keras /tmp/matplotlib

# Switch execution privileges to non-root user
USER appuser

# Expose web application port
EXPOSE 5000

# 8. CONTAINER HEALTH CHECK:
# Native Docker health-check polling the internal /health API endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# 9. Startup Command: Production-grade WSGI Server (Gunicorn)
# Uses 1 worker with 4 threads to minimize memory footprint (<500MB) for cloud free tiers (e.g. Render)
# while enabling concurrent request execution and dynamic cloud PORT binding.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 webapp.app:app

