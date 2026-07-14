# ============================================================
# Face Detection System — Docker Image
# ============================================================
# Build:  docker build -f docker/Dockerfile -t face-detect .
# Run (API):
#   docker run --rm -p 8080:8080 face-detect python main.py --mode api
# Run (batch):
#   docker run --rm \
#     -v /host/images:/app/input \
#     -v /host/output:/app/output \
#     face-detect python main.py --mode batch --source /app/input
# ============================================================

FROM python:3.11-slim

# --- System deps for OpenCV headless ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# --- Download model on build so runtime starts fast ---
RUN python - <<'EOF'
from src.detectors.dnn_detector import DNNDetector
DNNDetector(model_dir="models", download=True).close()
print("Model cached.")
EOF

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# Default: run API server
CMD ["python", "main.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8080"]
