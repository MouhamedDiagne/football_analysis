FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    UVICORN_WORKERS=1 \
    UVICORN_RELOAD=false \
    APP_CACHE_DIR=/app/.cache \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    TORCH_HOME=/app/.cache/torch \
    MODEL_DEVICE=auto \
    PRELOAD_MODELS=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.app.txt /app/requirements.app.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      torch==2.5.1 \
      torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install -r /app/requirements.app.txt

COPY . /app

RUN chmod +x /app/scripts/start_api.sh && \
    mkdir -p /app/.cache/ultralytics /app/.cache/torch /app/video/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl --fail http://127.0.0.1:8000/health || exit 1

CMD ["/app/scripts/start_api.sh"]
