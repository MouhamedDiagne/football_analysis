#!/usr/bin/env bash
set -euo pipefail

export APP_ENV="${APP_ENV:-production}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
export UVICORN_RELOAD="${UVICORN_RELOAD:-false}"
export APP_CACHE_DIR="${APP_CACHE_DIR:-/app/.cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${APP_CACHE_DIR}/ultralytics}"
export TORCH_HOME="${TORCH_HOME:-${APP_CACHE_DIR}/torch}"

mkdir -p "${APP_CACHE_DIR}" "${YOLO_CONFIG_DIR}" "${TORCH_HOME}" /app/video/uploads

if [[ "${PRELOAD_MODELS:-0}" == "1" ]]; then
  python download_models.py
fi

exec python -m uvicorn services.s12_api.api_service:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${UVICORN_WORKERS}"
