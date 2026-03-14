#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-football-analysis:local}"

docker build -t "${IMAGE_TAG}" .

docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e APP_ENV=production \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e MODEL_DEVICE=auto \
  -e PRELOAD_MODELS=0 \
  "${IMAGE_TAG}"
