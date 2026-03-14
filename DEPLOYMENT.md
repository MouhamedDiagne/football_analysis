# Deployment Guide

## Target

This project is prepared for GPU deployment on RunPod using a CUDA-enabled Docker image and GitHub Actions for image publishing.

## What Was Added

- `Dockerfile`: CUDA runtime container for Linux GPU hosts.
- `requirements.app.txt`: app dependencies excluding Torch/Torchvision, which are installed as CUDA wheels during image build.
- `scripts/start_api.sh`: production startup script.
- `scripts/docker-smoke.sh`: local GPU container smoke test helper.
- `scripts/bootstrap_repo.sh`: Git bootstrap helper before pushing to GitHub.
- `scripts/runpod-env.example`: RunPod-oriented environment variable set.
- `.env.example`: runtime environment variables.
- `.github/workflows/ci-cd.yml`: syntax checks plus Docker build/push to GHCR.

## Local Docker Smoke Test

If Docker and the NVIDIA container toolkit are available locally:

```bash
chmod +x scripts/docker-smoke.sh
./scripts/docker-smoke.sh football-analysis:local
```

Or manually:

```bash
docker build -t football-analysis:local .
docker run --rm --gpus all -p 8000:8000 football-analysis:local
```

Then check:

```bash
curl http://127.0.0.1:8000/health
```

## Recommended Runtime Settings

Set these environment variables in RunPod:

```bash
APP_ENV=production
HOST=0.0.0.0
PORT=8000
UVICORN_WORKERS=1
MODEL_DEVICE=auto
PRELOAD_MODELS=0
APP_CACHE_DIR=/app/.cache
YOLO_CONFIG_DIR=/app/.cache/ultralytics
TORCH_HOME=/app/.cache/torch
```

`MODEL_DEVICE=auto` will choose `cuda:0` when a GPU is present.

## GitHub Container Publishing

Every push to `main` builds and publishes:

```text
ghcr.io/<your-github-user-or-org>/football-analysis:latest
ghcr.io/<your-github-user-or-org>/football-analysis:sha-<commit>
```

## RunPod Setup

1. Push this repository to GitHub.
2. Enable GitHub Actions.
3. Go to GitHub Packages or GHCR and confirm the image is published.
4. In RunPod, create a pod or template using the image:
   `ghcr.io/<your-github-user-or-org>/football-analysis:latest`
5. Expose port `8000`.
6. Attach a persistent volume if you want uploads, reports, and model caches to survive restarts.
7. Add the environment variables from `.env.example`.

### Suggested RunPod Template

- Container image: `ghcr.io/<your-github-user-or-org>/football-analysis:latest`
- Container disk: `30 GB` minimum
- Volume disk: `50 GB` recommended if you keep uploads and cached weights
- Exposed HTTP port: `8000`
- GPU: any CUDA-compatible NVIDIA GPU, ideally `A10`, `A40`, `L4`, or better
- Start command:

```bash
/app/scripts/start_api.sh
```

### Suggested RunPod Environment Variables

Use [`scripts/runpod-env.example`](c:/Users/Dell/Desktop/AI_Jobs/football_analysis/scripts/runpod-env.example) as the source of truth.

Copy/paste values:

```bash
APP_ENV=production
HOST=0.0.0.0
PORT=8000
UVICORN_WORKERS=1
UVICORN_RELOAD=false
MODEL_DEVICE=auto
PRELOAD_MODELS=0
APP_CACHE_DIR=/runpod-volume/cache
YOLO_CONFIG_DIR=/runpod-volume/cache/ultralytics
TORCH_HOME=/runpod-volume/cache/torch
UPLOAD_DIR=/runpod-volume/uploads
REPORT_PATH=/runpod-volume/rapport_match.json
PROGRESS_PATH=/runpod-volume/analyse_progress.json
```

### GitHub Bootstrap

If this repo is not on GitHub yet:

```bash
chmod +x scripts/bootstrap_repo.sh
./scripts/bootstrap_repo.sh
```

Then add your remote, commit, and push.

## Suggested Persistent Paths

Mount a volume and point these there:

```bash
APP_CACHE_DIR=/runpod-volume/cache
YOLO_CONFIG_DIR=/runpod-volume/cache/ultralytics
TORCH_HOME=/runpod-volume/cache/torch
UPLOAD_DIR=/runpod-volume/uploads
REPORT_PATH=/runpod-volume/rapport_match.json
PROGRESS_PATH=/runpod-volume/analyse_progress.json
```

## Notes

- The first analysis run may download ResNet weights if they are not baked into the image cache.
- For faster startup, set `PRELOAD_MODELS=1` during image bake or container startup.
- If you want zero-downtime redeploys on RunPod, use immutable SHA tags instead of `latest`.
- Keep `UVICORN_WORKERS=1` unless you redesign the background-analysis state handling, because the current app stores active-process state in memory.
