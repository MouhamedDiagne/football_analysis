# Football Analysis

Football video analysis platform with:

- FastAPI web app
- background video processing pipeline
- YOLO-based player and pose detection
- ByteTrack-based tracking
- team classification and match report generation

## Local Python

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn services.s12_api.api_service:app --host 127.0.0.1 --port 8000
```

## Container Smoke Test

Linux/macOS:

```bash
chmod +x scripts/docker-smoke.sh
./scripts/docker-smoke.sh football-analysis:local
```

Manual Docker run:

```bash
docker build -t football-analysis:local .
docker run --rm --gpus all -p 8000:8000 football-analysis:local
```

## Cloud Deployment

Use the GitHub Actions workflow to publish the image, then deploy it on RunPod.

See [`DEPLOYMENT.md`](DEPLOYMENT.md).
