import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


APP_ENV = os.getenv("APP_ENV", "development")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = _as_int("PORT", 8000)
UVICORN_WORKERS = _as_int("UVICORN_WORKERS", 1)
UVICORN_RELOAD = _as_bool("UVICORN_RELOAD", APP_ENV == "development")

MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")
PRELOAD_MODELS = _as_bool("PRELOAD_MODELS", False)

CACHE_DIR = Path(os.getenv("APP_CACHE_DIR", BASE_DIR / ".cache"))
YOLO_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR", CACHE_DIR / "ultralytics"))
TORCH_HOME = Path(os.getenv("TORCH_HOME", CACHE_DIR / "torch"))

REPORT_PATH = Path(os.getenv("REPORT_PATH", BASE_DIR / "rapport_match.json"))
PROGRESS_PATH = Path(os.getenv("PROGRESS_PATH", BASE_DIR / "analyse_progress.json"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "video" / "uploads"))
INDEX_HTML_PATH = Path(
    os.getenv("INDEX_HTML_PATH", BASE_DIR / "services" / "s12_api" / "index.html")
)


def ensure_runtime_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def configure_runtime_environment() -> None:
    ensure_runtime_dirs()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


def get_model_device() -> str:
    if MODEL_DEVICE != "auto":
        return MODEL_DEVICE

    try:
        import torch
    except Exception:
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def runtime_summary() -> dict:
    device = get_model_device()
    cuda_available = False
    cuda_device_count = 0

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception:
        pass

    return {
        "app_env": APP_ENV,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cache_dir": str(CACHE_DIR),
        "yolo_config_dir": str(YOLO_CONFIG_DIR),
        "torch_home": str(TORCH_HOME),
    }
