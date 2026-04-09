import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_VIDEOS = int(os.getenv("MAX_UPLOAD_VIDEOS", "4"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
DEFAULT_LANE_CAPACITY = int(os.getenv("DEFAULT_LANE_CAPACITY", "20"))
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
