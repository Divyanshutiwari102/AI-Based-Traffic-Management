import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_VIDEOS = int(os.getenv("MAX_UPLOAD_VIDEOS", "4"))
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
LEGACY_YOLO_WEIGHTS = os.getenv("LEGACY_YOLO_WEIGHTS", "yolov4-tiny.weights")
LEGACY_YOLO_CONFIG = os.getenv("LEGACY_YOLO_CONFIG", "yolov4-tiny.cfg")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "classes.txt")
DEFAULT_LANE_CAPACITY = int(os.getenv("DEFAULT_LANE_CAPACITY", "20"))
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
MAX_TASK_WORKERS = int(os.getenv("MAX_TASK_WORKERS", "2"))
DEFAULT_MAX_VIDEO_FRAMES = int(os.getenv("DEFAULT_MAX_VIDEO_FRAMES", "120"))

GA_POP_SIZE = int(os.getenv("GA_POP_SIZE", "250"))
GA_MAX_ITER = int(os.getenv("GA_MAX_ITER", "20"))
GA_GREEN_MIN = int(os.getenv("GA_GREEN_MIN", "10"))
GA_GREEN_MAX = int(os.getenv("GA_GREEN_MAX", "60"))
GA_CYCLE_TIME = int(os.getenv("GA_CYCLE_TIME", "148"))
GA_MUTATION_RATE = float(os.getenv("GA_MUTATION_RATE", "0.02"))
GA_BETA = float(os.getenv("GA_BETA", "8"))

EPSILON = float(os.getenv("NUMERIC_EPSILON", "1e-9"))
AWT_WEIGHT = float(os.getenv("AWT_WEIGHT", "-0.6"))
MAX_QUEUE_WEIGHT = float(os.getenv("MAX_QUEUE_WEIGHT", "-0.3"))
JFI_WEIGHT = float(os.getenv("JFI_WEIGHT", "0.1"))
