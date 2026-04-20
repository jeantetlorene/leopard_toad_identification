import os

MODE = os.environ.get("AL_MODE", "pretrained")

# Data Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
YOLO_DIR = os.path.join(BASE_DIR, "active learning", "yolo")

PRETRAINED_WEIGHTS = os.path.join(
    BASE_DIR, "pretraining", "runs", "detect", "yolo_model", "weights", "best.pt"
)
SCRATCH_WEIGHTS = os.path.join(YOLO_DIR, "yolo26m.pt")

# AL Configuration
BUDGET_PER_CYCLE = 100
CONF_THRESHOLD = 0.01  # Lower threshold for uncertainty
IMG_SIZE = 640
TRAIN_BATCH_SIZE = 32
INFER_BATCH_SIZE = 512
DEVICE = "0"  # GPU device

EXCLUDED_CAMERAS = ["4R", "5Z"]

# Unlabeled Pool paths matching batch_inference.py
YEARS = {
    "2023": "/srv/shared_leopard_toad/2023",
    "2024": "/srv/shared_leopard_toad/2024",
    "2025": "/srv/shared_leopard_toad/2025/Documents",
}
FOLDERS = [
    "01.09.09.09.2025",
    "06.10-13.10.2025",
    "15.09-22.09.2025",
    "25.08-01.09.2025",
    "04.08.2025",
    "09.09.15.09.2025",
    "21-28.08.2025",
    "29.09.06.10.2025",
    "05.08.2025",
    "12.19.08.2025",
    "22.09-29.09.2025",
    "Cameras - AI Data",
    "Cameras-Master Data",
    "02.09.2024",
    "09.02.2024",
    "16.09.2024",
    "19.09.2024",
    "26.09.2024",
    "02.10.2024",
    "11.10.2024",
    "18.11.2024",
    "23.08.2024",
    "28.08.2024",
]

# AL specific paths

AL_STATE_JSON = os.path.join(YOLO_DIR, f"al_state_{MODE}.json")
