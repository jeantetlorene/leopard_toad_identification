import os

MODE = os.environ.get("AL_MODE", "pretrained")

# Data Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
YOLO_DIR = os.path.join(BASE_DIR, "active learning", "yolo_clahe")

PRETRAINED_WEIGHTS = os.path.join(
    BASE_DIR, "pretraining", "runs", "detect", "yolo_model", "weights", "best.pt"
)
# Make sure to point to original yolo weights if needed, or if it was copied
SCRATCH_WEIGHTS = os.path.join(BASE_DIR, "active learning", "yolo", "yolo26m.pt")

# AL Configuration
BUDGET_PER_CYCLE = 100
CONF_THRESHOLD = 0.01  # Lower threshold for uncertainty
IMG_SIZE = 640
TRAIN_BATCH_SIZE = 32
INFER_BATCH_SIZE = 512
DEVICE = "0"  # GPU device

EXCLUDED_CAMERAS = ["4R", "5Z"]

# AL specific paths
AL_STATE_JSON = os.path.join(YOLO_DIR, f"al_state_{MODE}.json")
