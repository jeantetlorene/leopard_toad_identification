import os

MODE = os.environ.get("AL_MODE", "pretrained")

# Data Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
RTDETR_DIR = os.path.join(BASE_DIR, "active learning", "rtdetr_clahe")

PRETRAINED_WEIGHTS = os.path.join(
    BASE_DIR, "pretraining", "runs", "detect", "rtdetr_finetuning", "weights", "best.pt"
)
SCRATCH_WEIGHTS = os.path.join(BASE_DIR, "active learning", "rtdetr", "rtdetr-l.pt")

# AL Configuration
BUDGET_PER_CYCLE = 100
CONF_THRESHOLD = 0.01  # Lower threshold for uncertainty
IMG_SIZE = 640
TRAIN_BATCH_SIZE = 32  # Keep conservative for training backpropagation memory
INFER_BATCH_SIZE = 512  # Maximize huge 48GB VRAM for forward pass only
DEVICE = "0"  # GPU device

EXCLUDED_CAMERAS = ["4R", "5Z"]

# AL specific paths
AL_STATE_JSON = os.path.join(RTDETR_DIR, f"al_state_{MODE}.json")
