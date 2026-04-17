import os
import sys

# Data Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
PRETRAINING_DIR = os.path.join(BASE_DIR, "pretraining")

# Import native dataset logic to perfectly replicate the fine-tuner
sys.path.append(PRETRAINING_DIR)
FASTER_RCNN_DIR = os.path.join(BASE_DIR, "active learning", "faster_rcnn")
DATASET_DIR = os.path.join(BASE_DIR, "active learning", "data", "detect_1")
TRAIN_IMAGES_DIR = os.path.join(
    BASE_DIR, "active learning", "data", "detect_1", "train", "images"
)
PRETRAINED_WEIGHTS = os.path.join(
    BASE_DIR,
    "pretraining",
    "runs",
    "faster_rcnn",
    "train_resnet50_1",
    "weights",
    "best.pt",
)
SCRATCH_WEIGHTS = "scratch"

# AL Configuration
BUDGET_PER_CYCLE = 100
CONF_THRESHOLD = 0.05  # Lower threshold for uncertainty
IMG_SIZE = 640
TRAIN_BATCH_SIZE = 32  # Keep conservative for training backpropagation memory
INFER_BATCH_SIZE = 512  # Maximize huge 48GB VRAM for forward pass only
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
CANDIDATE_OUTPUT_CSV = os.path.join(FASTER_RCNN_DIR, "al_query_candidates.csv")
AL_STATE_JSON = os.path.join(FASTER_RCNN_DIR, "al_state.json")
