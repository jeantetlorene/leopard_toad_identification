import os
import torch

# --- Directory Pathing Resolution ---
ACTIVE_LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(ACTIVE_LEARNING_DIR)
PROJECT_ROOT = os.path.dirname(DETECTION_DIR)

# Path for preprocessed CLAHE images
CLAHE_PREPROCESSED_DIR = os.environ.get(
    "CLAHE_PREPROCESSED_DIR",
    os.path.normpath(
        os.path.join(PROJECT_ROOT, "dataset", "shared_leopard_toad_clahe")
    ),
)

# --- Default Pipeline File Paths ---
DEFAULT_MODEL_PATH = os.path.join(
    DETECTION_DIR,
    "active learning",
    "rtdetr_clahe",
    "runs",
    "cycle_2_pretrained_phase2",
    "weights",
    "best.pt",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    DETECTION_DIR, "results", "detect_rtdetr_cycle2_clahe_pretrained"
)

# --- Default Inference Hyperparameters ---
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 128
DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"

# --- Bounding Box Spatial Filtering Settings ---
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_OCCURRENCE_THRESHOLD = 25

# --- Class Definitions & Optimal Confidence Thresholds ---
# The user can customize these to select which classes to train on. Set to None to default.
TARGET_CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
CLASS_MAPPING = None

# Resolve global CLASSES dictionary based on target classes
if TARGET_CLASSES is not None:
    CLASSES = {i: name for i, name in enumerate(TARGET_CLASSES)}
else:
    CLASSES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}

# Optimal validation analytical thresholds based on F1-Score maximization
ORIGINAL_DETECTION_THRESHOLDS = {
    "Other_Amphibian": 0.2,
    "Small_Mammal": 0.2,
    "Western_Leopard_Toad": 0.7,
}

# Resolve DETECTION_THRESHOLDS dynamically for target classes
DETECTION_THRESHOLDS = {}
for i, name in CLASSES.items():
    if name in ORIGINAL_DETECTION_THRESHOLDS:
        DETECTION_THRESHOLDS[i] = ORIGINAL_DETECTION_THRESHOLDS[name]
    else:
        DETECTION_THRESHOLDS[i] = 0.25  # Generic default threshold

# --- Active Learning Curation Settings ---
# Primary target class for active curation focus (e.g. "Western_Leopard_Toad" or any other main class)
CURATION_TARGET_CLASS = "Western_Leopard_Toad"

# Primary domain-pretrained ResNet50 model weights from Faster R-CNN pretraining
DEFAULT_PRETRAINED_RESNET_WEIGHTS = os.path.join(
    DETECTION_DIR,
    "pretraining",
    "runs",
    "faster_rcnn",
    "train_resnet50_1",
    "weights",
    "best.pt",
)

# Fallback path if the primary run directory differs
FALLBACK_PRETRAINED_RESNET_WEIGHTS = os.path.join(
    DETECTION_DIR,
    "pretraining",
    "runs",
    "faster_rcnn",
    "train_resnet50",
    "weights",
    "best.pt",
)

DEFAULT_CURATION_CONF_THRESHOLD = 0.85

# Proportional budget split for active learning curation (must sum to 1.0)
BUDGET_ALLOCATION_TARGET = 0.40  # Proportion for the primary curation target class
BUDGET_ALLOCATION_HARD_NEGS = (
    0.30  # Proportion for stationary false positives / hard negatives
)
BUDGET_ALLOCATION_OTHER_CLASSES = (
    0.30  # Proportion for all other active support classes
)

# Default total human annotation budget (n_clusters)
DEFAULT_CURATION_BUDGET = 100

# --- Model-Specific Training Configurations ---
YOLO_TRAIN_CONFIG = {
    "pretrained": {
        "phase1": {"epochs": 100, "patience": 25, "batch_size": 32, "freeze": 15},
        "phase2": {"epochs": 100, "patience": 15, "batch_size": 32, "freeze": 0},
    },
    "scratch": {"epochs": 60, "patience": 50, "batch_size": 32, "freeze": 0},
}

RTDETR_TRAIN_CONFIG = {
    "pretrained": {
        "phase1": {"epochs": 100, "patience": 25, "batch_size": 32, "freeze": 15},
        "phase2": {"epochs": 100, "patience": 15, "batch_size": 32, "freeze": 0},
    },
    "scratch": {"epochs": 300, "patience": 50, "batch_size": 32, "freeze": 0},
}

FASTER_RCNN_TRAIN_CONFIG = {
    "pretrained": {
        "phase1": {
            "epochs": 100,
            "patience": 25,
            "batch_size": 16,
            "freeze_backbone": True,
        },
        "phase2": {
            "epochs": 100,
            "patience": 15,
            "batch_size": 16,
            "freeze_backbone": False,
        },
    },
    "scratch": {
        "epochs": 100,
        "patience": 20,
        "batch_size": 16,
        "freeze_backbone": False,
    },
}
