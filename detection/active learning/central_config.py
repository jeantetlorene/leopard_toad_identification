import os
import torch

# --- Directory Pathing Resolution ---
ACTIVE_LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(ACTIVE_LEARNING_DIR)
PROJECT_ROOT = os.path.dirname(DETECTION_DIR)

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
DEFAULT_OCCURRENCE_THRESHOLD = 15

# --- Class Definitions & Optimal Confidence Thresholds ---
CLASSES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}

# Optimal validation analytical thresholds based on F1-Score maximization
DETECTION_THRESHOLDS = {0: 0.2, 1: 0.2, 2: 0.7}

# --- Active Learning Curation Settings ---
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
BUDGET_ALLOCATION_WLT = 0.40  # 40% Western Leopard Toads
BUDGET_ALLOCATION_HARD_NEGS = 0.30  # 30% Stationary False Positives / Hard Negatives
BUDGET_ALLOCATION_OTHER_FAUNA = 0.30  # 30% Other Amphibians & Mammals

# Default total human annotation budget (n_clusters)
DEFAULT_CURATION_BUDGET = 100
