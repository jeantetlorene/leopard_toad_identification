import os
import torch

# Base Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
DATA_DIR = os.path.join(EVAL_DIR, "data")
CONSENSUS_DIR = os.path.join(EVAL_DIR, "consensus_predictions")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
FILES_DIR = os.path.join(RESULTS_DIR, "files")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Path for preprocessed CLAHE images
CLAHE_PREPROCESSED_DIR = os.environ.get(
    "CLAHE_PREPROCESSED_DIR",
    "/media/lorene/Project-drive/shared_leopard_toad_2/shared_leopard_toad_clahe",
)


# Model Roots
MODEL_ROOTS = {
    "yolo": os.path.join(BASE_DIR, "active learning", "yolo"),
    "yolo_clahe": os.path.join(BASE_DIR, "active learning", "yolo_clahe"),
    "rtdetr": os.path.join(BASE_DIR, "active learning", "rtdetr"),
    "rtdetr_clahe": os.path.join(BASE_DIR, "active learning", "rtdetr_clahe"),
    "faster_rcnn": os.path.join(BASE_DIR, "active learning", "faster_rcnn"),
    "faster_rcnn_clahe": os.path.join(BASE_DIR, "active learning", "faster_rcnn_clahe"),
}

# Inference Settings
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 512
FASTER_RCNN_SUB_BATCH_SIZE = 128
MIN_CONF_THRESHOLD = 0.01
CONF_THRESHOLDS = [round(x, 2) for x in torch.linspace(0.01, 0.95, 30).tolist()]

# Dataset Mappings
MAPPING_PATH = os.path.join(DATA_DIR, "image_mapping.csv")

# Mapping CSV files to camera IDs and local GT directories
DATASETS = {
    "test": {
        "images_dir": os.path.join(DATA_DIR, "test", "images"),
        "labels_dir": os.path.join(DATA_DIR, "test", "labels"),
        "camera": "5Z",
    },
    "val": {
        "images_dir": os.path.join(DATA_DIR, "val", "images"),
        "labels_dir": os.path.join(DATA_DIR, "val", "labels"),
        "camera": "4R",
    },
}

# Class Mapping (consistent across all models)
CLASSES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}
