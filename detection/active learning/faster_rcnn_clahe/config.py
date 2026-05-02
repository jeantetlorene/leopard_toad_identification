import os
import sys

# Inherit mostly from original faster rcnn config
MODE = os.environ.get("AL_MODE", "pretrained")

BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection"
FASTER_RCNN_DIR = os.path.join(BASE_DIR, "active learning", "faster_rcnn_clahe")

PRETRAINED_WEIGHTS = os.path.join(
    BASE_DIR,
    "pretraining",
    "runs",
    "faster_rcnn",
    "train_resnet50_1",
    "weights",
    "best.pt",
)
# Note: For Faster R-CNN, "scratch" uses FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# But if there's a specific file, we can use it. The original faster_rcnn doesn't define SCRATCH_WEIGHTS in its config, wait...
# Let me look back at my previous `find` or `list_dir` for faster_rcnn/config.py
# In `main_al_loop.py` for faster_rcnn, it uses `from config import PRETRAINED_WEIGHTS, SCRATCH_WEIGHTS`
# So we need to define it. We'll set it to "scratch" or an empty string since get_model handles from scratch.
SCRATCH_WEIGHTS = "scratch"

BUDGET_PER_CYCLE = 100
IMG_SIZE = 640
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 64
DEVICE = "cuda"

AL_STATE_JSON = os.path.join(FASTER_RCNN_DIR, f"al_state_{MODE}.json")
