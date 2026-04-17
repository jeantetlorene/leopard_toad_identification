import os
from ultralytics import YOLO
from config import DATASET_YAML, YOLO_DIR, BATCH_SIZE, IMG_SIZE, DEVICE


def train_phase_1(model_weights, run_name, freeze=15, epochs=100, patience=15):
    """
    Phase 1: Freeze the backbone and train the detection head until convergence.
    """
    model = YOLO(model_weights)
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        patience=patience,  # Stop when metrics converge
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_phase1",
        freeze=freeze,
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_phase1", "weights", "best.pt")


def train_phase_2(model_weights, run_name, epochs=30):
    """
    Phase 2: Unfreeze the entire network so deeper layers can adapt to specific noise.
    """
    model = YOLO(model_weights)
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_phase2",
        freeze=0,
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_phase2", "weights", "best.pt")


def train_scratch(model_weights, run_name, epochs=60):
    """
    Standard training for the from-scratch model.
    """
    model = YOLO(model_weights)  # Create from scratch
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_scratch",
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_scratch", "weights", "best.pt")
