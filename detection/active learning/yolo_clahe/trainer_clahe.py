import os
import cv2
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from config import YOLO_DIR, TRAIN_BATCH_SIZE, IMG_SIZE, DEVICE

# --- Monkey Patch Ultralytics Dataset to apply CLAHE on-the-fly ---
original_load_image = YOLODataset.load_image


def patched_load_image(self, i, *args, **kwargs):
    # Load the image using original Ultralytics method
    im, (h, w), (h0, w0) = original_load_image(self, i, *args, **kwargs)

    # Apply CLAHE
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return im_clahe, (h, w), (h0, w0)


# Apply the patch
YOLODataset.load_image = patched_load_image
# ------------------------------------------------------------------


def train_phase_1(
    model_weights, run_name, dataset_yaml, freeze=15, epochs=100, patience=25
):
    model = YOLO(model_weights)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_phase1",
        freeze=freeze,
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_phase1", "weights", "best.pt")


def train_phase_2(model_weights, run_name, dataset_yaml, epochs=30):
    model = YOLO(model_weights)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_phase2",
        freeze=0,
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_phase2", "weights", "best.pt")


def train_scratch(model_weights, run_name, dataset_yaml, epochs=300, patience=50):
    model = YOLO(model_weights)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH_SIZE,
        project=os.path.join(YOLO_DIR, "runs"),
        name=f"{run_name}_scratch",
        device=DEVICE,
        verbose=False,
    )
    return os.path.join(YOLO_DIR, "runs", f"{run_name}_scratch", "weights", "best.pt")
