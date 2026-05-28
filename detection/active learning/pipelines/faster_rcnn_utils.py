import os
import sys
import glob
import cv2
import PIL.ImageEnhance as ImageEnhance
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Ensure pretraining directories can be resolved for legacy hooks
DETECTION_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PRETRAINING_DIR = os.path.join(DETECTION_DIR, "pretraining", "pipelines")
if PRETRAINING_DIR not in sys.path:
    sys.path.append(PRETRAINING_DIR)


class ActiveLearningFasterRCNNDataset(Dataset):
    def __init__(
        self, dataset_dir, split="train", img_size=640, augment=False, apply_clahe=False
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.apply_clahe = apply_clahe

        custom_img_dir = os.path.join(dataset_dir, split, "images")
        custom_lbl_dir = os.path.join(dataset_dir, split, "labels")
        yolo_img_dir = os.path.join(dataset_dir, "images", split)
        yolo_lbl_dir = os.path.join(dataset_dir, "labels", split)

        if os.path.exists(custom_img_dir):
            self.img_dir = custom_img_dir
            self.lbl_dir = custom_lbl_dir
        else:
            self.img_dir = yolo_img_dir
            self.lbl_dir = yolo_lbl_dir

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_files = [
            f
            for f in self.img_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")

            if self.apply_clahe:
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_channel)
                limg = cv2.merge((cl, a, b))
                img_clahe_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                img = Image.fromarray(cv2.cvtColor(img_clahe_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        xmin = (xc - w / 2) * self.img_size
                        ymin = (yc - h / 2) * self.img_size
                        xmax = (xc + w / 2) * self.img_size
                        ymax = (yc + h / 2) * self.img_size

                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = min(float(self.img_size), xmax)
                        ymax = min(float(self.img_size), ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id + 1)

        if self.augment:
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes) > 0:
                    new_boxes = []
                    for b in boxes:
                        new_boxes.append(
                            [self.img_size - b[2], b[1], self.img_size - b[0], b[3]]
                        )
                    boxes = new_boxes

            if np.random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(np.random.uniform(0.8, 1.2))

            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                img = img.rotate(angle, resample=Image.BILINEAR)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            final_boxes = []
            final_labels = []
            for i, b in enumerate(boxes):
                x1 = max(0.0, min(b[0], float(self.img_size)))
                y1 = max(0.0, min(b[1], float(self.img_size)))
                x2 = max(0.0, min(b[2], float(self.img_size)))
                y2 = max(0.0, min(b[3], float(self.img_size)))
                if (x2 > x1 + 1) and (y2 > y1 + 1):
                    final_boxes.append([x1, y1, x2, y2])
                    final_labels.append(labels[i])

            if not final_boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
                labels = torch.as_tensor(final_labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_model(num_classes=3, freeze_backbone=False):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_faster_rcnn(
    weights,
    run_name,
    project_dir,
    dataset_dir,
    freeze_backbone,
    epochs,
    patience,
    batch_size,
    apply_clahe,
    num_classes=3,
):
    """Training routine for Faster R-CNN using torch native classes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=num_classes, freeze_backbone=freeze_backbone)

    if weights and weights != "scratch" and os.path.exists(weights):
        print(f"Loading weights state dict from {weights}")
        model.load_state_dict(torch.load(weights, map_location=device))

    model = model.to(device)
    run_dir = os.path.join(project_dir, run_name)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_loader = DataLoader(
        ActiveLearningFasterRCNNDataset(
            dataset_dir, "train", img_size=640, augment=True, apply_clahe=apply_clahe
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ActiveLearningFasterRCNNDataset(
            dataset_dir, "val", img_size=640, augment=False, apply_clahe=apply_clahe
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=patience)
    best_loss = float("inf")
    history = []

    print(f"Starting Faster R-CNN Training: {run_name} for {epochs} epochs")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_comps = {
            "loss_classifier": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        }

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()
            for k in train_comps.keys():
                if k in loss_dict:
                    train_comps[k] += loss_dict[k].item()

        with torch.no_grad():
            val_loss = 0
            val_comps = {
                "loss_classifier": 0,
                "loss_box_reg": 0,
                "loss_objectness": 0,
                "loss_rpn_box_reg": 0,
            }
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
                for k in val_comps.keys():
                    if k in loss_dict:
                        val_comps[k] += loss_dict[k].item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, LR: {current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "train_classifier_loss": train_comps["loss_classifier"]
                / max(1, len(train_loader)),
                "val_classifier_loss": val_comps["loss_classifier"]
                / max(1, len(val_loader)),
                "learning_rate": current_lr,
            }
        )

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "results.csv"), index=False)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))

        scheduler.step(avg_val)
        early_stopping(avg_val)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    best_path = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    return best_path
