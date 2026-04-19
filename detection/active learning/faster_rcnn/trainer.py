import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from config import DATASET_DIR, FASTER_RCNN_DIR, TRAIN_BATCH_SIZE, IMG_SIZE, DEVICE
from train_faster_rcnn import YoloToFasterRCNNDataset, collate_fn, EarlyStopping


def get_model(num_classes=3, freeze_backbone=False):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _train_routine(model, run_name, epochs=100, patience=15):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_dir = os.path.join(FASTER_RCNN_DIR, "runs", run_name)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # We define validation dynamically as detect_1 val, since AL is dynamically gathering
    train_loader = DataLoader(
        YoloToFasterRCNNDataset(DATASET_DIR, "train", img_size=IMG_SIZE, augment=True),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        YoloToFasterRCNNDataset(
            DATASET_DIR,
            "val",
            img_size=IMG_SIZE,
            augment=False,  # Use the true val set!
        ),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=patience)
    best_loss = float("inf")

    history = []

    print(f"Starting Training: {run_name} for {epochs} epochs")
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

        # Simplified Val
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

        current_lr = optimizer.param_groups[0]['lr']
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


def train_phase_1(model_weights, run_name, freeze=True, epochs=100, patience=15):
    """Phase 1: Freeze backbone"""
    model = get_model(num_classes=3, freeze_backbone=freeze)
    if model_weights and model_weights != "scratch" and os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights))

    return _train_routine(model, f"{run_name}_phase1", epochs, patience)


def train_phase_2(model_weights, run_name, epochs=30):
    """Phase 2: Unfreeze Backbone"""
    model = get_model(num_classes=3, freeze_backbone=False)
    if model_weights and os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights))

    return _train_routine(model, f"{run_name}_phase2", epochs, 15)


def train_scratch(model_weights, run_name, epochs=100, patience=20):
    """Train completely from scratch until convergence"""
    model = get_model(num_classes=3, freeze_backbone=False)
    # ignore string weights since it's from scratch
    return _train_routine(model, f"{run_name}_scratch", epochs, patience)
