#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Define path resolutions
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
DETECTION_DIR = os.path.dirname(ACTIVE_LEARNING_DIR)

# Dynamically add detection/pretraining to sys.path for Faster R-CNN dataset loaders
PRETRAINING_DIR = os.path.join(DETECTION_DIR, "pretraining")
if PRETRAINING_DIR not in sys.path:
    sys.path.append(PRETRAINING_DIR)

# Dynamically add ACTIVE_LEARNING_DIR to sys.path for importing central config
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

# Centralized imports
from central_config import (
    DEFAULT_DEVICE,
    YOLO_TRAIN_CONFIG,
    RTDETR_TRAIN_CONFIG,
    FASTER_RCNN_TRAIN_CONFIG,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Active Learning Model Training Suite."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["yolo", "rtdetr", "faster_rcnn"],
        required=True,
        help="Model architecture type to train.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrained", "scratch"],
        required=True,
        help="Train model starting from domain-pretrained weights or from scratch.",
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Active learning cycle number."
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Train model using CLAHE contrast preprocessing.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional custom experiment/run name to separate training runs, datasets, and candidates.",
    )
    return parser.parse_args()


def resolve_base_weights(model_type, mode):
    """Resolves the initial base weights for training cycles."""
    if mode == "pretrained":
        if model_type == "yolo":
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "detect",
                "yolo_finetuning",
                "weights",
                "best.pt",
            )
        elif model_type == "rtdetr":
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "detect",
                "rtdetr_finetuning",
                "weights",
                "best.pt",
            )
        else:  # faster_rcnn
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "faster_rcnn",
                "train_resnet50_1",
                "weights",
                "best.pt",
            )
            if not os.path.exists(w_path):
                # Fallback to alternative pretrained path
                w_path = os.path.join(
                    DETECTION_DIR,
                    "pretraining",
                    "runs",
                    "faster_rcnn",
                    "train_resnet50",
                    "weights",
                    "best.pt",
                )
    else:  # scratch
        if model_type == "yolo":
            w_path = os.path.join(ACTIVE_LEARNING_DIR, "yolo", "yolo26n.pt")
            if not os.path.exists(w_path):
                w_path = "yolo26n.pt"  # Fallback to auto-download from ultralytics
        elif model_type == "rtdetr":
            w_path = os.path.join(ACTIVE_LEARNING_DIR, "rtdetr", "rtdetr-l.pt")
            if not os.path.exists(w_path):
                w_path = "rtdetr-l.pt"  # Fallback to auto-download from ultralytics
        else:
            w_path = "scratch"

    return w_path


def train_ultralytics(
    model_class,
    model_type,
    weights,
    run_name,
    project_dir,
    dataset_yaml,
    freeze,
    epochs,
    patience,
    batch_size,
):
    """Generic training routine for Ultralytics models (YOLO & RT-DETR)."""
    model = model_class(weights)
    device_val = "0" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=batch_size,
        project=project_dir,
        name=run_name,
        freeze=freeze,
        device=device_val,
        verbose=False,
    )
    return os.path.join(project_dir, run_name, "weights", "best.pt")


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
                import cv2

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

            import PIL.ImageEnhance as ImageEnhance

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


def get_model(num_classes=3, freeze_backbone=False):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
):
    """Training routine for Faster R-CNN using torch native classes."""
    from torch.utils.data import DataLoader
    import pandas as pd
    from train_faster_rcnn import collate_fn, EarlyStopping

    # Intercept trainer configs or define custom run routine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=3, freeze_backbone=freeze_backbone)

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


def main():
    args = parse_args()

    # Resolve the model-specific training config dictionary
    if args.model_type == "yolo":
        train_config = YOLO_TRAIN_CONFIG
    elif args.model_type == "rtdetr":
        train_config = RTDETR_TRAIN_CONFIG
    elif args.model_type == "faster_rcnn":
        train_config = FASTER_RCNN_TRAIN_CONFIG
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # 1. Resolve folders and paths
    clahe_suffix = "clahe" if args.clahe else "plain"
    model_folder = (
        f"{args.model_type}_{clahe_suffix}" if args.clahe else args.model_type
    )

    # Change working dir to active learning specific model folder to let YOLO/RT-DETR save caches properly
    model_dir = os.path.join(ACTIVE_LEARNING_DIR, model_folder)
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)

    # Add model_dir to sys.path to resolve internal trainer hooks
    if model_dir not in sys.path:
        sys.path.append(model_dir)

    # Dynamically apply CLAHE on-the-fly to YOLODataset loaded images for Ultralytics models
    if args.clahe and args.model_type in ["yolo", "rtdetr"]:
        import cv2
        from ultralytics.data.dataset import YOLODataset

        original_load_image = YOLODataset.load_image

        def patched_load_image(self, i, *args, **kwargs):
            im, (h, w), (h0, w0) = original_load_image(self, i, *args, **kwargs)
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return im_clahe, (h, w), (h0, w0)

        YOLODataset.load_image = patched_load_image
        print(
            "[Patch] Successfully monkey-patched YOLODataset.load_image to apply CLAHE on-the-fly!"
        )

    print(f"\n=======================================================")
    print(f"STARTING UNIFIED MODEL TRAINING")
    print(f"  Architecture:  {args.model_type.upper()}")
    print(f"  Preprocessing: {clahe_suffix.upper()}")
    print(f"  Mode:          {args.mode.upper()}")
    print(f"  Cycle:         {args.cycle}")
    print(f"  Model Dir:     {model_dir}")
    print("=======================================================")

    if args.experiment_name:
        dataset_dir = os.path.join(
            ACTIVE_LEARNING_DIR,
            "data",
            args.model_type,
            args.mode,
            args.experiment_name,
            f"cycle_{args.cycle}",
        )
    else:
        dataset_dir = os.path.join(
            ACTIVE_LEARNING_DIR,
            "data",
            args.model_type,
            args.mode,
            f"cycle_{args.cycle}",
        )
    cycle_parent = os.path.join(model_dir, "cycles", args.mode)
    if args.experiment_name:
        cycle_parent = os.path.join(cycle_parent, args.experiment_name)
    cycle_dir = os.path.join(cycle_parent, f"cycle_{args.cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    dataset_yaml = os.path.join(
        cycle_dir, f"dataset_{args.mode}_cycle_{args.cycle}.yaml"
    )

    # Create dataset yaml file dynamically
    yaml_content = f"""path: {dataset_dir}
train: train/images
val: val/images
test: test/images

names:
  0: Other_Amphibian
  1: Small_Mammal
  2: Western_Leopard_Toad
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Created dataset YAML at: {dataset_yaml}")

    # Resolve initial base weights
    base_weights = resolve_base_weights(args.model_type, args.mode)
    print(f"Base weights resolved: {base_weights}")

    # Establish project runs directories
    project_dir = os.path.join(model_dir, "runs")
    if args.experiment_name:
        project_dir = os.path.join(project_dir, args.experiment_name)
    os.makedirs(project_dir, exist_ok=True)

    # Define outputs paths based on cycle mode
    if args.mode == "pretrained":
        expected_p2_model = os.path.join(
            project_dir, f"cycle_{args.cycle}_pretrained_phase2", "weights", "best.pt"
        )
        if os.path.exists(expected_p2_model):
            print(
                f"Trained model for Cycle {args.cycle} (Phase 2) already exists at: {expected_p2_model}. Skipping training."
            )
            return

        p1_cfg = train_config["pretrained"]["phase1"]
        p2_cfg = train_config["pretrained"]["phase2"]

        print("\n--- PHASE 1: Fine-tune Head Only (Backbone Frozen) ---")
        if args.model_type in ["yolo", "rtdetr"]:
            from ultralytics import YOLO, RTDETR

            model_class = YOLO if args.model_type == "yolo" else RTDETR

            p1_weights = train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase1",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=p1_cfg["freeze"],
                epochs=p1_cfg["epochs"],
                patience=p1_cfg["patience"],
                batch_size=p1_cfg["batch_size"],
            )

            print("\n--- PHASE 2: Adapt Entire Network (Backbone Unfrozen) ---")
            train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=p1_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase2",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=p2_cfg["freeze"],
                epochs=p2_cfg["epochs"],
                patience=p2_cfg["patience"],
                batch_size=p2_cfg["batch_size"],
            )
        else:  # faster_rcnn
            p1_weights = train_faster_rcnn(
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase1",
                project_dir=project_dir,
                dataset_dir=dataset_dir,
                freeze_backbone=p1_cfg["freeze_backbone"],
                epochs=p1_cfg["epochs"],
                patience=p1_cfg["patience"],
                batch_size=p1_cfg["batch_size"],
                apply_clahe=args.clahe,
            )

            print("\n--- PHASE 2: Adapt Entire Network (Backbone Unfrozen) ---")
            train_faster_rcnn(
                weights=p1_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase2",
                project_dir=project_dir,
                dataset_dir=dataset_dir,
                freeze_backbone=p2_cfg["freeze_backbone"],
                epochs=p2_cfg["epochs"],
                patience=p2_cfg["patience"],
                batch_size=p2_cfg["batch_size"],
                apply_clahe=args.clahe,
            )
    else:  # scratch mode
        expected_scratch_model = os.path.join(
            project_dir, f"cycle_{args.cycle}_scratch_scratch", "weights", "best.pt"
        )
        if os.path.exists(expected_scratch_model):
            print(
                f"Trained model for Cycle {args.cycle} (Scratch) already exists at: {expected_scratch_model}. Skipping training."
            )
            return

        scratch_cfg = train_config["scratch"]

        print("\n--- FROM-SCRATCH MODEL TRAINING ---")
        if args.model_type in ["yolo", "rtdetr"]:
            from ultralytics import YOLO, RTDETR

            model_class = YOLO if args.model_type == "yolo" else RTDETR

            train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_scratch_scratch",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=scratch_cfg.get("freeze", 0),
                epochs=scratch_cfg["epochs"],
                patience=scratch_cfg["patience"],
                batch_size=scratch_cfg["batch_size"],
            )
        else:  # faster_rcnn
            train_faster_rcnn(
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_scratch_scratch",
                project_dir=project_dir,
                dataset_dir=dataset_dir,
                freeze_backbone=scratch_cfg["freeze_backbone"],
                epochs=scratch_cfg["epochs"],
                patience=scratch_cfg["patience"],
                batch_size=scratch_cfg["batch_size"],
                apply_clahe=args.clahe,
            )

    print("\n=======================================================")
    print("UNIFIED MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
