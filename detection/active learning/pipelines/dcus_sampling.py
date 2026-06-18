#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

from config import CLASSES, CURATION_TARGET_CLASS


def load_detector_model(model_path, device, num_classes=3):
    """Loads YOLO, RT-DETR, or Faster R-CNN model dynamically."""
    model_name = os.path.basename(model_path).lower()
    if "rtdetr" in model_name or "rtdetr" in model_path.lower():
        from ultralytics import RTDETR

        print(f"DCUS: Loading RT-DETR model from {model_path}")
        return RTDETR(model_path)
    elif "faster_rcnn" in model_name or "faster_rcnn" in model_path.lower():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        print(f"DCUS: Loading Faster R-CNN model from {model_path}")
        model = fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        from ultralytics import YOLO

        print(f"DCUS: Loading YOLO model from {model_path}")
        return YOLO(model_path)


def get_ap_from_json(model_path):
    """Retrieves class-wise AP50 scores from results_dict.json if available."""
    # Sibling eval folders search
    run_dir = os.path.dirname(os.path.dirname(model_path))
    for split in ["val_eval", "test_eval"]:
        json_path = os.path.join(run_dir, split, "results_dict.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                ap_dict = {}
                for k, v in data.items():
                    if k.startswith("AP50_"):
                        class_name = k.replace("AP50_", "")
                        ap_dict[class_name] = float(v)
                if ap_dict:
                    print(f"DCUS: Loaded class APs from {json_path}")
                    return ap_dict
            except Exception as e:
                print(f"DCUS Warning: Error reading {json_path}: {e}")
    return None


def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def load_yolo_labels(label_path, img_width, img_height):
    """Loads and converts YOLO format labels into absolute coordinates."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = (xc - w / 2) * img_width
                    y1 = (yc - h / 2) * img_height
                    x2 = (xc + w / 2) * img_width
                    y2 = (yc + h / 2) * img_height
                    boxes.append({"class_id": cls_id, "bbox": [x1, y1, x2, y2]})
    return boxes


def run_val_inference(model, val_dir, device, max_images=200):
    """Runs model on validation images to collect box-level matching statistics."""
    # Find val images and labels
    img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    val_images_dir = None
    val_labels_dir = None

    # Try different structures
    for img_sub, lbl_sub in [
        ("val/images", "val/labels"),
        ("images/val", "labels/val"),
    ]:
        img_p = os.path.join(val_dir, img_sub)
        lbl_p = os.path.join(val_dir, lbl_sub)
        if os.path.exists(img_p):
            val_images_dir = img_p
            val_labels_dir = lbl_p
            break

    if not val_images_dir or not os.path.exists(val_images_dir):
        print("DCUS: Validation images directory not found.")
        return None

    img_files = sorted(
        [
            os.path.join(val_images_dir, f)
            for f in os.listdir(val_images_dir)
            if os.path.splitext(f)[1].lower() in img_extensions
        ]
    )[:max_images]

    if not img_files:
        print("DCUS: No validation images found.")
        return None

    print(f"DCUS: Running validation inference on {len(img_files)} images...")

    class_difficulties = {cid: [] for cid in CLASSES.keys()}

    is_ultralytics = hasattr(model, "predict")

    for img_path in img_files:
        # Get dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        # Load ground truths
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = (
            os.path.join(val_labels_dir, base_name + ".txt") if val_labels_dir else ""
        )
        gt_boxes = load_yolo_labels(label_path, w, h)

        # Predict
        preds = []
        if is_ultralytics:
            res = model.predict(img_path, imgsz=640, verbose=False, device=device)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                preds.append({"class_id": cls_id, "conf": conf, "bbox": coords})
        else:  # Faster R-CNN PyTorch model
            with torch.no_grad():
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                outputs = model(img_tensor)
                out = outputs[0]
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                for j in range(len(scores)):
                    preds.append(
                        {
                            "class_id": int(labels[j])
                            - 1,  # Convert 1-indexed back to 0-indexed
                            "conf": float(scores[j]),
                            "bbox": boxes[j].tolist(),
                        }
                    )

        # Match predicted boxes to ground truth
        for pred in preds:
            pred_cls = pred["class_id"]
            if pred_cls not in class_difficulties:
                continue
            # Find best match in ground truth of the same class
            best_iou = 0.0
            for gt in gt_boxes:
                if gt["class_id"] == pred_cls:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou

            # Match condition
            if best_iou >= 0.5:
                classification_diff = 1.0 - pred["conf"]
                localization_diff = 1.0 - best_iou
                difficulty = classification_diff + localization_diff
                class_difficulties[pred_cls].append(difficulty)

    # Calculate average difficulty
    avg_difficulties = {}
    for cid, diffs in class_difficulties.items():
        if diffs:
            avg_difficulties[CLASSES[cid]] = float(np.mean(diffs))
        else:
            avg_difficulties[CLASSES[cid]] = 1.5  # Default maximum difficulty

    return avg_difficulties


def compute_shannon_entropy(conf, num_classes):
    """Computes Shannon entropy for top-1 class confidence score."""
    if num_classes <= 1:
        return 0.0
    p = np.clip(conf, 1e-6, 1.0 - 1e-6)
    p_other = np.clip((1.0 - p) / (num_classes - 1), 1e-6, 1.0 - 1e-6)
    return -p * np.log(p) - (num_classes - 1) * p_other * np.log(p_other)


def main():
    parser = argparse.ArgumentParser(
        description="DCUS Query Uncertainty Sampling Script."
    )
    parser.add_argument(
        "--predictions_csv", type=str, required=True, help="Path topredictions CSV."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save output uncertainty CSV.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Path to validation dataset directory.",
    )
    parser.add_argument(
        "--beta", type=float, default=2.0, help="Difficulty weight multiplier."
    )
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device.")

    args = parser.parse_args()

    # Load predictions
    print(f"DCUS: Loading predictions from {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    if df.empty:
        print("DCUS: Empty predictions CSV. Copying to output.")
        df["entropy"] = []
        df["difficulty_coeff"] = []
        df["box_uncertainty"] = []
        df["uncertainty"] = []
        df.to_csv(args.output_csv, index=False)
        return

    num_classes = len(CLASSES)

    # 1. Resolve Class Difficulty weights
    difficulty_weights = {}

    # Attempt empirical validation matching
    empirical_difficulties = None
    if args.val_dir and os.path.exists(args.val_dir):
        try:
            model = load_detector_model(args.model_path, args.device, num_classes)
            empirical_difficulties = run_val_inference(model, args.val_dir, args.device)
            # Free model weights
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"DCUS Warning: Validation inference failed: {e}")

    if empirical_difficulties:
        print(
            "DCUS: Computing difficulty coefficients from validation matching statistics:"
        )
        for name, diff in empirical_difficulties.items():
            w = 1.0 + args.beta * diff
            difficulty_weights[name] = w
            print(f"  Class '{name}': difficulty = {diff:.4f}, weight (w_c) = {w:.4f}")
    else:
        # Fall back to results_dict.json AP50 values
        ap_dict = get_ap_from_json(args.model_path)
        if ap_dict:
            print("DCUS: Computing difficulty coefficients from class AP50 scores:")
            for idx, name in CLASSES.items():
                ap = ap_dict.get(name, 0.0)
                w = 1.0 + 2.0 * (1.0 - ap)
                difficulty_weights[name] = w
                print(f"  Class '{name}': AP50 = {ap:.4f}, weight (w_c) = {w:.4f}")
        else:
            # Fall back to hardcoded default values
            print("DCUS: Falling back to default class difficulty weights:")
            for idx, name in CLASSES.items():
                if name == CURATION_TARGET_CLASS:
                    w = 3.0
                elif name == "Small_Mammal":
                    w = 1.2
                else:
                    w = 2.0
                difficulty_weights[name] = w
                print(f"  Class '{name}': weight (w_c) = {w:.4f}")

    # 2. Compute entropy for each prediction box
    print("DCUS: Computing object-level Shannon entropy...")
    df["entropy"] = df["confidence"].apply(
        lambda p: compute_shannon_entropy(p, num_classes)
    )

    # 3. Apply difficulty coefficients
    df["difficulty_coeff"] = df["class_name"].map(difficulty_weights).fillna(2.0)
    df["box_uncertainty"] = df["difficulty_coeff"] * df["entropy"]

    # 4. Aggregate image-level uncertainty (sum of box difficulties)
    print("DCUS: Aggregating image-level uncertainty scores...")
    image_uncertainties = df.groupby("image_path")["box_uncertainty"].sum().to_dict()
    df["uncertainty"] = df["image_path"].map(image_uncertainties)

    # Save output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"DCUS: Saved uncertainty-scored predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
