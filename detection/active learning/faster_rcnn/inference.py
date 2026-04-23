import os
import torch
import cv2
import csv
import numpy as np
import concurrent.futures
from pathlib import Path
from config import (
    YEARS,
    FOLDERS,
    EXCLUDED_CAMERAS,
    CONF_THRESHOLD,
    IMG_SIZE,
    DEVICE,
)


def get_unlabeled_pool(mode, current_cycle):
    from config import BASE_DIR

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pool = []

    # Step 1: Securely identify all images that have already been annotated
    annotated_basenames = set()

    # 1a. Exclude ALL images that are statically assigned natively
    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(BASE_DIR, "active learning", "data", csv_file)
        if os.path.exists(csv_path):
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "image_name" in row:
                        annotated_basenames.add(row["image_name"])
                    elif "image_path" in row:
                        annotated_basenames.add(os.path.basename(row["image_path"]))

    # 1b. Exclude all images that have been suggested in previous cycles for this exact model & mode
    for c in range(current_cycle):
        # 1. Check the new structure for candidate CSVs
        candidate_csv_path_new = os.path.join(
            BASE_DIR,
            "active learning",
            "faster_rcnn",
            "cycles",
            mode,
            f"cycle_{c}",
            f"al_query_candidates_{mode}_cycle_{c}.csv",
        )
        # 2. Check the old/legacy structure for candidate CSVs
        candidate_csv_path_old = os.path.join(
            BASE_DIR,
            "active learning",
            "faster_rcnn",
            f"al_query_candidates_{mode}_cycle_{c}.csv",
        )

        for candidate_csv_path in [candidate_csv_path_new, candidate_csv_path_old]:
            if os.path.exists(candidate_csv_path):
                with open(candidate_csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "image_path" in row:
                            annotated_basenames.add(os.path.basename(row["image_path"]))

        # 3. As an ultimate fallback, exclude any images physically present in the training directories of previous cycles
        train_img_dir = os.path.join(
            BASE_DIR,
            "active learning",
            "data",
            "faster_rcnn",
            mode,
            f"cycle_{c}",
            "train",
            "images",
        )
        if os.path.exists(train_img_dir):
            for img_name in os.listdir(train_img_dir):
                if img_name.lower().endswith(tuple(image_extensions)):
                    annotated_basenames.add(img_name)

    for year, base_input_dir in YEARS.items():
        for folder in FOLDERS:
            in_dir = Path(base_input_dir) / folder
            if not in_dir.exists():
                continue

            for img_path in in_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    if not any(f"/{cam}/" in str(img_path) for cam in EXCLUDED_CAMERAS):
                        if img_path.name not in annotated_basenames:
                            pool.append(str(img_path))
    return pool


def _parallel_read_and_tensor(path):
    try:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
        )
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)  # C, H, W
        return img_tensor
    except Exception:
        return None


def extract_features_and_boxes_batch(model, chunk_paths):
    chunk_boxes = []
    chunk_features = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        loaded_tensors = list(executor.map(_parallel_read_and_tensor, chunk_paths))

    valid_paths = []
    valid_tensors = []
    for path, t in zip(chunk_paths, loaded_tensors):
        if t is not None:
            valid_paths.append(path)
            valid_tensors.append(t)

    if not valid_tensors:
        return [], [], []

    try:
        # Pytorch Faster R-CNN requires list of tensors
        inputs = [t.to(DEVICE) for t in valid_tensors]

        # Pytorch hook for deep features
        features_list = []

        def hook(module, inputs, outputs):
            # outputs is an OrderedDict from FPN. 'pool' gives [N, 256, 4, 4]. GAP it down to [N, 256]
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                outputs["pool"], (1, 1)
            ).flatten(1)
            features_list.append(pooled.cpu().numpy())

        handle = model.backbone.register_forward_hook(hook)

        with torch.no_grad():
            outputs = model(inputs)

        handle.remove()
        batch_features = (
            np.concatenate(features_list, axis=0)
            if features_list
            else np.zeros((len(inputs), 256))
        )

        for k, out in enumerate(outputs):
            path = valid_paths[k]
            boxes = []

            pred_boxes = out["boxes"].cpu().numpy()
            pred_scores = out["scores"].cpu().numpy()
            pred_labels = out["labels"].cpu().numpy()

            # Filter by conf
            for i in range(len(pred_scores)):
                if pred_scores[i] >= CONF_THRESHOLD:
                    # native torch Faster r-cnn labels are 1-indexed. Yolo is 0.
                    boxes.append(
                        {"cls": int(pred_labels[i]) - 1, "conf": float(pred_scores[i])}
                    )

            chunk_boxes.append(boxes)
            chunk_features.append(batch_features[k].tolist())

        del inputs
        del outputs
        torch.cuda.empty_cache()

        return valid_paths, chunk_boxes, chunk_features
    except Exception as e:
        print(f"Failed to infer batch: {e}")
        torch.cuda.empty_cache()
        return [], [], []
