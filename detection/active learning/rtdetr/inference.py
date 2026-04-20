import os
import cv2
import concurrent.futures
from pathlib import Path
from config import (
    YEARS,
    FOLDERS,
    EXCLUDED_CAMERAS,
    CONF_THRESHOLD,
    IMG_SIZE,
    DEVICE,
    TRAIN_IMAGES_DIR,
)


def get_unlabeled_pool(mode, current_cycle):
    from config import BASE_DIR

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pool = []

    import csv

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
        candidate_csv_path = os.path.join(
            BASE_DIR,
            "active learning",
            "rtdetr",
            "cycles",
            mode,
            f"cycle_{c}",
            f"al_query_candidates_{mode}_cycle_{c}.csv",
        )
        if os.path.exists(candidate_csv_path):
            with open(candidate_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "image_path" in row:
                        annotated_basenames.add(os.path.basename(row["image_path"]))

    for year, base_input_dir in YEARS.items():
        for folder in FOLDERS:
            in_dir = Path(base_input_dir) / folder
            if not in_dir.exists():
                continue

            for img_path in in_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    # Skip excluded Static Eval cameras
                    if not any(f"/{cam}/" in str(img_path) for cam in EXCLUDED_CAMERAS):
                        # Skip anything that already exists in the training set
                        if img_path.name not in annotated_basenames:
                            pool.append(str(img_path))
    return pool


def _parallel_read(path):
    # A fast, lightweight byte load can destroy network I/O latency
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


def extract_features_and_boxes_batch(model, chunk_paths):
    """
    Run YOLO inference batch-wise.
    Extract bounding boxes for DCUS and dummy semantic features for Diversity.
    """
    chunk_boxes = []
    chunk_features = []

    # [OPTIMIZATION] Parallelize the massive slow sequential disk/network I/O load!
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        loaded_imgs = list(executor.map(_parallel_read, chunk_paths))

    # Safely extract what was successfully loaded
    valid_paths = []
    valid_imgs = []
    for path, img in zip(chunk_paths, loaded_imgs):
        if img is not None:
            valid_paths.append(path)
            valid_imgs.append(img)

    if not valid_imgs:
        return [], [], []

    try:
        # [OPTIMIZATION] Set batch=len(valid_imgs) to ensure YOLO pushes massive tensors together!
        results = model(
            valid_imgs,
            verbose=False,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=True,
            batch=len(valid_imgs),
        )

        for k, result in enumerate(results):
            path = valid_paths[k]
            boxes = []

            for box in result.boxes:
                boxes.append({"cls": int(box.cls[0]), "conf": float(box.conf[0])})

            # If standard embedding is heavy, we represent the 'semantic layout'
            # of the image purely via extreme dimensionality reduction: the
            # confidences and classes of the top 3 objects.
            semantic_features = []
            for b in boxes[:3]:
                semantic_features.extend([b["cls"], b["conf"]])
            semantic_features += [0.0] * (6 - len(semantic_features))  # Pad to 6

            chunk_boxes.append(boxes)
            chunk_features.append(semantic_features)

        return valid_paths, chunk_boxes, chunk_features
    except Exception as e:
        print(f"Failed to infer batch: {e}")
        return [], [], []
