import os
from pathlib import Path
from config import YEARS, FOLDERS, EXCLUDED_CAMERAS, CONF_THRESHOLD


def get_unlabeled_pool():
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    pool = []

    for year, base_input_dir in YEARS.items():
        for folder in FOLDERS:
            in_dir = Path(base_input_dir) / folder
            if not in_dir.exists():
                continue

            for img_path in in_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    if not any(f"/{cam}/" in str(img_path) for cam in EXCLUDED_CAMERAS):
                        pool.append(str(img_path))
    return pool


def extract_features_and_boxes_batch(model, chunk_paths):
    """
    Run YOLO inference batch-wise.
    Extract bounding boxes for DCUS and dummy semantic features for Diversity.
    """
    valid_paths = []
    chunk_boxes = []
    chunk_features = []

    try:
        results = model(chunk_paths, verbose=False, conf=CONF_THRESHOLD)

        for k, result in enumerate(results):
            path = chunk_paths[k]
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

            valid_paths.append(path)
            chunk_boxes.append(boxes)
            chunk_features.append(semantic_features)

        return valid_paths, chunk_boxes, chunk_features
    except Exception as e:
        print(f"Failed to infer batch: {e}")
        return [], [], []
