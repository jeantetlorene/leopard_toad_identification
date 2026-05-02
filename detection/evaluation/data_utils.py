import os
import cv2
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from config import DATASETS


def apply_clahe(im):
    """Apply CLAHE preprocessing to an BGR image."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_clahe


def load_labels(label_dir):
    """Load all YOLO-format labels from a directory into a map."""
    labels = {}
    if not os.path.exists(label_dir):
        return labels
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            # Extract original basename by splitting off Label Studio hash
            parts = label_file.replace(".txt", "").split("-", 1)
            original_name = parts[-1] if len(parts) > 1 else parts[0]

            with open(os.path.join(label_dir, label_file), "r") as f:
                boxes = []
                for line in f:
                    try:
                        cls, x, y, w, h = map(float, line.strip().split())
                        boxes.append({"cls": int(cls), "bbox": [x, y, w, h]})
                    except ValueError:
                        continue
                if original_name not in labels:
                    labels[original_name] = []
                labels[original_name].append({"full_name": label_file, "boxes": boxes})
    return labels


def get_best_label_match(original_path, label_map):
    """Disambiguate label matching using folder context if necessary."""
    basename = os.path.basename(original_path).replace(".JPG", "").replace(".jpg", "")
    if basename not in label_map:
        return []

    matches = label_map[basename]
    if len(matches) == 1:
        return matches[0]["boxes"]

    parent_folder = os.path.basename(os.path.dirname(original_path)).lower()
    for match in matches:
        if parent_folder in match["full_name"].lower():
            return match["boxes"]
    return matches[0]["boxes"]


def get_camera_images(camera_id, root_path="/srv/shared_leopard_toad"):
    """Find all image paths for a specific camera ID in the shared drive."""
    all_images = []
    print(f"Crawling {root_path} for camera {camera_id} images...")
    for root, dirs, files in os.walk(root_path):
        if os.path.basename(root) == camera_id:
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg")):
                    all_images.append(os.path.join(root, file))
    return all_images


def get_ground_truth_positives(dataset_name):
    """Get a set of paths for known positive images from the consensus CSV."""
    ds_info = DATASETS[dataset_name]
    df = pd.read_csv(ds_info["csv"], header=None, names=["path", "label", "id"])
    positives = set(df[df["label"].isin(["Correct", "Missed Animal"])]["path"].tolist())
    return positives
