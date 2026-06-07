import os
import cv2
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from functools import lru_cache
from eval_utils.config import DATASETS, MAPPING_PATH
from eval_utils.spatial_filter import apply_spatial_filter
import json
from eval_utils.config import USE_FILTERED_PREDICTIONS, FILTERED_FILE_SUFFIX


def apply_clahe(im):
    """Apply CLAHE preprocessing to an BGR image."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_clahe


@lru_cache(maxsize=1)
def get_image_mapping():
    """Load the image mapping CSV into a DataFrame."""
    if not os.path.exists(MAPPING_PATH):
        return pd.DataFrame()
    return pd.read_csv(MAPPING_PATH)


@lru_cache(maxsize=1)
def get_image_mapping_dict():
    """Load the image mapping CSV into a dictionary for O(1) lookup."""
    df = get_image_mapping()
    if df.empty:
        return {}
    # Map normalized original_path to (unique_name, split)
    return {
        os.path.normpath(row["original_path"]): (row["unique_name"], row["split"])
        for _, row in df.iterrows()
    }


# Cache for label directory listings to avoid expensive os.path.exists disk I/O calls
_label_files_cache = {}


@lru_cache(maxsize=None)
def get_clean_ground_truth(original_path):
    """
    Get clean ground truth boxes and image-level label for an original image path.
    Returns: (is_positive, gt_boxes, split)
    """
    mapping_dict = get_image_mapping_dict()
    if not mapping_dict:
        return False, [], None

    # Normalize path for comparison
    norm_path = os.path.normpath(original_path)
    match = mapping_dict.get(norm_path)

    if not match:
        return False, [], None

    unique_name, split = match

    label_dir = DATASETS[split]["labels_dir"]
    if label_dir not in _label_files_cache:
        if os.path.exists(label_dir):
            _label_files_cache[label_dir] = set(os.listdir(label_dir))
        else:
            _label_files_cache[label_dir] = set()

    label_filename = os.path.splitext(unique_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_filename)

    gt_boxes = []
    if label_filename in _label_files_cache[label_dir]:
        with open(label_path, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        gt_boxes.append({"cls": int(cls), "bbox": [x, y, w, h]})
                except ValueError:
                    continue

    is_positive = len(gt_boxes) > 0
    return is_positive, gt_boxes, split


@lru_cache(maxsize=None)
def get_camera_images(camera_id, root_path="/srv/shared_leopard_toad"):
    """Find all image paths for a specific camera ID in the shared drive."""
    all_images = []
    print(f"Crawling {root_path} for camera {camera_id} images...")
    for root, dirs, files in os.walk(root_path):
        norm_root = os.path.normpath(root)
        path_parts = norm_root.split(os.sep)

        if camera_id in path_parts:
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg")):
                    all_images.append(os.path.normpath(os.path.join(root, file)))
    return all_images


@lru_cache(maxsize=None)
def get_ground_truth_positives(dataset_name):
    """Get a set of original paths for known positive images from the clean labels."""
    mapping = get_image_mapping()
    if mapping.empty:
        return set()

    split_mapping = mapping[mapping["split"] == dataset_name]
    positives = []

    for _, row in split_mapping.iterrows():
        unique_name = row["unique_name"]
        label_dir = DATASETS[dataset_name]["labels_dir"]
        label_path = os.path.join(label_dir, os.path.splitext(unique_name)[0] + ".txt")

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            positives.append(os.path.normpath(row["original_path"]))

    return set(positives)


@lru_cache(maxsize=None)
def get_dataset_images(dataset_name):
    """Get all original image paths for a dataset split from the mapping."""
    mapping = get_image_mapping()
    if mapping.empty:
        return []

    return (
        mapping[mapping["split"] == dataset_name]["original_path"]
        .map(os.path.normpath)
        .tolist()
    )


def refresh_results(results, is_full_seq=False):
    """
    Synchronize a list of raw prediction results with the latest clean ground truth.
    Returns a new list containing only images present in the mapping (or all images if is_full_seq).
    """
    refreshed = []
    for res in results:
        is_positive, gt_boxes, split = get_clean_ground_truth(res["path"])
        if split is not None:
            # Found in clean mapping, update GT
            res["is_positive"] = is_positive
            res["gt_boxes"] = gt_boxes
            refreshed.append(res)
        elif is_full_seq:
            # Not in clean mapping but we are in full sequence mode
            # Treat as negative (empty background)
            res["is_positive"] = False
            res["gt_boxes"] = []
            refreshed.append(res)
    return refreshed


def load_predictions_from_json(json_path, is_full_seq=False):
    """
    Loads predictions from a JSON file. If USE_FILTERED_PREDICTIONS is True
    and the corresponding filtered file exists, loads the pre-filtered predictions.
    Otherwise, loads the raw predictions, refreshes ground truth, and applies
    spatial filtering on-the-fly.
    """

    # Determine filtered json path
    if json_path.endswith("_raw.json"):
        filtered_path = json_path.replace("_raw.json", FILTERED_FILE_SUFFIX)
    elif json_path.endswith(FILTERED_FILE_SUFFIX):
        filtered_path = json_path
    else:
        base, ext = os.path.splitext(json_path)
        filtered_path = f"{base}_filtered{ext}"

    if USE_FILTERED_PREDICTIONS and os.path.exists(filtered_path):
        with open(filtered_path, "r") as f:
            return json.load(f)

    # Fallback: load raw predictions, refresh GT, and apply spatial filter on the fly
    raw_path = json_path
    if not os.path.exists(raw_path) and json_path.endswith(FILTERED_FILE_SUFFIX):
        raw_path = json_path.replace(FILTERED_FILE_SUFFIX, "_raw.json")

    with open(raw_path, "r") as f:
        results = json.load(f)

    results = refresh_results(results, is_full_seq=is_full_seq)

    if USE_FILTERED_PREDICTIONS:
        results = apply_spatial_filter(results)

    return results
