"""
Crop images from a dataset based on YOLO labels or a predictions CSV file with optional CLAHE contrast enhancement.
"""

import cv2
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
IMAGES_DIR = "leopard_toad_identification/dataset/reid/images"
LABELS_DIR = "leopard_toad_identification/dataset/labels"
OUTPUT_DIR = "leopard_toad_identification/dataset/dataset_reid_crops"


def yolo_to_pixels(yolo_coords, img_w, img_h):
    """
    Convert YOLO normalized coordinates to pixel coordinates.
    yolo_coords: [class_id, x_center, y_center, width, height]
    """
    # Parse values
    _, x_c, y_c, w, h = map(float, yolo_coords)

    # Calculate corners
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)

    # Clamp to image boundaries (prevent negative or out-of-bounds)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2


def apply_clahe_preprocessing(crop):
    """
    Apply CLAHE to enhance image contrast. For color images, CLAHE is applied to the L (Lightness)
    channel of the LAB color space to preserve color structure without shift.
    """
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        # Color image: convert to LAB, apply CLAHE to L channel, convert back to BGR
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image: apply CLAHE directly
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(crop)


def generate_reid_dataset(apply_clahe=False):
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f
        for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    print(f"Found {len(image_files)} images. Starting cropping...")

    crops_created = 0

    for img_filename in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        # Check if label exists
        if not os.path.exists(label_path):
            continue

        # Read Image
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Read Labels
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Process each box in the file
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Get coordinates
            x1, y1, x2, y2 = yolo_to_pixels(parts, img_w, img_h)

            # Crop
            crop = img[y1:y2, x1:x2]

            # Skip empty crops (e.g. if box was 0 size)
            if crop.size == 0:
                continue

            # Apply CLAHE preprocessing if specified
            if apply_clahe:
                crop = apply_clahe_preprocessing(crop)

            # Save Crop
            save_name = f"{os.path.splitext(img_filename)[0]}_crop{i}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, crop)
            crops_created += 1

    print("Processing Complete!")
    print(f"Total Toads Cropped: {crops_created}")
    print(f"Saved to: {OUTPUT_DIR}")


def crop_from_csv(csv_path, output_dir, apply_clahe=False):
    """
    Crop images from a predictions CSV based on xmin, ymin, xmax, ymax.
    """
    print(f"Reading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # Clean up existing old crops to avoid filename format mixing
    print(f"Cleaning existing crops in: {output_dir}")
    for f in os.listdir(output_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                os.remove(os.path.join(output_dir, f))
            except Exception as e:
                print(f"Error removing {f}: {e}")

    print(f"Found {len(df)} predictions. Starting cropping...")

    crops_created = 0

    # We will track crop indices per trackable name to name crops uniquely (e.g., trackableName_crop0.jpg)
    image_crop_counts = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"]

        # Check if image path exists
        if not os.path.exists(img_path):
            print(f"\nWarning: Image path not found: {img_path}. Skipping.")
            continue

        # Get coordinates (convert to int)
        xmin = int(round(float(row["xmin"])))
        ymin = int(round(float(row["ymin"])))
        xmax = int(round(float(row["xmax"])))
        ymax = int(round(float(row["ymax"])))

        # Read Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"\nWarning: Failed to read image: {img_path}. Skipping.")
            continue

        img_h, img_w = img.shape[:2]

        # Clamp coordinates to image boundaries
        x1 = max(0, min(img_w, xmin))
        y1 = max(0, min(img_h, ymin))
        x2 = max(0, min(img_w, xmax))
        y2 = max(0, min(img_h, ymax))

        # Crop
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Apply CLAHE preprocessing if specified
        if apply_clahe:
            crop = apply_clahe_preprocessing(crop)

        # Determine trackable base name from the relative path under /srv/shared_leopard_toad/
        ref_prefix = "/srv/shared_leopard_toad/"
        if ref_prefix in img_path:
            rel_path = img_path.split(ref_prefix, 1)[1]
        else:
            rel_path = img_path.lstrip("/")

        # Extract relative path without extension and replace slash separators with double underscores
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        trackable_name = rel_path_no_ext.replace("/", "__")

        # Track crop index
        crop_idx = image_crop_counts.get(trackable_name, 0)
        image_crop_counts[trackable_name] = crop_idx + 1

        # Save Crop
        save_name = f"{trackable_name}_crop{crop_idx}.jpg"
        save_path = os.path.join(output_dir, save_name)

        cv2.imwrite(save_path, crop)
        crops_created += 1

    print("\nProcessing Complete!")
    print(f"Total Toads Cropped: {crops_created}")
    print(f"Saved to: {output_dir}")


def crop_from_json(json_path, output_dir, apply_clahe=False, conf_threshold=0.5, class_id=2):
    """
    Crop images from a predictions JSON file. 
    The JSON should be formatted with normalized [x_center, y_center, w, h] predictions.
    """
    print(f"Reading predictions from JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Cleaning existing crops in: {output_dir}")
    for f in os.listdir(output_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                os.remove(os.path.join(output_dir, f))
            except Exception as e:
                print(f"Error removing {f}: {e}")

    # Filter data first to only process images with valid predictions
    valid_data = []
    for entry in data:
        valid_preds = [p for p in entry.get("predictions", []) if p["cls"] == class_id and p["conf"] >= conf_threshold]
        if valid_preds:
            valid_data.append((entry["path"], valid_preds))

    print(f"Found {len(valid_data)} images containing class {class_id} with confidence >= {conf_threshold}. Starting cropping...")

    crops_created = 0
    image_crop_counts = {}

    for img_path, preds in tqdm(valid_data):
        if not os.path.exists(img_path):
            print(f"\nWarning: Image path not found: {img_path}. Skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"\nWarning: Failed to read image: {img_path}. Skipping.")
            continue

        img_h, img_w = img.shape[:2]

        # Trackable name logic (same as CSV)
        ref_prefix = "/srv/shared_leopard_toad/"
        if ref_prefix in img_path:
            rel_path = img_path.split(ref_prefix, 1)[1]
        else:
            rel_path = img_path.lstrip("/")
        
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        trackable_name = rel_path_no_ext.replace("/", "__")

        for pred in preds:
            # pred["bbox"] is [x_center, y_center, w, h] normalized
            yolo_coords = [pred["cls"]] + pred["bbox"]
            x1, y1, x2, y2 = yolo_to_pixels(yolo_coords, img_w, img_h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if apply_clahe:
                crop = apply_clahe_preprocessing(crop)

            crop_idx = image_crop_counts.get(trackable_name, 0)
            image_crop_counts[trackable_name] = crop_idx + 1

            save_name = f"{trackable_name}_crop{crop_idx}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, crop)
            crops_created += 1

    print("\nProcessing Complete!")
    print(f"Total Toads Cropped: {crops_created}")
    print(f"Saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Crop images from a dataset based on YOLO labels or CSV predictions with optional CLAHE preprocessing."
    )
    parser.add_argument(
        "--csv", type=str, help="Path to predictions CSV file to crop from."
    )
    parser.add_argument(
        "--json", type=str, help="Path to predictions JSON file to crop from."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory for crops when using CSV or JSON mode.",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE contrast enhancement preprocessing to each crop before saving.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for crops when using JSON mode. Default 0.5."
    )

    args = parser.parse_args()

    if args.csv:
        if not args.output_dir:
            parser.error("--output-dir is required when using --csv mode.")
        crop_from_csv(args.csv, args.output_dir, apply_clahe=args.clahe)
    elif args.json:
        if not args.output_dir:
            parser.error("--output-dir is required when using --json mode.")
        crop_from_json(args.json, args.output_dir, apply_clahe=args.clahe, conf_threshold=args.conf)
    else:
        generate_reid_dataset(apply_clahe=args.clahe)


if __name__ == "__main__":
    main()
