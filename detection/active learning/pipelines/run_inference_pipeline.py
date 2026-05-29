#!/usr/bin/env python3
import os
import sys
import cv2
import csv
import argparse
from pathlib import Path
import torch
from ultralytics import RTDETR, YOLO
from tqdm import tqdm
import concurrent.futures
import pandas as pd

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

# Import central configurations
from config import (
    CLASSES,
    DETECTION_THRESHOLDS,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_IMG_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    CLAHE_PREPROCESSED_DIR,
)

# Import static filter utility from sibling module
from filter_static_false_positives import filter_static_detections


def apply_clahe(im):
    """
    Applies CLAHE preprocessing in LAB space.
    Input: BGR image numpy array
    Output: BGR image numpy array with CLAHE applied
    """
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_clahe


def _process_image(args_tuple):
    """
    Helper function to process an image path (reads and applies CLAHE if requested).
    """
    img_path, apply_clahe_flag = args_tuple
    try:
        if apply_clahe_flag:
            # Map '/srv/shared_leopard_toad' -> CLAHE_PREPROCESSED_DIR in workspace
            norm_path = os.path.normpath(str(img_path))
            clahe_path = norm_path.replace(
                "/srv/shared_leopard_toad", CLAHE_PREPROCESSED_DIR
            )
            if os.path.exists(clahe_path):
                img_clahe = cv2.imread(clahe_path)
                if img_clahe is not None:
                    return img_clahe, img_path

            # Fallback: read original and apply CLAHE on the fly
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                return None, img_path
            input_img = apply_clahe(img_bgr)
        else:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                return None, img_path
            input_img = img_bgr
        return input_img, img_path
    except Exception:
        return None, img_path


def process_folder(
    input_folder,
    output_folder,
    model,
    img_size,
    batch_size,
    device,
    all_writer,
    apply_clahe_flag=True,
):
    """
    Runs batch inference on a target folder's image files.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Directory {input_path} does not exist. Skipping.")
        return []

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    excluded_cameras = {"4R", "5Z"}

    # Find immediate subdirectories and the root folder
    immediate_subfolders = [d for d in input_path.iterdir() if d.is_dir()]
    targets = [input_path] + immediate_subfolders

    folder_detections = []

    for target_dir in targets:
        # Determine image paths and CSV filename
        if target_dir == input_path:
            images = [
                f
                for f in target_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{input_path.name}_root.csv"
        else:
            images = [
                f
                for f in target_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{target_dir.name}.csv"

        # Filter out images from the test/val cameras (4R, 5Z)
        images = [
            f for f in images if not any(cam in f.parts for cam in excluded_cameras)
        ]

        if not images:
            continue

        csv_path = output_path / csv_name
        print(
            f"Found {len(images)} unlabeled images in '{target_dir.name}'. Saving predictions to {csv_name}..."
        )

        with open(csv_path, mode="w", newline="") as f_out:
            writer = csv.writer(f_out)
            headers = [
                "image_path",
                "image_name",
                "subfolder",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            writer.writerow(headers)

            with tqdm(total=len(images), desc=f"Processing {target_dir.name}") as pbar:
                for i in range(0, len(images), batch_size):
                    batch_img_paths = images[i : i + batch_size]
                    batch_input_imgs = []
                    valid_img_paths = []

                    # Run image loading and preprocessing in parallel
                    process_args = [(p, apply_clahe_flag) for p in batch_img_paths]
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=16
                    ) as executor:
                        results = list(executor.map(_process_image, process_args))

                    for input_img, img_path in results:
                        if input_img is None:
                            print(
                                f"Warning: Could not read image {img_path}. Skipping."
                            )
                        else:
                            batch_input_imgs.append(input_img)
                            valid_img_paths.append(img_path)

                    if not batch_input_imgs:
                        pbar.update(len(batch_img_paths))
                        continue

                    # Run batch inference
                    batch_results = model.predict(
                        batch_input_imgs,
                        conf=0.001,
                        imgsz=img_size,
                        verbose=False,
                        device=device,
                        half=(device != "cpu"),  # FP16 Half precision for GPU speedup
                        batch=len(batch_input_imgs),
                    )

                    for img_path, result in zip(valid_img_paths, batch_results):
                        if target_dir == input_path:
                            subfolder_name = "root"
                        else:
                            try:
                                subfolder_name = str(
                                    img_path.parent.relative_to(input_path)
                                )
                            except ValueError:
                                subfolder_name = img_path.parent.name

                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            class_name = model.names.get(
                                cls_id, CLASSES.get(cls_id, f"class_{cls_id}")
                            )
                            conf = float(box.conf[0])

                            # Apply class-specific optimal validation threshold
                            if conf >= DETECTION_THRESHOLDS.get(cls_id, 0.25):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                row_data = [
                                    str(img_path),
                                    img_path.name,
                                    subfolder_name,
                                    cls_id,
                                    class_name,
                                    f"{conf:.4f}",
                                    round(x1, 1),
                                    round(y1, 1),
                                    round(x2, 1),
                                    round(y2, 1),
                                ]
                                writer.writerow(row_data)
                                all_writer.writerow(row_data)
                                folder_detections.append(row_data)

                    pbar.update(len(batch_img_paths))

    return folder_detections


def main():
    parser = argparse.ArgumentParser(
        description="Run batch object detection inference on camera trap directories with RT-DETR."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained RT-DETR weights file (.pt) (default: '{DEFAULT_MODEL_PATH}').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output folder for prediction CSVs (default: '{DEFAULT_OUTPUT_DIR}').",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Inference image size (default: {DEFAULT_IMG_SIZE}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for parallel model predictions (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Hardware execution device (e.g. '0', 'cpu', 'cuda'). Defaults to auto-select ({DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--apply_clahe",
        action="store_true",
        default=True,
        help="Apply CLAHE preprocessing in LAB space.",
    )
    parser.add_argument(
        "--no_clahe",
        action="store_false",
        dest="apply_clahe",
        help="Disable CLAHE preprocessing.",
    )
    parser.add_argument(
        "--filter_static",
        action="store_true",
        help="Automatically apply spatial static bounding box filter after inference is done.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for clustering static bounding boxes (default: {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help=f"Maximum triggers allowed per bounding box cluster before suppression (default: {DEFAULT_OCCURRENCE_THRESHOLD}).",
    )

    args = parser.parse_args()

    # Determine execution device
    if args.device is None:
        device = DEFAULT_DEVICE
    else:
        device = args.device

    print("\n=========================================")
    print(f"LOADING INFERENCE PIPELINE")
    print(f"  Model Path:    {args.model_path}")
    print(f"  Device:        {device}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Image Size:    {args.img_size}")
    print(f"  CLAHE Prep:    {args.apply_clahe}")
    print(f"  Auto-Filter:   {args.filter_static}")
    print("=========================================")

    # Load Model (RT-DETR or YOLO) dynamically based on path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return

    model_name = os.path.basename(args.model_path).lower()
    if "rtdetr" in model_name or "rtdetr" in args.model_path.lower():
        print(f"Loading RT-DETR model from {args.model_path}")
        model = RTDETR(args.model_path)
    else:
        print(f"Loading YOLO model from {args.model_path}")
        model = YOLO(args.model_path)

    # Input year directories to run inference on
    years = {
        "2023": "/srv/shared_leopard_toad/2023",
        "2024": "/srv/shared_leopard_toad/2024",
        "2025": "/srv/shared_leopard_toad/2025/Documents",
    }

    # Consolidated combined predictions file path
    unified_csv_path = os.path.join(args.output_dir, "all_unlabeled_predictions.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(unified_csv_path, mode="w", newline="") as f_all:
        all_writer = csv.writer(f_all)
        headers = [
            "image_path",
            "image_name",
            "subfolder",
            "class_id",
            "class_name",
            "confidence",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ]
        all_writer.writerow(headers)

        grand_total_boxes = 0

        for year, base_input_dir in years.items():
            if not os.path.exists(base_input_dir):
                print(f"Year directory {base_input_dir} not found. Skipping.")
                continue

            # Gather folders inside the year directory
            folders = sorted(
                [d.name for d in Path(base_input_dir).iterdir() if d.is_dir()]
            )

            for folder in folders:
                in_dir = os.path.join(base_input_dir, folder)
                out_dir = os.path.join(args.output_dir, year, folder)

                print(f"\n--> Starting on folder: {year} / {folder}")
                detections = process_folder(
                    input_folder=in_dir,
                    output_folder=out_dir,
                    model=model,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    device=device,
                    all_writer=all_writer,
                    apply_clahe_flag=args.apply_clahe,
                )
                grand_total_boxes += len(detections)

    print("\n=========================================")
    print("ALL BATCH INFERENCE COMPLETED!")
    print(f"  Folder CSVs: {args.output_dir}")
    print(f"  Unified CSV: {unified_csv_path}")
    print(f"  Total BBoxes: {grand_total_boxes}")
    print("=========================================")

    # Run Integrated Post-processing Static Bounding Box Filter
    if args.filter_static:
        print("\n=========================================")
        print("RUNNING INTEGRATED POST-PROCESSING STATIC FILTER")
        print("=========================================")

        df = pd.read_csv(unified_csv_path)
        original_count = len(df)

        filtered_df, removed_count = filter_static_detections(
            df, args.iou_threshold, args.occurrence_threshold
        )

        filtered_csv_path = unified_csv_path.replace(".csv", "_filtered.csv")
        filtered_df.to_csv(filtered_csv_path, index=False)

        print("\n---------------------------------------------------------")
        print(f"Auto-Filtering complete:")
        print(f"  Original predictions:      {original_count}")
        print(
            f"  Suppressed static boxes:   {removed_count} ({removed_count / original_count:.1% if original_count > 0 else 0})"
        )
        print(f"  Remaining predictions:     {len(filtered_df)}")
        print(f"  Cleaned unified CSV saved: {filtered_csv_path}")
        print("=========================================")


if __name__ == "__main__":
    main()
