import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import cv2
import shutil

# Project imports
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import MODEL_ROOTS, DEVICE, CLASSES, DEFAULT_BATCH_SIZE
from eval_utils.data_utils import apply_clahe
from eval_utils.metrics import calculate_detection_metrics, calculate_map50_95
from eval_utils.models.sahi_wrapper import SAHIWrapper
from eval_utils.inference import generate_predictions

DATA_DIR = (
    "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation/data"
)
RESULTS_DIR_FILES = "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation/results/files"


def prepare_clahe_set(split):
    base_dir = os.path.join(DATA_DIR, split)
    # Use a standard YOLO structure: clahe_set/images and clahe_set/labels
    clahe_root = os.path.join(base_dir, "clahe_set")
    clahe_images_dir = os.path.join(clahe_root, "images")
    clahe_labels_dir = os.path.join(clahe_root, "labels")

    if os.path.exists(clahe_images_dir) and len(os.listdir(clahe_images_dir)) > 0:
        if not os.path.exists(clahe_labels_dir):
            print(f"Creating missing symlink for {split} CLAHE labels...")
            os.symlink(os.path.join(base_dir, "labels"), clahe_labels_dir)
        return clahe_root

    os.makedirs(clahe_images_dir, exist_ok=True)

    # Create symlink for labels to the original labels directory
    if not os.path.exists(clahe_labels_dir):
        print(f"Creating symlink for {split} CLAHE labels...")
        os.symlink(os.path.join(base_dir, "labels"), clahe_labels_dir)

    print(f"Generating CLAHE-processed {split} set...")
    images_dir = os.path.join(base_dir, "images")
    for img_name in tqdm(os.listdir(images_dir)):
        if img_name.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_clahe = apply_clahe(img)
                cv2.imwrite(os.path.join(clahe_images_dir, img_name), img_clahe)
    return clahe_root


def run_evaluation():
    os.makedirs(RESULTS_DIR_FILES, exist_ok=True)

    splits = ["test", "val"]

    # Prepare sets (no YAML needed for SAHI inference loop)
    for split in splits:
        prepare_clahe_set(split)

    # Discover models
    for model_key, root_dir in MODEL_ROOTS.items():
        if not os.path.exists(root_dir):
            continue

        runs_dir = os.path.join(root_dir, "runs")
        if not os.path.exists(runs_dir):
            continue

        is_clahe = "clahe" in model_key
        model_type = (
            "yolo"
            if "yolo" in model_key
            else ("rtdetr" if "rtdetr" in model_key else "faster_rcnn")
        )

        print(f"\nProcessing {model_key} models with SAHI...")

        for run_name in sorted(os.listdir(runs_dir)):
            model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
            if not os.path.exists(model_path):
                continue

            for split in splits:
                eval_name = f"{split}_sahi_eval"
                eval_dir = os.path.join(runs_dir, run_name, eval_name)

                # Check if evaluation already exists
                if os.path.exists(os.path.join(eval_dir, "results_dict.json")):
                    print(
                        f"Skipping {run_name} on {split} (already evaluated with SAHI)."
                    )
                    continue

                # Ensure the eval dir exists
                os.makedirs(eval_dir, exist_ok=True)

                print(
                    f"Evaluating {run_name} on {split} (CLAHE={is_clahe}) using SAHI..."
                )

                # Initialize SAHI Wrapper
                wrapper = SAHIWrapper(
                    model_type=model_type,
                    model_path=model_path,
                    device=DEVICE,
                    confidence_threshold=0.001,
                    sahi_batch_size=64,
                    no_standard_prediction=True,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.1,
                )

                # Run SAHI inference
                # Note: SAHI inference is slower, so we use a smaller batch size for the loop
                # although predict_batch itself loops over images.
                eval_results = generate_predictions(
                    wrapper,
                    split,
                    use_clahe=is_clahe,
                    batch_size=32,
                )

                # Calculate metrics
                det_metrics = calculate_detection_metrics(eval_results)

                metrics = {
                    "model_key": model_key,
                    "run_name": run_name,
                    "type": model_type,
                    "clahe": is_clahe,
                    "split": split,
                    "inference_type": "sahi",
                }

                metrics["metrics/mAP50(B)"] = det_metrics["mAP"]
                metrics["metrics/mAP50-95(B)"] = calculate_map50_95(eval_results)

                # Add per-class APs
                for i, cls_name in CLASSES.items():
                    ap50_val = det_metrics["class_aps"].get(i, 0.0)
                    metrics[f"metrics/AP50({cls_name})"] = ap50_val

                # Save result dict to JSON
                with open(os.path.join(eval_dir, "results_dict.json"), "w") as f:
                    json.dump(metrics, f, indent=4)

                # Save local results.csv
                metrics_summary = {
                    "mAP50": metrics["metrics/mAP50(B)"],
                    "mAP50-95": metrics["metrics/mAP50-95(B)"],
                }
                for i, cls_name in CLASSES.items():
                    metrics_summary[f"AP50_{cls_name}"] = metrics[
                        f"metrics/AP50({cls_name})"
                    ]

                pd.DataFrame([metrics_summary]).to_csv(
                    os.path.join(eval_dir, "results.csv"), index=False
                )

    print(f"\nSAHI Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
