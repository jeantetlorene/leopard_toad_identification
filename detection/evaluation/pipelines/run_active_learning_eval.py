import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import cv2
import shutil
import argparse

# Project imports
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import MODEL_ROOTS, DEVICE, CLASSES, DEFAULT_BATCH_SIZE
from eval_utils.data_utils import apply_clahe
from eval_utils.metrics import calculate_detection_metrics, calculate_map50_95
from eval_utils.models.faster_rcnn_wrapper import FasterRCNNWrapper
from ultralytics import YOLO, RTDETR
from eval_utils.inference import generate_predictions

DATA_DIR = (
    "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation/data"
)
RESULTS_DIR_FILES = "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation/results/files"


def prepare_clahe_set(split, overwrite=False):
    """
    Prepares a CLAHE-processed version of the dataset split.
    Saves to data/<split>_clahe/ structure compatible with YOLO.
    """
    split_dir = os.path.join(DATA_DIR, split)
    clahe_root = os.path.join(DATA_DIR, f"{split}_clahe")
    clahe_images_dir = os.path.join(clahe_root, "images")
    clahe_labels_dir = os.path.join(clahe_root, "labels")

    os.makedirs(clahe_images_dir, exist_ok=True)

    # Symlink labels for YOLO compatibility
    if not os.path.exists(clahe_labels_dir):
        print(f"Creating symlink for {split} labels -> {clahe_labels_dir}")
        os.symlink(os.path.join(split_dir, "labels"), clahe_labels_dir)

    # Tracking manifest
    manifest_path = os.path.join(clahe_root, "manifest.csv")

    # Check if already processed
    if not overwrite and len(os.listdir(clahe_images_dir)) > 0:
        print(f"CLAHE set for {split} already exists. Skipping generation.")
        return clahe_root

    print(f"Generating CLAHE-processed {split} set...")
    images_dir = os.path.join(split_dir, "images")

    # Load mapping for manifest tracking
    from eval_utils.data_utils import get_image_mapping

    mapping_df = get_image_mapping()
    manifest_data = []

    for img_name in tqdm(os.listdir(images_dir)):
        if img_name.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_clahe = apply_clahe(img)
                cv2.imwrite(os.path.join(clahe_images_dir, img_name), img_clahe)

                # Track original path if available
                orig_path = "Unknown"
                if not mapping_df.empty:
                    match = mapping_df[mapping_df["unique_name"] == img_name]
                    if not match.empty:
                        orig_path = match["original_path"].values[0]

                manifest_data.append(
                    {"unique_name": img_name, "original_path": orig_path}
                )

    # Save manifest
    pd.DataFrame(manifest_data).to_csv(manifest_path, index=False)
    print(f"Saved tracking manifest to {manifest_path}")

    return clahe_root


def create_yaml(split_dir, images_subdir, yaml_path):
    # Ultralytics requires train and val keys even for test/val evaluation
    content = f"""
path: {split_dir}
train: images
val: images
test: {images_subdir}
labels: labels
names:
  0: Other_Amphibian
  1: Small_Mammal
  2: Western_Leopard_Toad
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def run_evaluation(overwrite=False):
    os.makedirs(RESULTS_DIR_FILES, exist_ok=True)

    splits = ["test", "val"]

    # Prepare sets and YAMLs
    yamls = {}
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        clahe_root = prepare_clahe_set(split, overwrite=overwrite)

        plain_yaml = os.path.join(DATA_DIR, f"{split}_plain.yaml")
        create_yaml(split_dir, "images", plain_yaml)

        clahe_yaml = os.path.join(DATA_DIR, f"{split}_clahe.yaml")
        create_yaml(clahe_root, "images", clahe_yaml)

        yamls[split] = {"plain": plain_yaml, "clahe": clahe_yaml}

    frcnn_combined_results = []

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

        print(f"\nProcessing {model_key} models...")

        for run_name in sorted(os.listdir(runs_dir)):
            model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
            if not os.path.exists(model_path):
                continue

            for split in splits:
                eval_name = f"{split}_eval"
                eval_dir = os.path.join(runs_dir, run_name, eval_name)

                # Check if evaluation already exists
                if not overwrite and os.path.exists(
                    os.path.join(eval_dir, "results_dict.json")
                ):
                    print(f"Skipping {run_name} on {split} (already evaluated).")
                    # If Faster R-CNN, we still want to load it for the combined CSV if it was missed
                    if model_type == "faster_rcnn":
                        with open(
                            os.path.join(eval_dir, "results_dict.json"), "r"
                        ) as f:
                            frcnn_combined_results.append(json.load(f))
                    continue

                # Ensure the eval dir exists
                os.makedirs(eval_dir, exist_ok=True)

                print(f"Evaluating {run_name} on {split} (CLAHE={is_clahe})...")
                yaml_to_use = (
                    yamls[split]["clahe"] if is_clahe else yamls[split]["plain"]
                )

                if model_type in ["yolo", "rtdetr"]:
                    ModelClass = YOLO if model_type == "yolo" else RTDETR
                    model = ModelClass(model_path)

                    # exist_ok=True prevents creating multiple folders like test_eval, test_eval2, etc.
                    val_results = model.val(
                        data=yaml_to_use,
                        split="test",
                        batch=256,
                        device=DEVICE,
                        verbose=False,
                        plots=True,
                        save_json=True,
                        project=os.path.join(runs_dir, run_name),
                        name=eval_name,
                        exist_ok=True,
                    )

                    # Save the full results dictionary to JSON
                    results_dict = val_results.results_dict
                    # Add per-class APs for easier access
                    for i, cls_name in CLASSES.items():
                        # Safety check for empty or short metrics arrays (e.g. if no labels found)
                        ap50_val = 0.0
                        if hasattr(val_results, "box") and hasattr(
                            val_results.box, "ap50"
                        ):
                            if i < len(val_results.box.ap50):
                                ap50_val = float(val_results.box.ap50[i])

                        results_dict[f"metrics/AP50({cls_name})"] = ap50_val

                    with open(os.path.join(eval_dir, "results_dict.json"), "w") as f:
                        json.dump(results_dict, f, indent=4)

                    # Also save key metrics to local results.csv
                    metrics_summary = {
                        "mAP50": results_dict.get("metrics/mAP50(B)", 0.0),
                        "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0.0),
                        "precision": results_dict.get("metrics/precision(B)", 0.0),
                        "recall": results_dict.get("metrics/recall(B)", 0.0),
                    }
                    for i, cls_name in CLASSES.items():
                        metrics_summary[f"AP50_{cls_name}"] = results_dict.get(
                            f"metrics/AP50({cls_name})", 0.0
                        )

                    pd.DataFrame([metrics_summary]).to_csv(
                        os.path.join(eval_dir, "results.csv"), index=False
                    )

                else:
                    # Faster R-CNN
                    metrics = {
                        "model_key": model_key,
                        "run_name": run_name,
                        "type": model_type,
                        "clahe": is_clahe,
                        "split": split,
                    }
                    wrapper = FasterRCNNWrapper(model_path, device=DEVICE)

                    eval_results = generate_predictions(
                        wrapper,
                        split,
                        use_clahe=is_clahe,
                        batch_size=64,
                    )

                    det_metrics = calculate_detection_metrics(eval_results)
                    metrics["mAP50"] = det_metrics["mAP"]
                    metrics["mAP50-95"] = calculate_map50_95(eval_results)
                    for i, cls_name in CLASSES.items():
                        metrics[f"AP50_{cls_name}"] = det_metrics["class_aps"].get(
                            i, 0.0
                        )

                    # Save result dict to JSON
                    with open(os.path.join(eval_dir, "results_dict.json"), "w") as f:
                        json.dump(metrics, f, indent=4)

                    # Save local results.csv
                    local_metrics = {
                        "mAP50": metrics["mAP50"],
                        "mAP50-95": metrics["mAP50-95"],
                    }
                    for i, cls_name in CLASSES.items():
                        local_metrics[f"AP50_{cls_name}"] = metrics[f"AP50_{cls_name}"]

                    pd.DataFrame([local_metrics]).to_csv(
                        os.path.join(eval_dir, "results.csv"), index=False
                    )

                    frcnn_combined_results.append(metrics)

    # Save the combined Faster R-CNN results at the end
    if frcnn_combined_results:
        os.makedirs(RESULTS_DIR_FILES, exist_ok=True)
        pd.DataFrame(frcnn_combined_results).to_csv(
            os.path.join(RESULTS_DIR_FILES, "active_learning_frcnn_combined_eval.csv"),
            index=False,
        )

    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Active Learning Evaluation Pipeline"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-evaluation of all models and re-generation of CLAHE sets.",
    )
    args = parser.parse_args()

    run_evaluation(overwrite=args.overwrite)
