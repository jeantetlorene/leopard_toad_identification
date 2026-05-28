#!/usr/bin/env python3
import argparse
import json
import os
import sys
import shutil
import cv2
import torch
from ultralytics import YOLO, RTDETR
from ultralytics.data.dataset import YOLODataset

# Define path resolutions
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
DETECTION_DIR = os.path.dirname(ACTIVE_LEARNING_DIR)

# Dynamically add ACTIVE_LEARNING_DIR to sys.path for importing central config
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

# Centralized imports
from central_config import (
    DEFAULT_DEVICE,
    YOLO_TRAIN_CONFIG,
    RTDETR_TRAIN_CONFIG,
    FASTER_RCNN_TRAIN_CONFIG,
    TARGET_CLASSES,
    CLASS_MAPPING,
)
from dataset_utils import load_original_classes, resolve_target_classes, map_split
from faster_rcnn_utils import train_faster_rcnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Active Learning Model Training Suite."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["yolo", "rtdetr", "faster_rcnn"],
        required=True,
        help="Model architecture type to train.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrained", "scratch"],
        required=True,
        help="Train model starting from domain-pretrained weights or from scratch.",
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Active learning cycle number."
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Train model using CLAHE contrast preprocessing.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional custom experiment/run name to separate training runs, datasets, and candidates.",
    )
    return parser.parse_args()


def resolve_base_weights(model_type, mode):
    """Resolves the initial base weights for training cycles."""
    if mode == "pretrained":
        if model_type == "yolo":
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "detect",
                "yolo_finetuning",
                "weights",
                "best.pt",
            )
        elif model_type == "rtdetr":
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "detect",
                "rtdetr_finetuning",
                "weights",
                "best.pt",
            )
        else:  # faster_rcnn
            w_path = os.path.join(
                DETECTION_DIR,
                "pretraining",
                "runs",
                "faster_rcnn",
                "train_resnet50_1",
                "weights",
                "best.pt",
            )
            if not os.path.exists(w_path):
                w_path = os.path.join(
                    DETECTION_DIR,
                    "pretraining",
                    "runs",
                    "faster_rcnn",
                    "train_resnet50",
                    "weights",
                    "best.pt",
                )
    else:  # scratch mode
        if model_type == "yolo":
            w_path = os.path.join(ACTIVE_LEARNING_DIR, "yolo", "yolo26n.pt")
            if not os.path.exists(w_path):
                w_path = "yolo26n.pt"
        elif model_type == "rtdetr":
            w_path = os.path.join(ACTIVE_LEARNING_DIR, "rtdetr", "rtdetr-l.pt")
            if not os.path.exists(w_path):
                w_path = "rtdetr-l.pt"
        else:
            w_path = "scratch"

    return w_path


def train_ultralytics(
    model_class,
    model_type,
    weights,
    run_name,
    project_dir,
    dataset_yaml,
    freeze,
    epochs,
    patience,
    batch_size,
):
    """Generic training routine for Ultralytics models (YOLO & RT-DETR)."""
    model = model_class(weights)
    device_val = "0" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=batch_size,
        project=project_dir,
        name=run_name,
        freeze=freeze,
        device=device_val,
        verbose=False,
    )
    return os.path.join(project_dir, run_name, "weights", "best.pt")


def main():
    args = parse_args()

    # Resolve the model-specific training config dictionary
    if args.model_type == "yolo":
        train_config = YOLO_TRAIN_CONFIG
    elif args.model_type == "rtdetr":
        train_config = RTDETR_TRAIN_CONFIG
    elif args.model_type == "faster_rcnn":
        train_config = FASTER_RCNN_TRAIN_CONFIG
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Resolve folders and paths
    clahe_suffix = "clahe" if args.clahe else "plain"
    model_folder = (
        f"{args.model_type}_{clahe_suffix}" if args.clahe else args.model_type
    )

    model_dir = os.path.join(ACTIVE_LEARNING_DIR, model_folder)
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)

    # Add model_dir to sys.path to resolve internal trainer hooks
    if model_dir not in sys.path:
        sys.path.append(model_dir)

    # Dynamically apply CLAHE on-the-fly to YOLODataset loaded images for Ultralytics models
    if args.clahe and args.model_type in ["yolo", "rtdetr"]:
        original_load_image = YOLODataset.load_image

        def patched_load_image(self, i, *args, **kwargs):
            im, (h, w), (h0, w0) = original_load_image(self, i, *args, **kwargs)
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            im_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            return im_clahe, (h, w), (h0, w0)

        YOLODataset.load_image = patched_load_image
        print(
            "[Patch] Successfully monkey-patched YOLODataset.load_image to apply CLAHE on-the-fly!"
        )

    print(f"\n=======================================================")
    print(f"STARTING UNIFIED MODEL TRAINING")
    print(f"  Architecture:  {args.model_type.upper()}")
    print(f"  Preprocessing: {clahe_suffix.upper()}")
    print(f"  Mode:          {args.mode.upper()}")
    print(f"  Cycle:         {args.cycle}")
    print(f"  Model Dir:     {model_dir}")
    print("=======================================================")

    if args.experiment_name:
        dataset_dir = os.path.join(
            ACTIVE_LEARNING_DIR,
            "data",
            args.model_type,
            args.mode,
            args.experiment_name,
            f"cycle_{args.cycle}",
        )
    else:
        dataset_dir = os.path.join(
            ACTIVE_LEARNING_DIR,
            "data",
            args.model_type,
            args.mode,
            f"cycle_{args.cycle}",
        )

    # Resolve target classes dynamically and map dataset annotations
    print("\n--- RESOLVING DYNAMIC TARGET CLASSES ---")
    original_classes = load_original_classes(dataset_dir)
    resolved_classes, id_mapping = resolve_target_classes(
        original_classes, TARGET_CLASSES, CLASS_MAPPING
    )
    print(f"Original dataset classes: {original_classes}")
    print(f"Resolved target classes:   {resolved_classes}")
    print(f"Class mapping indices:     {id_mapping}")

    # Build lightweight mapped dataset with symlinked images and mapped labels
    dataset_dir_mapped = os.path.join(dataset_dir, "mapped")
    if os.path.exists(dataset_dir_mapped):
        shutil.rmtree(dataset_dir_mapped)
    os.makedirs(dataset_dir_mapped, exist_ok=True)

    splits_mapped = 0
    for split in ["train", "val", "test"]:
        if map_split(
            dataset_dir, dataset_dir_mapped, split, id_mapping, resolved_classes
        ):
            splits_mapped += 1

    print(f"Successfully mapped {splits_mapped} splits to: {dataset_dir_mapped}")

    cycle_parent = os.path.join(model_dir, "cycles", args.mode)
    if args.experiment_name:
        cycle_parent = os.path.join(cycle_parent, args.experiment_name)
    cycle_dir = os.path.join(cycle_parent, f"cycle_{args.cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    dataset_yaml = os.path.join(
        cycle_dir, f"dataset_{args.mode}_cycle_{args.cycle}.yaml"
    )

    # Create dataset yaml file dynamically using the resolved target classes
    names_content = "\n".join(
        [f"  {i}: {name}" for i, name in enumerate(resolved_classes)]
    )
    yaml_content = f"""path: {dataset_dir_mapped}
train: train/images
val: val/images
test: test/images

names:
{names_content}
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)

    print(f"Created dataset YAML at: {dataset_yaml}")

    # Resolve initial base weights
    base_weights = resolve_base_weights(args.model_type, args.mode)
    print(f"Base weights resolved: {base_weights}")

    # Establish project runs directories
    project_dir = os.path.join(model_dir, "runs")
    if args.experiment_name:
        project_dir = os.path.join(project_dir, args.experiment_name)
    os.makedirs(project_dir, exist_ok=True)

    # Define outputs paths based on cycle mode
    if args.mode == "pretrained":
        expected_p2_model = os.path.join(
            project_dir, f"cycle_{args.cycle}_pretrained_phase2", "weights", "best.pt"
        )
        if os.path.exists(expected_p2_model):
            print(
                f"Trained model for Cycle {args.cycle} (Phase 2) already exists at: {expected_p2_model}. Skipping training."
            )
            return

        p1_cfg = train_config["pretrained"]["phase1"]
        p2_cfg = train_config["pretrained"]["phase2"]

        print("\n--- PHASE 1: Fine-tune Head Only (Backbone Frozen) ---")
        if args.model_type in ["yolo", "rtdetr"]:
            model_class = YOLO if args.model_type == "yolo" else RTDETR

            p1_weights = train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase1",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=p1_cfg["freeze"],
                epochs=p1_cfg["epochs"],
                patience=p1_cfg["patience"],
                batch_size=p1_cfg["batch_size"],
            )

            print("\n--- PHASE 2: Adapt Entire Network (Backbone Unfrozen) ---")
            train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=p1_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase2",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=p2_cfg["freeze"],
                epochs=p2_cfg["epochs"],
                patience=p2_cfg["patience"],
                batch_size=p2_cfg["batch_size"],
            )
        else:  # faster_rcnn
            p1_weights = train_faster_rcnn(
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase1",
                project_dir=project_dir,
                dataset_dir=dataset_dir_mapped,
                freeze_backbone=p1_cfg["freeze_backbone"],
                epochs=p1_cfg["epochs"],
                patience=p1_cfg["patience"],
                batch_size=p1_cfg["batch_size"],
                apply_clahe=args.clahe,
                num_classes=len(resolved_classes),
            )

            print("\n--- PHASE 2: Adapt Entire Network (Backbone Unfrozen) ---")
            train_faster_rcnn(
                weights=p1_weights,
                run_name=f"cycle_{args.cycle}_pretrained_phase2",
                project_dir=project_dir,
                dataset_dir=dataset_dir_mapped,
                freeze_backbone=p2_cfg["freeze_backbone"],
                epochs=p2_cfg["epochs"],
                patience=p2_cfg["patience"],
                batch_size=p2_cfg["batch_size"],
                apply_clahe=args.clahe,
                num_classes=len(resolved_classes),
            )
    else:  # scratch mode
        expected_scratch_model = os.path.join(
            project_dir, f"cycle_{args.cycle}_scratch_scratch", "weights", "best.pt"
        )
        if os.path.exists(expected_scratch_model):
            print(
                f"Trained model for Cycle {args.cycle} (Scratch) already exists at: {expected_scratch_model}. Skipping training."
            )
            return

        scratch_cfg = train_config["scratch"]

        print("\n--- FROM-SCRATCH MODEL TRAINING ---")
        if args.model_type in ["yolo", "rtdetr"]:
            model_class = YOLO if args.model_type == "yolo" else RTDETR

            train_ultralytics(
                model_class=model_class,
                model_type=args.model_type,
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_scratch_scratch",
                project_dir=project_dir,
                dataset_yaml=dataset_yaml,
                freeze=scratch_cfg.get("freeze", 0),
                epochs=scratch_cfg["epochs"],
                patience=scratch_cfg["patience"],
                batch_size=scratch_cfg["batch_size"],
            )
        else:  # faster_rcnn
            train_faster_rcnn(
                weights=base_weights,
                run_name=f"cycle_{args.cycle}_scratch_scratch",
                project_dir=project_dir,
                dataset_dir=dataset_dir_mapped,
                freeze_backbone=scratch_cfg["freeze_backbone"],
                epochs=scratch_cfg["epochs"],
                patience=scratch_cfg["patience"],
                batch_size=scratch_cfg["batch_size"],
                apply_clahe=args.clahe,
                num_classes=len(resolved_classes),
            )

    print("\n=======================================================")
    print("UNIFIED MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
