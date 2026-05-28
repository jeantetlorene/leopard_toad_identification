#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import subprocess
import pandas as pd

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

# Import central configurations
from config import (
    DEFAULT_CURATION_BUDGET,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    DETECTION_DIR,
)


def load_state(state_file):
    """Loads the active learning cycle state from a JSON file."""
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"cycle": 0, "model_paths": {}}


def save_state(state, state_file):
    """Saves the active learning cycle state to a JSON file."""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)


def run_command(cmd, desc):
    """Helper to run shell subprocesses with print logging."""
    print(f"\n>>> Running: {desc}...")
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"Error: {desc} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def write_candidates_csv(selected_rows, csv_path):
    """Writes the curated oracle queries to a clean CSV file for Label Studio auditing."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "image_name",
                "subfolder",
                "class_name",
                "confidence",
                "curation_reason",
                "status",
            ]
        )
        for row in selected_rows:
            writer.writerow(
                [
                    row["image_path"],
                    row["image_name"],
                    row["subfolder"],
                    row["class_name"],
                    row["confidence"],
                    row["curation_reason"],
                    "To annotate",
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Unified Active Learning Loop Orchestrator for all object detection models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["yolo", "rtdetr", "faster_rcnn"],
        required=True,
        help="Object detection architecture type to run (yolo, rtdetr, or faster_rcnn).",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        default=True,
        help="Run loop with CLAHE contrast preprocessing enabled.",
    )
    parser.add_argument(
        "--no_clahe",
        action="store_false",
        dest="clahe",
        help="Run loop with plain (non-CLAHE) images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrained", "scratch"],
        default="pretrained",
        help="Run active learning loop using either pretrained weights or training from scratch.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_CURATION_BUDGET,
        help=f"Total human annotation budget (n_clusters) per cycle (default: {DEFAULT_CURATION_BUDGET}).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for clustering static background trigger boxes (default: {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help=f"Triggers count threshold for identifying static trigger boxes (default: {DEFAULT_OCCURRENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset active learning loop cycle tracker back to Cycle 0.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional custom experiment/run name to separate training runs, datasets, and candidates."
    )

    args = parser.parse_args()

    # Define State JSON file path
    clahe_suffix = "clahe" if args.clahe else "plain"
    exp_suffix = f"_{args.experiment_name}" if args.experiment_name else ""
    state_file = os.path.join(
        DETECTION_DIR,
        "active learning",
        "pipelines",
        f"al_state_{args.model_type}_{clahe_suffix}_{args.mode}{exp_suffix}.json",
    )

    if args.reset:
        print(
            f"Resetting active learning state for {args.model_type} ({clahe_suffix}, {args.mode}, {args.experiment_name or 'default'})..."
        )
        if os.path.exists(state_file):
            os.remove(state_file)
        state = {"cycle": 0, "model_paths": {}}
        save_state(state, state_file)
    else:
        state = load_state(state_file)

    cycle = state["cycle"]

    print("\n=======================================================")
    print(f"STARTING UNIFIED ACTIVE LEARNING LOOP")
    print(f"  Model Type:   {args.model_type.upper()}")
    print(f"  Preprocessing: {clahe_suffix.upper()}")
    print(f"  Mode:         {args.mode.upper()}")
    print(f"  Cycle Num:    {cycle}")
    print(f"  State File:   {state_file}")
    print("=======================================================")

    # Resolve paths relative to Project root
    python_interpreter = sys.executable

    # ----------------------------------------------------
    # PHASE 1: MODEL TRAINING
    # ----------------------------------------------------
    print(f"\n--- [Phase 1: Model Training] Training Cycle {cycle} Model ---")

    # Construct script path for training
    training_script = os.path.join(PIPELINES_DIR, "train_model.py")

    if not os.path.exists(training_script):
        print(f"Error: Training script not found at '{training_script}'.")
        return

    # Trigger model training via subprocess
    train_cmd = [
        python_interpreter,
        training_script,
        "--model_type",
        args.model_type,
        "--mode",
        args.mode,
        "--cycle",
        str(cycle),
    ]
    if args.clahe:
        train_cmd.append("--clahe")
    if args.experiment_name:
        train_cmd.extend(["--experiment_name", args.experiment_name])

    run_command(train_cmd, f"Cycle {cycle} Model Training")

    # Resolve newly trained model path
    # Weights are saved under runs/ relative to model directory
    model_folder = (
        f"{args.model_type}_{clahe_suffix}" if args.clahe else args.model_type
    )
    runs_parent = os.path.join(DETECTION_DIR, "active learning", model_folder, "runs")
    if args.experiment_name:
        runs_parent = os.path.join(runs_parent, args.experiment_name)

    weights_folder = (
        f"cycle_{cycle}_{args.mode}_phase2"
        if args.mode == "pretrained"
        else f"cycle_{cycle}_{args.mode}_{args.mode}"
    )
    model_weight = os.path.join(
        runs_parent,
        weights_folder,
        "weights",
        "best.pt",
    )

    if not os.path.exists(model_weight):
        print(f"Error: Trained model weights file not found at '{model_weight}'.")
        return

    state["model_paths"][args.mode] = model_weight
    print(f"Trained model verified at: {model_weight}")

    # ----------------------------------------------------
    # PHASE 2: AUTOMATED BATCH INFERENCE
    # ----------------------------------------------------
    print(f"\n--- [Phase 2: Batch Inference] Running predictions on unlabeled pool ---")
    inference_script = os.path.join(PIPELINES_DIR, "run_inference_pipeline.py")
    results_parent = os.path.join(DETECTION_DIR, "results")
    if args.experiment_name:
        results_parent = os.path.join(results_parent, args.experiment_name)
    output_dir = os.path.join(
        results_parent,
        f"detect_{args.model_type}_cycle{cycle}_{clahe_suffix}_{args.mode}",
    )

    infer_cmd = [
        python_interpreter,
        inference_script,
        "--model_path",
        model_weight,
        "--output_dir",
        output_dir,
        "--iou_threshold",
        str(args.iou_threshold),
        "--occurrence_threshold",
        str(args.occurrence_threshold),
    ]
    if args.clahe:
        infer_cmd.append("--apply_clahe")
    else:
        infer_cmd.append("--no_clahe")

    # Auto-filter switch tells inference script to run spatial static bounding box filter immediately
    infer_cmd.append("--filter_static")

    run_command(infer_cmd, "Batch Inference & Static Filter")

    unified_predictions_csv = os.path.join(output_dir, "all_unlabeled_predictions.csv")
    filtered_predictions_csv = os.path.join(
        output_dir, "all_unlabeled_predictions_filtered.csv"
    )

    # ----------------------------------------------------
    # PHASE 3: CATEGORY-BIASED ACTIVE CURATION
    # ----------------------------------------------------
    print(
        f"\n--- [Phase 3: Active Curation] Selecting diverse priority annotations ---"
    )
    curation_script = os.path.join(PIPELINES_DIR, "active_curation.py")
    curation_priority_csv = os.path.join(output_dir, "curation_priority.csv")

    curate_cmd = [
        python_interpreter,
        curation_script,
        "--consensus_csv",
        unified_predictions_csv,  # pass unified predictions so it flags triggers itself
        "--output_csv",
        curation_priority_csv,
        "--n_clusters",
        str(args.budget),
        "--iou_threshold",
        str(args.iou_threshold),
        "--occurrence_threshold",
        str(args.occurrence_threshold),
    ]
    run_command(curate_cmd, "Active Learning Curation Priority Selection")

    # ----------------------------------------------------
    # PHASE 4: ORACLE QUERY EXPORT
    # ----------------------------------------------------
    print(f"\n--- [Phase 4: Oracle Export] Generating priority queries ---")
    if not os.path.exists(curation_priority_csv):
        print(
            f"Error: Curation priority CSV file not found at '{curation_priority_csv}'."
        )
        return

    curation_df = pd.read_csv(curation_priority_csv)
    representatives = curation_df[curation_df["is_representative"] == True]

    cycle_parent = os.path.join(DETECTION_DIR, "active learning", model_folder, "cycles", args.mode)
    if args.experiment_name:
        cycle_parent = os.path.join(cycle_parent, args.experiment_name)
    cycle_dir = os.path.join(cycle_parent, f"cycle_{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    oracle_csv_path = os.path.join(
        cycle_dir, f"al_query_candidates_{args.mode}_cycle_{cycle}.csv"
    )
    write_candidates_csv(representatives.to_dict("records"), oracle_csv_path)

    # ----------------------------------------------------
    # PHASE 5: CYCLE INCREMENT & UPDATE STATE
    # ----------------------------------------------------
    state["cycle"] += 1
    save_state(state, state_file)

    print("\n=======================================================")
    print("CYCLE INFERENCE & CURATION LOOP COMPLETED")
    print(f"  [ORACLE PAUSE] Exported {len(representatives)} queries to:")
    print(f"                 {oracle_csv_path}")
    print("-------------------------------------------------------")
    print("INSTRUCTIONS FOR HUMAN ANNOTATOR:")
    print("  1. Review and annotate these prioritized images in Label Studio.")
    print("  2. Combine all previous training images with these new annotated images.")
    print(f"  3. Save the new combined training dataset in the next cycle folder:")
    data_subpath = f"{args.model_type}/{args.mode}/{args.experiment_name}/cycle_{cycle + 1}/" if args.experiment_name else f"{args.model_type}/{args.mode}/cycle_{cycle + 1}/"
    print(f"     detection/active learning/data/{data_subpath}")
    print("  4. Rerun this unified loop script to start the next cycle!")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
