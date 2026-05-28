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
        nargs="+",
        choices=["yolo", "rtdetr", "faster_rcnn"],
        required=True,
        help="Object detection architecture types to run (yolo, rtdetr, and/or faster_rcnn).",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Run loop with CLAHE contrast preprocessing enabled.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run loop with plain (non-CLAHE) images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        choices=["pretrained", "scratch"],
        default=["pretrained"],
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
        help="Reset active learning loop cycle tracker back to Cycle 0 for the resolved configurations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of model training, batch inference, and curation, bypassing any skipping/caching.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional custom experiment/run name to separate training runs, datasets, and candidates.",
    )

    args = parser.parse_args()

    # 1. Resolve combinations to execute
    models_to_run = args.model_type
    modes_to_run = args.mode

    preps_to_run = []
    if args.clahe:
        preps_to_run.append("clahe")
    if args.plain:
        preps_to_run.append("plain")
    if not preps_to_run:
        # Default behavior: run CLAHE only
        preps_to_run = ["clahe"]

    print("\n=======================================================")
    print(f"SCHEDULING ACTIVE LEARNING COMBINATIONS")
    print(f"  Models:        {', '.join(models_to_run).upper()}")
    print(f"  Modes:         {', '.join(modes_to_run).upper()}")
    print(f"  Preprocessing: {', '.join(preps_to_run).upper()}")
    print(f"  Experiment:    {args.experiment_name or 'default'}")
    print(f"  Force Flag:    {args.force}")
    print("=======================================================")

    # 2. PRE-FLIGHT SANITY CHECKS FOR ALL PLANNED RUNS
    print("\n>>> PERFORMING PRE-FLIGHT SANITY CHECKS...")
    all_ok = True

    for m_type in models_to_run:
        for mode_val in modes_to_run:
            for prep_val in preps_to_run:
                clahe_suffix = "clahe" if prep_val == "clahe" else "plain"
                exp_suffix = f"_{args.experiment_name}" if args.experiment_name else ""
                state_file = os.path.join(
                    DETECTION_DIR,
                    "active learning",
                    "pipelines",
                    f"al_state_{m_type}_{clahe_suffix}_{mode_val}{exp_suffix}.json",
                )

                if args.reset:
                    cycle = 0
                else:
                    state_tmp = load_state(state_file)
                    cycle = state_tmp["cycle"]

                # A. Verify initial starting dataset directory availability
                if args.experiment_name:
                    dataset_dir = os.path.join(
                        DETECTION_DIR,
                        "active learning",
                        "data",
                        m_type,
                        mode_val,
                        args.experiment_name,
                        f"cycle_{cycle}",
                    )
                else:
                    dataset_dir = os.path.join(
                        DETECTION_DIR,
                        "active learning",
                        "data",
                        m_type,
                        mode_val,
                        f"cycle_{cycle}",
                    )

                if not os.path.exists(dataset_dir):
                    print(
                        f"  [FAIL] {m_type.upper()} ({prep_val}, {mode_val}) Cycle {cycle} dataset folder NOT found at: {dataset_dir}"
                    )
                    all_ok = False
                else:
                    train_img_dir = os.path.join(dataset_dir, "train", "images")
                    yolo_train_img_dir = os.path.join(dataset_dir, "images", "train")
                    if not os.path.exists(train_img_dir) and not os.path.exists(
                        yolo_train_img_dir
                    ):
                        print(
                            f"  [FAIL] {m_type.upper()} ({prep_val}, {mode_val}) Cycle {cycle} dataset at {dataset_dir} does not contain training images!"
                        )
                        all_ok = False
                    else:
                        print(
                            f"  [ OK ] {m_type.upper()} ({prep_val}, {mode_val}) Cycle {cycle} dataset validated."
                        )

                # B. Verify pretrained weights if using 'pretrained' mode
                if mode_val == "pretrained":
                    if m_type == "yolo":
                        w_path = os.path.join(
                            DETECTION_DIR,
                            "pretraining",
                            "runs",
                            "detect",
                            "yolo_model",
                            "weights",
                            "best.pt",
                        )
                    elif m_type == "rtdetr":
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

                    if not os.path.exists(w_path):
                        print(
                            f"  [FAIL] {m_type.upper()} ({prep_val}, {mode_val}) Pretrained weights not found at: {w_path}"
                        )
                        all_ok = False
                    else:
                        print(
                            f"  [ OK ] {m_type.upper()} ({prep_val}, {mode_val}) Domain-pretrained weights found."
                        )

    if not all_ok:
        print(
            "\n>>> SANITY CHECKS FAILED! Please resolve the missing configurations or paths above before running.\n"
        )
        sys.exit(1)
    else:
        print(">>> ALL SANITY CHECKS PASSED SUCCESSFULLY!\n")

    # 3. CONCURRENT BATCH MODEL EXECUTION
    python_interpreter = sys.executable

    for m_type in models_to_run:
        for mode_val in modes_to_run:
            for prep_val in preps_to_run:
                clahe_suffix = "clahe" if prep_val == "clahe" else "plain"
                exp_suffix = f"_{args.experiment_name}" if args.experiment_name else ""
                state_file = os.path.join(
                    DETECTION_DIR,
                    "active learning",
                    "pipelines",
                    f"al_state_{m_type}_{clahe_suffix}_{mode_val}{exp_suffix}.json",
                )

                if args.reset:
                    if os.path.exists(state_file):
                        os.remove(state_file)
                    state = {"cycle": 0, "model_paths": {}}
                    save_state(state, state_file)
                else:
                    state = load_state(state_file)

                cycle = state["cycle"]

                print(f"\n=======================================================")
                print(
                    f"RUNNING CONFIGURATION: {m_type.upper()} | {prep_val.upper()} | {mode_val.upper()} | CYCLE {cycle}"
                )
                print(f"  State File: {state_file}")
                print("=======================================================")

                # Resolve run/outputs folders
                model_folder = (
                    f"{m_type}_{clahe_suffix}" if prep_val == "clahe" else m_type
                )
                runs_parent = os.path.join(
                    DETECTION_DIR, "active learning", model_folder, "runs"
                )
                if args.experiment_name:
                    runs_parent = os.path.join(runs_parent, args.experiment_name)

                weights_folder = (
                    f"cycle_{cycle}_{mode_val}_phase2"
                    if mode_val == "pretrained"
                    else f"cycle_{cycle}_{mode_val}_{mode_val}"
                )
                model_weight = os.path.join(
                    runs_parent, weights_folder, "weights", "best.pt"
                )

                results_parent = os.path.join(DETECTION_DIR, "results")
                if args.experiment_name:
                    results_parent = os.path.join(results_parent, args.experiment_name)
                output_dir = os.path.join(
                    results_parent,
                    f"detect_{m_type}_cycle{cycle}_{clahe_suffix}_{mode_val}",
                )

                unified_predictions_csv = os.path.join(
                    output_dir, "all_unlabeled_predictions.csv"
                )
                filtered_predictions_csv = os.path.join(
                    output_dir, "all_unlabeled_predictions_filtered.csv"
                )
                curation_priority_csv = os.path.join(
                    output_dir, "curation_priority.csv"
                )

                cycle_parent = os.path.join(
                    DETECTION_DIR, "active learning", model_folder, "cycles", mode_val
                )
                if args.experiment_name:
                    cycle_parent = os.path.join(cycle_parent, args.experiment_name)
                cycle_dir = os.path.join(cycle_parent, f"cycle_{cycle}")
                oracle_csv_path = os.path.join(
                    cycle_dir, f"al_query_candidates_{mode_val}_cycle_{cycle}.csv"
                )

                # Skip entire cycle if oracle candidates already exist
                if not args.force and os.path.exists(oracle_csv_path):
                    print(
                        f"\n[Skip] Entire Cycle {cycle} already completed for {m_type} ({prep_val}, {mode_val})."
                    )
                    print(f"       Found candidates at: {oracle_csv_path}")
                    continue

                # ----------------------------------------------------
                # PHASE 1: MODEL TRAINING
                # ----------------------------------------------------
                if not args.force and os.path.exists(model_weight):
                    print(
                        f"\n[Skip] Phase 1: Model training already completed. Found weights at: {model_weight}"
                    )
                else:
                    print(
                        f"\n--- [Phase 1: Model Training] Training Cycle {cycle} Model ---"
                    )
                    training_script = os.path.join(PIPELINES_DIR, "train_model.py")

                    train_cmd = [
                        python_interpreter,
                        training_script,
                        "--model_type",
                        m_type,
                        "--mode",
                        mode_val,
                        "--cycle",
                        str(cycle),
                    ]
                    if prep_val == "clahe":
                        train_cmd.append("--clahe")
                    if args.experiment_name:
                        train_cmd.extend(["--experiment_name", args.experiment_name])

                    run_command(
                        train_cmd,
                        f"{m_type.upper()} ({prep_val}, {mode_val}) Cycle {cycle} Model Training",
                    )

                if not os.path.exists(model_weight):
                    print(
                        f"Error: Trained model weights file not found at '{model_weight}'."
                    )
                    sys.exit(1)

                state["model_paths"][mode_val] = model_weight
                print(f"Trained model verified at: {model_weight}")

                # ----------------------------------------------------
                # PHASE 2: AUTOMATED BATCH INFERENCE & FILTERING
                # ----------------------------------------------------
                if (
                    not args.force
                    and os.path.exists(unified_predictions_csv)
                    and os.path.exists(filtered_predictions_csv)
                ):
                    print(
                        f"\n[Skip] Phase 2: Batch inference & static filtering already completed. Found predictions at: {filtered_predictions_csv}"
                    )
                elif not args.force and os.path.exists(unified_predictions_csv):
                    print(
                        f"\n[Skip] Phase 2: Batch inference already completed (static filter bypassed). Found predictions at: {unified_predictions_csv}"
                    )
                else:
                    print(
                        f"\n--- [Phase 2: Batch Inference] Running predictions on unlabeled pool ---"
                    )
                    inference_script = os.path.join(
                        PIPELINES_DIR, "run_inference_pipeline.py"
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
                    if prep_val == "clahe":
                        infer_cmd.append("--apply_clahe")
                    else:
                        infer_cmd.append("--no_clahe")

                    infer_cmd.append("--filter_static")

                    run_command(
                        infer_cmd,
                        f"{m_type.upper()} ({prep_val}, {mode_val}) Batch Inference & Static Filter",
                    )

                # ----------------------------------------------------
                # PHASE 3: CATEGORY-BIASED ACTIVE CURATION
                # ----------------------------------------------------
                if not args.force and os.path.exists(curation_priority_csv):
                    print(
                        f"\n[Skip] Phase 3: Active curation already completed. Found priority CSV at: {curation_priority_csv}"
                    )
                else:
                    print(
                        f"\n--- [Phase 3: Active Curation] Selecting diverse priority annotations ---"
                    )
                    curation_script = os.path.join(PIPELINES_DIR, "active_curation.py")

                    curate_cmd = [
                        python_interpreter,
                        curation_script,
                        "--consensus_csv",
                        unified_predictions_csv,
                        "--output_csv",
                        curation_priority_csv,
                        "--n_clusters",
                        str(args.budget),
                        "--iou_threshold",
                        str(args.iou_threshold),
                        "--occurrence_threshold",
                        str(args.occurrence_threshold),
                    ]
                    run_command(
                        curate_cmd,
                        f"{m_type.upper()} ({prep_val}, {mode_val}) Active Learning Curation Selection",
                    )

                # ----------------------------------------------------
                # PHASE 4: ORACLE QUERY EXPORT
                # ----------------------------------------------------
                if not args.force and os.path.exists(oracle_csv_path):
                    print(
                        f"\n[Skip] Phase 4: Oracle query export already completed. Found candidates at: {oracle_csv_path}"
                    )
                    representatives_count = (
                        len(pd.read_csv(oracle_csv_path))
                        if os.path.exists(oracle_csv_path)
                        else args.budget
                    )
                else:
                    print(
                        f"\n--- [Phase 4: Oracle Export] Generating priority queries ---"
                    )
                    if not os.path.exists(curation_priority_csv):
                        print(
                            f"Error: Curation priority CSV file not found at '{curation_priority_csv}'."
                        )
                        sys.exit(1)

                    curation_df = pd.read_csv(curation_priority_csv)
                    representatives = curation_df[
                        curation_df["is_representative"] == True
                    ]
                    representatives_count = len(representatives)

                    os.makedirs(cycle_dir, exist_ok=True)
                    write_candidates_csv(
                        representatives.to_dict("records"), oracle_csv_path
                    )

                # ----------------------------------------------------
                # PHASE 5: CYCLE INCREMENT & UPDATE STATE
                # ----------------------------------------------------
                state["cycle"] += 1
                save_state(state, state_file)

                print("\n=======================================================")
                print(
                    f"CONFIGURATION COMPLETED: {m_type.upper()} | {prep_val.upper()} | {mode_val.upper()} | CYCLE {cycle}"
                )
                print(f"  [ORACLE PAUSE] Exported {representatives_count} queries to:")
                print(f"                 {oracle_csv_path}")
                print("=======================================================\n")


if __name__ == "__main__":
    main()
