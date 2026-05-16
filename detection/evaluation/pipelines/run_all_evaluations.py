import os
import argparse
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import MODEL_ROOTS, RESULTS_DIR, DEVICE, DATASETS
from eval_utils.inference import generate_predictions
from eval_utils.evaluation_suite import run_evaluation_suite
from eval_utils.data_utils import get_camera_images, get_dataset_images


def run_all(
    limit=None,
    batch_size=32,
    target_models=None,
    target_cycles=None,
    target_variants=None,
    full_sequence=False,
    overwrite=False,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_types = ["yolo", "rtdetr", "faster_rcnn"]
    processing_types = ["plain", "clahe"]
    datasets = ["test", "val"]

    expected_lengths = {}
    for ds_name in datasets:
        if full_sequence:
            expected_lengths[ds_name] = len(
                get_camera_images(DATASETS[ds_name]["camera"])
            )
        else:
            expected_lengths[ds_name] = len(get_dataset_images(ds_name))
        if limit:
            expected_lengths[ds_name] = min(expected_lengths[ds_name], limit)

    for m_type in model_types:
        if target_models and m_type not in target_models:
            continue

        for p_type in processing_types:
            root_key = f"{m_type}_{p_type}" if p_type == "clahe" else m_type
            root_dir = MODEL_ROOTS.get(root_key)
            if not root_dir or not os.path.exists(root_dir):
                print(f"Skipping {m_type} {p_type}: root dir not found")
                continue

            runs_dir = os.path.join(root_dir, "runs")
            if not os.path.exists(runs_dir):
                continue

            # Dynamically discover cycles and variants
            model_cycles = set()
            model_variants = set()
            for run_name in os.listdir(runs_dir):
                if run_name.startswith("cycle_"):
                    parts = run_name.split("_")
                    if len(parts) >= 3:
                        try:
                            model_cycles.add(int(parts[1]))
                            model_variants.add(parts[2])
                        except ValueError:
                            continue

            model_cycles = sorted(list(model_cycles))
            model_variants = sorted(list(model_variants))

            for cycle in model_cycles:
                if target_cycles and cycle not in target_cycles:
                    continue

                for var in model_variants:
                    if target_variants and var not in target_variants:
                        continue

                    # Dynamically find the best.pt for this cycle and variant
                    best_pt = None
                    for folder in os.listdir(runs_dir):
                        if folder.startswith(f"cycle_{cycle}_{var}_"):
                            path_attempt = os.path.join(
                                runs_dir, folder, "weights", "best.pt"
                            )
                            if os.path.exists(path_attempt):
                                best_pt = path_attempt
                                # Prefer phase2 or scratch over phase1 if available
                                if "phase2" in folder or "scratch" in folder:
                                    break

                    if not best_pt:
                        continue
                    model_path = best_pt

                    res_dir = os.path.join(RESULTS_DIR, f"{m_type}_{p_type}")
                    os.makedirs(res_dir, exist_ok=True)

                    # Check if all datasets have cached predictions
                    all_cached = True
                    for ds_name in datasets:
                        file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
                        if full_sequence:
                            file_prefix += "_full_seq"
                        raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")

                        if not os.path.exists(raw_file) or overwrite:
                            all_cached = False
                            break

                        try:
                            with open(raw_file, "r") as f:
                                existing_data = json.load(f)
                                if len(existing_data) < expected_lengths[ds_name]:
                                    all_cached = False
                                    break
                        except (json.JSONDecodeError, KeyError):
                            all_cached = False
                            break

                    if all_cached:
                        print(
                            f"\n>>> Skipping Inference for {m_type} | {p_type} | Cycle {cycle} | {var} (Predictions cached)"
                        )
                        continue

                    print(
                        f"\n>>> Running Inference for {m_type} | {p_type} | Cycle {cycle} | {var}"
                    )

                    # Lazy import to save memory/time if all are cached
                    if m_type in ["yolo", "rtdetr"]:
                        from eval_utils.models.ultralytics_wrapper import (
                            UltralyticsWrapper,
                        )

                        wrapper = UltralyticsWrapper(m_type, model_path, device=DEVICE)
                    else:
                        from eval_utils.models.faster_rcnn_wrapper import (
                            FasterRCNNWrapper,
                        )

                        wrapper = FasterRCNNWrapper(model_path, device=DEVICE)

                    for ds_name in datasets:
                        file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
                        if full_sequence:
                            file_prefix += "_full_seq"
                        raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")

                        existing_results = []
                        processed_paths = set()

                        if os.path.exists(raw_file):
                            if overwrite:
                                print(f"Overwriting existing dataset {ds_name}...")
                            else:
                                print(
                                    f"Found existing predictions for {ds_name}, attempting to resume..."
                                )
                                try:
                                    with open(raw_file, "r") as f:
                                        existing_results = json.load(f)
                                        processed_paths = {
                                            res["path"] for res in existing_results
                                        }
                                except (json.JSONDecodeError, KeyError):
                                    print(
                                        "Failed to read existing JSON, starting fresh."
                                    )
                                    existing_results = []
                                    processed_paths = set()

                        use_clahe = p_type == "clahe"

                        raw_results = generate_predictions(
                            wrapper,
                            ds_name,
                            use_clahe=use_clahe,
                            limit=limit,
                            batch_size=batch_size,
                            full_sequence=full_sequence,
                            processed_paths=processed_paths,
                            output_file=raw_file,
                            existing_results=existing_results,
                        )

    print("\n=========================================")
    print(">>> All predictions generated. Running Evaluation Suite...")
    print("=========================================")
    run_evaluation_suite(
        target_models=target_models,
        target_cycles=target_cycles,
        target_variants=target_variants,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cycles", type=int, nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Run over the entire full camera sequence instead of just ground-truth pool.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files instead of appending/resuming.",
    )
    args = parser.parse_args()

    run_all(
        limit=args.limit,
        batch_size=args.batch_size,
        target_models=args.models,
        target_cycles=args.cycles,
        target_variants=args.variants,
        full_sequence=args.full_sequence,
        overwrite=args.overwrite,
    )
