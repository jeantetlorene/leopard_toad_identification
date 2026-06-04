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
    target_datasets=None,
    model_path=None,
    model_type=None,
    processing_type=None,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_types = ["yolo", "rtdetr", "faster_rcnn"]
    processing_types = ["plain", "clahe"]
    datasets = target_datasets if target_datasets else ["test", "val"]

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

    # Determine run configurations: list of (m_type, p_type, cycle, variant, path_to_run)
    run_configs = []
    if model_path:
        # Extract cycle and variant from the path if matching cycle_*_*
        cycle = target_cycles[0] if target_cycles else 0
        var = target_variants[0] if target_variants else "custom"
        path_parts = model_path.replace("\\", "/").split("/")
        for part in path_parts:
            if part.startswith("cycle_"):
                parts = part.split("_")
                if len(parts) >= 3:
                    try:
                        if not target_cycles:
                            cycle = int(parts[1])
                        if not target_variants:
                            var = parts[2]
                        break
                    except ValueError:
                        continue

        if not model_type:
            path_lower = model_path.lower()
            if "rtdetr" in path_lower:
                model_type = "rtdetr"
            elif "faster_rcnn" in path_lower or "fasterrcnn" in path_lower:
                model_type = "faster_rcnn"
            elif "yolo" in path_lower:
                model_type = "yolo"
            else:
                model_type = "yolo"

        if not processing_type:
            if "clahe" in model_path.lower():
                processing_type = "clahe"
            else:
                processing_type = "plain"

        run_configs.append((model_type, processing_type, cycle, var, model_path))
    else:
        for m_type in model_types:
            if target_models and m_type not in target_models:
                continue

            for p_type in processing_types:
                if processing_type and p_type != processing_type:
                    continue
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
                        run_configs.append((m_type, p_type, cycle, var, best_pt))

    for m_type, p_type, cycle, var, path_to_run in run_configs:
        res_dir = os.path.join(RESULTS_DIR, f"{m_type}_{p_type}")
        os.makedirs(res_dir, exist_ok=True)

        # Check if all datasets have cached predictions
        all_cached = True
        for ds_name in datasets:
            file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
            if full_sequence:
                file_prefix += "_full_seq"
            raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")

            if overwrite:
                all_cached = False
                break

            # Fast check to see if predictions are cached without parsing massive JSON
            is_cached = False
            if os.path.exists(raw_file):
                try:
                    with open(raw_file, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        if size >= 2:
                            # Verify the file is complete and not truncated
                            seek_pos = max(0, size - 100)
                            f.seek(seek_pos)
                            tail = f.read().strip()
                            if tail.endswith(b"]") or tail.endswith(b"}"):
                                # Count items by counting '"path":' occurrences
                                f.seek(0)
                                count = 0
                                chunk_size = 1024 * 1024
                                overlap = b""
                                search_str = b'"path":'
                                while True:
                                    chunk = f.read(chunk_size)
                                    if not chunk:
                                        break
                                    count += (overlap + chunk).count(search_str)
                                    overlap = chunk[-len(search_str) :]

                                if count >= expected_lengths[ds_name]:
                                    is_cached = True
                except OSError:
                    pass

            if not is_cached:
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

            wrapper = UltralyticsWrapper(m_type, path_to_run, device=DEVICE)
        else:
            from eval_utils.models.faster_rcnn_wrapper import (
                FasterRCNNWrapper,
            )

            wrapper = FasterRCNNWrapper(path_to_run, device=DEVICE)

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
                            processed_paths = {res["path"] for res in existing_results}
                    except (json.JSONDecodeError, KeyError):
                        print("Failed to read existing JSON, starting fresh.")
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
    if model_path:
        run_evaluation_suite(
            target_models=[model_type],
            target_processing=[processing_type],
            target_cycles=[cycle],
            target_variants=[var],
        )
    else:
        run_evaluation_suite(
            target_models=target_models,
            target_cycles=target_cycles,
            target_variants=target_variants,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--models",
        "--model",
        nargs="+",
        default=None,
        help="Target models (e.g. yolo, rtdetr, faster_rcnn)",
    )
    parser.add_argument("--cycles", type=int, nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Run over the entire full camera sequence instead of just ground-truth pool.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate (e.g., test val)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files instead of appending/resuming.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a specific model checkpoint to run inference/evaluation on.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["yolo", "rtdetr", "faster_rcnn"],
        help="Model type for the specified model_path (yolo, rtdetr, faster_rcnn).",
    )
    parser.add_argument(
        "--processing_type",
        type=str,
        default=None,
        choices=["plain", "clahe"],
        help="Processing type for the specified model_path (plain, clahe).",
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
        target_datasets=args.datasets,
        model_path=args.model_path,
        model_type=args.model_type,
        processing_type=args.processing_type,
    )
