import os
import argparse
import json
from config import MODEL_ROOTS, RESULTS_DIR, DEVICE
from inference import generate_predictions
from evaluation_suite import run_evaluation_suite


def run_all(
    limit=None,
    batch_size=32,
    target_models=None,
    target_cycles=None,
    target_variants=None,
    full_sequence=False,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_types = ["yolo", "rtdetr", "faster_rcnn"]
    processing_types = ["plain", "clahe"]
    variants = ["pretrained", "scratch"]
    cycles = range(5)
    datasets = ["test", "val"]

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

            for cycle in cycles:
                if target_cycles and cycle not in target_cycles:
                    continue

                for var in variants:
                    if target_variants and var not in target_variants:
                        continue

                    run_name = (
                        f"cycle_{cycle}_{var}_phase2"
                        if var == "pretrained"
                        else f"cycle_{cycle}_{var}_scratch"
                    )
                    model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

                    if not os.path.exists(model_path):
                        if var == "pretrained":
                            run_name_p1 = f"cycle_{cycle}_{var}_phase1"
                            model_path = os.path.join(
                                runs_dir, run_name_p1, "weights", "best.pt"
                            )
                            if not os.path.exists(model_path):
                                continue
                        else:
                            continue

                    res_dir = os.path.join(RESULTS_DIR, f"{m_type}_{p_type}")
                    os.makedirs(res_dir, exist_ok=True)

                    # Check if all datasets have cached predictions
                    all_cached = True
                    for ds_name in datasets:
                        file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
                        if full_sequence:
                            file_prefix += "_full_seq"
                        raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")
                        if not os.path.exists(raw_file):
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
                        from models.ultralytics_wrapper import UltralyticsWrapper

                        wrapper = UltralyticsWrapper(m_type, model_path, device=DEVICE)
                    else:
                        from models.faster_rcnn_wrapper import FasterRCNNWrapper

                        wrapper = FasterRCNNWrapper(model_path, device=DEVICE)

                    for ds_name in datasets:
                        file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
                        if full_sequence:
                            file_prefix += "_full_seq"
                        raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")

                        if os.path.exists(raw_file):
                            print(f"Skipping dataset {ds_name}, predictions exist.")
                            continue

                        use_clahe = p_type == "clahe"

                        raw_results = generate_predictions(
                            wrapper,
                            ds_name,
                            use_clahe=use_clahe,
                            limit=limit,
                            batch_size=batch_size,
                            full_sequence=full_sequence,
                        )

                        with open(raw_file, "w") as f:
                            json.dump(raw_results, f, indent=2)

    print("\n=========================================")
    print(">>> All predictions generated. Running Evaluation Suite...")
    print("=========================================")
    run_evaluation_suite()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cycles", type=int, nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--full_sequence",
        action="store_true",
        help="Run over the entire full camera sequence instead of just ground-truth pool.",
    )
    args = parser.parse_args()

    run_all(
        limit=args.limit,
        batch_size=args.batch_size,
        target_models=args.models,
        target_cycles=args.cycles,
        target_variants=args.variants,
        full_sequence=args.full_sequence,
    )
