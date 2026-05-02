import os
import pandas as pd
import json
import argparse
from config import MODEL_ROOTS, RESULTS_DIR, DEVICE, DATASETS
from models.ultralytics_wrapper import UltralyticsWrapper
from models.faster_rcnn_wrapper import FasterRCNNWrapper
from evaluate_single import evaluate_single_model


def run_all(
    limit=None,
    batch_size=32,
    target_models=None,
    target_cycles=None,
    target_variants=None,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_types = ["yolo", "rtdetr", "faster_rcnn"]
    processing_types = ["plain", "clahe"]
    variants = ["pretrained", "scratch"]
    cycles = range(5)

    summary_rows = []

    for m_type in model_types:
        if target_models and m_type not in target_models:
            continue

        for p_type in processing_types:
            # Determine root directory based on model type and processing
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

                    # Construct model path
                    # Note: We prefer phase2 for pretrained
                    run_name = (
                        f"cycle_{cycle}_{var}_phase2"
                        if var == "pretrained"
                        else f"cycle_{cycle}_{var}_scratch"
                    )
                    model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

                    if not os.path.exists(model_path):
                        # Try phase1 as fallback for pretrained if phase2 is missing
                        if var == "pretrained":
                            run_name_p1 = f"cycle_{cycle}_{var}_phase1"
                            model_path = os.path.join(
                                runs_dir, run_name_p1, "weights", "best.pt"
                            )
                            if not os.path.exists(model_path):
                                continue
                        else:
                            continue

                    print(
                        f"\n>>> Evaluating {m_type} | {p_type} | Cycle {cycle} | {var}"
                    )

                    # Load model wrapper
                    if m_type in ["yolo", "rtdetr"]:
                        wrapper = UltralyticsWrapper(m_type, model_path, device=DEVICE)
                    else:
                        wrapper = FasterRCNNWrapper(model_path, device=DEVICE)

                    # Evaluate on both datasets (test and val)
                    for ds_name in DATASETS.keys():
                        res_dir = os.path.join(RESULTS_DIR, f"{m_type}_{p_type}")
                        os.makedirs(res_dir, exist_ok=True)

                        file_prefix = f"cycle_{cycle}_{var}_{ds_name}"
                        raw_file = os.path.join(res_dir, f"{file_prefix}_raw.json")
                        metrics_file = os.path.join(
                            res_dir, f"{file_prefix}_metrics.csv"
                        )

                        # Use CLAHE preprocessing if it's a CLAHE model
                        use_clahe = p_type == "clahe"

                        raw_results, metrics = evaluate_single_model(
                            wrapper,
                            ds_name,
                            use_clahe=use_clahe,
                            limit=limit,
                            batch_size=batch_size,
                        )

                        # Save results
                        with open(raw_file, "w") as f:
                            json.dump(raw_results, f)
                        pd.DataFrame(metrics).to_csv(metrics_file, index=False)

                        # Summary entry (at 0.1 threshold)
                        metrics_df = pd.DataFrame(metrics)
                        idx = (metrics_df["threshold"] - 0.1).abs().idxmin()
                        summary_rows.append(
                            {
                                "model": m_type,
                                "processing": p_type,
                                "cycle": cycle,
                                "variant": var,
                                "dataset": ds_name,
                                "recall_0.1": metrics_df.loc[idx, "recall"],
                                "specificity_0.1": metrics_df.loc[idx, "specificity"],
                            }
                        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, "all_models_summary.csv"), index=False
        )
        print("\nFull Evaluation Complete! Summary saved to all_models_summary.csv")
        print(summary_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cycles", type=int, nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    args = parser.parse_args()

    run_all(
        limit=args.limit,
        batch_size=args.batch_size,
        target_models=args.models,
        target_cycles=args.cycles,
        target_variants=args.variants,
    )
