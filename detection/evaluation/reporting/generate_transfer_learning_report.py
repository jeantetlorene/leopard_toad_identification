import pandas as pd
import os
import json
import numpy as np

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, MODEL_ROOTS, RESULTS_DIR, DEVICE
from eval_utils.metrics import calculate_map50_95
from eval_utils.data_utils import refresh_results
from eval_utils.models.faster_rcnn_wrapper import FasterRCNNWrapper
from eval_utils.models.ultralytics_wrapper import UltralyticsWrapper
from reporting.generate_architecture_report import calculate_flops_params

UNIFIED_CSV = os.path.join(FILES_DIR, "active_learning_unified_evaluation.csv")

CYCLE = 0
PROCESSING = "clahe"
MODELS_TO_EVALUATE = ["yolo", "rtdetr", "faster_rcnn"]
VARIANTS = ["scratch", "pretrained"]


def generate_report():
    if not os.path.exists(UNIFIED_CSV):
        print("Error: Evaluation CSV not found. Please run evaluation_suite.py first.")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)

    results_table = []

    for m_type in MODELS_TO_EVALUATE:
        for variant in VARIANTS:
            root_key = f"{m_type}_{PROCESSING}"

            # Extract metrics from unified CSV (test set)
            df_u = df_unified[
                (df_unified["model"] == m_type)
                & (df_unified["variant"] == variant)
                & (df_unified["processing"] == PROCESSING)
                & (df_unified["cycle"] == CYCLE)
                & (df_unified["dataset"] == "test")
            ]
            map50 = df_u["mAP"].values[0] if not df_u.empty else "N/A"
            # Extract mAR from unified CSV
            mar = df_u["mAR"].values[0] if not df_u.empty and "mAR" in df_u else "N/A"

            # Calculate mAP50-95
            map50_95 = "N/A"
            raw_json_path = os.path.join(
                RESULTS_DIR, root_key, f"cycle_{CYCLE}_{variant}_test_raw.json"
            )
            if os.path.exists(raw_json_path):
                with open(raw_json_path, "r") as f:
                    raw_results = json.load(f)
                # Refresh ground truth from clean data
                raw_results = refresh_results(raw_results, is_full_seq=False)
                map50_95 = calculate_map50_95(raw_results)

            # Calculate Parameters
            params = "N/A"
            root_dir = MODEL_ROOTS.get(root_key)
            runs_dir = os.path.join(root_dir, "runs")
            run_name = (
                f"cycle_{CYCLE}_{variant}_scratch"
                if variant == "scratch"
                else f"cycle_{CYCLE}_{variant}_phase2"
            )
            model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

            if os.path.exists(model_path):
                try:
                    if m_type in ["yolo", "rtdetr"]:
                        wrapper = UltralyticsWrapper(m_type, model_path, device=DEVICE)
                    else:
                        wrapper = FasterRCNNWrapper(model_path, device=DEVICE)
                    params, _ = calculate_flops_params(wrapper, m_type)
                except Exception as e:
                    print(f"Error calculating params for {m_type} {variant}: {e}")

            results_table.append(
                {
                    "Architecture": m_type.upper(),
                    "Variant": variant.capitalize(),
                    "mAP50": f"{map50:.4f}"
                    if isinstance(map50, (float, np.floating))
                    else map50,
                    "mAP50-95": f"{map50_95:.4f}"
                    if isinstance(map50_95, (float, np.floating))
                    else map50_95,
                    "mAR": f"{mar:.4f}"
                    if isinstance(mar, (float, np.floating))
                    else mar,
                    "Trainable Parameters (M)": params,
                }
            )

    # Write Markdown Report
    os.makedirs(FILES_DIR, exist_ok=True)
    report_path = os.path.join(FILES_DIR, "transfer_learning_results.md")
    with open(report_path, "w") as f:
        f.write("# Results: Effect of Transfer Learning (Cycle 0)\n\n")
        f.write(
            "This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.\n\n"
        )

        f.write("### Comprehensive Transfer Learning Performance Table\n")
        f.write(
            "| Architecture | Variant | mAP50 | mAP50-95 | mean Average Recall | Trainable Parameters (M) |\n"
        )
        f.write(
            "|--------------|---------|-------|----------|---------------------|--------------------------|\n"
        )

        for row in results_table:
            f.write(
                f"| {row['Architecture']} | {row['Variant']} | {row['mAP50']} | {row['mAP50-95']} | {row['mAR']} | {row['Trainable Parameters (M)']} |\n"
            )

    print(f"Transfer learning report saved to {report_path}")


if __name__ == "__main__":
    generate_report()
