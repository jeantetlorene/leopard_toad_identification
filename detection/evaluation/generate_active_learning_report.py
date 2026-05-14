import pandas as pd
import os
import json
import numpy as np

from config import FILES_DIR, RESULTS_DIR
from metrics import calculate_map50_95

UNIFIED_CSV = os.path.join(FILES_DIR, "unified_model_evaluation.csv")
SWEEP_CSV = os.path.join(FILES_DIR, "per_class_threshold_sweep.csv")

MODELS = ["yolo", "faster_rcnn", "rtdetr"]
PROCESSINGS = ["clahe", "plain"]
VARIANTS = ["pretrained", "scratch"]
CYCLES = [0, 1, 2, 3, 4]

BUDGET_MAP = {0: 130, 1: 230, 2: 330, 3: 430, 4: 530}


def generate_report():
    if not os.path.exists(UNIFIED_CSV) or not os.path.exists(SWEEP_CSV):
        print("Error: Evaluation CSVs not found.")
        return

    df_unified = pd.read_csv(UNIFIED_CSV)

    # Filter only test set
    df_unified = df_unified[df_unified["dataset"] == "test"]

    results_table = []

    for m_type in MODELS:
        for proc in PROCESSINGS:
            for var in VARIANTS:
                # Check if this combination exists at all
                if df_unified[
                    (df_unified["model"] == m_type)
                    & (df_unified["processing"] == proc)
                    & (df_unified["variant"] == var)
                ].empty:
                    continue

                for cycle in CYCLES:
                    root_key = f"{m_type}_{proc}"

                    df_u = df_unified[
                        (df_unified["model"] == m_type)
                        & (df_unified["processing"] == proc)
                        & (df_unified["variant"] == var)
                        & (df_unified["cycle"] == cycle)
                    ]

                    if df_u.empty:
                        continue

                    map50 = df_u["mAP"].values[0]

                    # Macro average precision and recall at optimal threshold
                    # We can average across classes
                    precisions = []
                    recalls = []
                    for col in df_u.columns:
                        if col.endswith("_precision_optimal"):
                            precisions.append(df_u[col].values[0])
                        elif col.endswith("_recall_optimal"):
                            recalls.append(df_u[col].values[0])

                    precision = np.mean(precisions) if precisions else "N/A"
                    recall = np.mean(recalls) if recalls else "N/A"

                    # Calculate mAP50-95
                    map50_95 = "N/A"
                    raw_json_path = os.path.join(
                        RESULTS_DIR, root_key, f"cycle_{cycle}_{var}_test_raw.json"
                    )
                    if os.path.exists(raw_json_path):
                        with open(raw_json_path, "r") as f:
                            raw_results = json.load(f)
                        map50_95 = calculate_map50_95(raw_results)

                    results_table.append(
                        {
                            "Architecture": m_type.upper(),
                            "Processing": proc.capitalize(),
                            "Variant": var.capitalize(),
                            "Cycle": cycle,
                            "Cumulative Images": BUDGET_MAP.get(cycle, "Unknown"),
                            "mAP50": f"{map50:.4f}"
                            if isinstance(map50, (float, np.floating))
                            else map50,
                            "mAP50-95": f"{map50_95:.4f}"
                            if isinstance(map50_95, (float, np.floating))
                            else map50_95,
                            "Precision": f"{precision:.4f}"
                            if isinstance(precision, (float, np.floating))
                            else precision,
                            "Recall": f"{recall:.4f}"
                            if isinstance(recall, (float, np.floating))
                            else recall,
                        }
                    )

    # Write Markdown Report
    report_path = os.path.join(FILES_DIR, "active_learning_results.md")
    with open(report_path, "w") as f:
        f.write("# Results: Effect of Active Learning\n\n")
        f.write(
            "This report documents the cycle-by-cycle evolution of the detection models through the targeted active learning querying mechanism.\n\n"
        )

        f.write("### Active Learning Progression Table\n")
        f.write(
            "| Architecture | Processing | Variant | Cycle | Cumulative Images | mAP50 | mAP50-95 | Precision | Recall |\n"
        )
        f.write(
            "|--------------|------------|---------|-------|-------------------|-------|----------|-----------|--------|\n"
        )

        for row in results_table:
            f.write(
                f"| {row['Architecture']} | {row['Processing']} | {row['Variant']} | {row['Cycle']} | {row['Cumulative Images']} | {row['mAP50']} | {row['mAP50-95']} | {row['Precision']} | {row['Recall']} |\n"
            )

    print(f"Active learning report saved to {report_path}")


if __name__ == "__main__":
    generate_report()
