import pandas as pd
import os
import json
import numpy as np

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, RESULTS_DIR, BASE_DIR
from eval_utils.metrics import calculate_map50_95
from eval_utils.data_utils import refresh_results

UNIFIED_CSV = os.path.join(FILES_DIR, "unified_model_evaluation.csv")

MODELS = ["yolo", "faster_rcnn", "rtdetr"]
PROCESSINGS = ["clahe", "plain"]
VARIANTS = ["pretrained", "scratch"]
CYCLES = [0, 1, 2, 3, 4]

BUDGET_MAP = {0: 130, 1: 230, 2: 330, 3: 430, 4: 530}

CLASS_MAP = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}


def count_training_instances(m_type, var, cycle):
    # Determine the directory
    labels_dir = os.path.join(
        BASE_DIR,
        "active learning",
        "data",
        m_type,
        var,
        f"cycle_{cycle}",
        "train",
        "labels",
    )
    counts = {0: 0, 1: 0, 2: 0}
    if not os.path.exists(labels_dir):
        return counts

    for file in os.listdir(labels_dir):
        if not file.endswith(".txt") or file == "classes.txt":
            continue
        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in counts:
                    counts[cls_id] += 1
    return counts


def generate_report():
    if not os.path.exists(UNIFIED_CSV):
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

                    # Calculate mAP50-95
                    map50_95 = "N/A"
                    raw_json_path = os.path.join(
                        RESULTS_DIR, root_key, f"cycle_{cycle}_{var}_test_raw.json"
                    )
                    if os.path.exists(raw_json_path):
                        with open(raw_json_path, "r") as f:
                            raw_results = json.load(f)
                        # Refresh ground truth from clean data
                        raw_results = refresh_results(raw_results, is_full_seq=False)
                        map50_95 = calculate_map50_95(raw_results)

                    # Get training instance counts
                    train_counts = count_training_instances(m_type, var, cycle)

                    # Get class metrics
                    cls_metrics = {}
                    for cls_id, cls_name in CLASS_MAP.items():
                        ap_col = f"{cls_name}_ap"
                        prec_col = f"{cls_name}_precision_optimal"
                        rec_col = f"{cls_name}_recall_optimal"
                        spec_col = f"{cls_name}_specificity_optimal"

                        cls_metrics[cls_name] = {
                            "inst": train_counts[cls_id],
                            "ap": df_u[ap_col].values[0] if ap_col in df_u else "N/A",
                            "prec": df_u[prec_col].values[0]
                            if prec_col in df_u
                            else "N/A",
                            "rec": df_u[rec_col].values[0]
                            if rec_col in df_u
                            else "N/A",
                            "spec": df_u[spec_col].values[0]
                            if spec_col in df_u
                            else "N/A",
                        }

                    results_table.append(
                        {
                            "Architecture": m_type.upper(),
                            "Processing": proc.capitalize(),
                            "Variant": var.capitalize(),
                            "Cycle": cycle,
                            "Cumul Imgs": BUDGET_MAP.get(cycle, "Unknown"),
                            "mAP50": f"{map50:.4f}"
                            if isinstance(map50, (float, np.floating))
                            else map50,
                            "mAP50-95": f"{map50_95:.4f}"
                            if isinstance(map50_95, (float, np.floating))
                            else map50_95,
                            "WLT Inst": cls_metrics["Western_Leopard_Toad"]["inst"],
                            "WLT AP50": f"{cls_metrics['Western_Leopard_Toad']['ap']:.4f}"
                            if isinstance(
                                cls_metrics["Western_Leopard_Toad"]["ap"], float
                            )
                            else cls_metrics["Western_Leopard_Toad"]["ap"],
                            "WLT Prec": f"{cls_metrics['Western_Leopard_Toad']['prec']:.4f}"
                            if isinstance(
                                cls_metrics["Western_Leopard_Toad"]["prec"], float
                            )
                            else cls_metrics["Western_Leopard_Toad"]["prec"],
                            "WLT Rec": f"{cls_metrics['Western_Leopard_Toad']['rec']:.4f}"
                            if isinstance(
                                cls_metrics["Western_Leopard_Toad"]["rec"], float
                            )
                            else cls_metrics["Western_Leopard_Toad"]["rec"],
                            "WLT Spec": f"{cls_metrics['Western_Leopard_Toad']['spec']:.4f}"
                            if isinstance(
                                cls_metrics["Western_Leopard_Toad"]["spec"], float
                            )
                            else cls_metrics["Western_Leopard_Toad"]["spec"],
                            "SM Inst": cls_metrics["Small_Mammal"]["inst"],
                            "SM AP50": f"{cls_metrics['Small_Mammal']['ap']:.4f}"
                            if isinstance(cls_metrics["Small_Mammal"]["ap"], float)
                            else cls_metrics["Small_Mammal"]["ap"],
                            "SM Prec": f"{cls_metrics['Small_Mammal']['prec']:.4f}"
                            if isinstance(cls_metrics["Small_Mammal"]["prec"], float)
                            else cls_metrics["Small_Mammal"]["prec"],
                            "SM Rec": f"{cls_metrics['Small_Mammal']['rec']:.4f}"
                            if isinstance(cls_metrics["Small_Mammal"]["rec"], float)
                            else cls_metrics["Small_Mammal"]["rec"],
                            "SM Spec": f"{cls_metrics['Small_Mammal']['spec']:.4f}"
                            if isinstance(cls_metrics["Small_Mammal"]["spec"], float)
                            else cls_metrics["Small_Mammal"]["spec"],
                            "OA Inst": cls_metrics["Other_Amphibian"]["inst"],
                            "OA AP50": f"{cls_metrics['Other_Amphibian']['ap']:.4f}"
                            if isinstance(cls_metrics["Other_Amphibian"]["ap"], float)
                            else cls_metrics["Other_Amphibian"]["ap"],
                            "OA Prec": f"{cls_metrics['Other_Amphibian']['prec']:.4f}"
                            if isinstance(cls_metrics["Other_Amphibian"]["prec"], float)
                            else cls_metrics["Other_Amphibian"]["prec"],
                            "OA Rec": f"{cls_metrics['Other_Amphibian']['rec']:.4f}"
                            if isinstance(cls_metrics["Other_Amphibian"]["rec"], float)
                            else cls_metrics["Other_Amphibian"]["rec"],
                            "OA Spec": f"{cls_metrics['Other_Amphibian']['spec']:.4f}"
                            if isinstance(cls_metrics["Other_Amphibian"]["spec"], float)
                            else cls_metrics["Other_Amphibian"]["spec"],
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
            "| Arch | Proc | Variant | Cycle | Budget | mAP50 | mAP50-95 | WLT Inst | WLT AP50 | WLT Prec | WLT Rec | WLT Spec | SM Inst | SM AP50 | SM Prec | SM Rec | SM Spec | OA Inst | OA AP50 | OA Prec | OA Rec | OA Spec |\n"
        )
        f.write(
            "|------|------|---------|-------|--------|-------|----------|----------|----------|----------|---------|----------|---------|---------|---------|--------|---------|---------|---------|---------|--------|---------|\n"
        )

        for row in results_table:
            f.write(
                f"| {row['Architecture']} | {row['Processing']} | {row['Variant']} | {row['Cycle']} | {row['Cumul Imgs']} | {row['mAP50']} | {row['mAP50-95']} | "
                f"{row['WLT Inst']} | {row['WLT AP50']} | {row['WLT Prec']} | {row['WLT Rec']} | {row['WLT Spec']} | "
                f"{row['SM Inst']} | {row['SM AP50']} | {row['SM Prec']} | {row['SM Rec']} | {row['SM Spec']} | "
                f"{row['OA Inst']} | {row['OA AP50']} | {row['OA Prec']} | {row['OA Rec']} | {row['OA Spec']} |\n"
            )

    print(f"Active learning report saved to {report_path}")


if __name__ == "__main__":
    generate_report()
