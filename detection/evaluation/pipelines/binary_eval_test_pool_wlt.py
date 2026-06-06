import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, FILES_DIR, PLOTS_DIR, CONF_THRESHOLDS
from eval_utils.metrics import calculate_image_level_metrics
from eval_utils.data_utils import load_predictions_from_json


def compute_wlt_sweep(results, thresholds):
    """
    Compute binary classification metrics focusing strictly on Class 2 (Western Leopard Toad).
    """
    # 1. Determine ground truth focusing strictly on Class 2 (WLT)
    is_positive = np.array(
        [any(gt["cls"] == 2 for gt in res["gt_boxes"]) for res in results],
        dtype=bool,
    )

    # 2. Determine maximum confidence score for Class 2 predictions
    max_confs = []
    for res in results:
        preds = res["predictions"]
        scores = [p["conf"] for p in preds if p["cls"] == 2]
        max_confs.append(max(scores + [0.0]))
    max_confs = np.array(max_confs)

    # 3. Sweep thresholds
    metrics = []
    total_images = len(results)

    for thresh in thresholds:
        has_detection = max_confs >= thresh
        tp = np.sum(has_detection & is_positive)
        fn = np.sum(~has_detection & is_positive)
        fp = np.sum(has_detection & ~is_positive)
        tn = np.sum(~has_detection & ~is_positive)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        labor_reduction = (tn + fn) / total_images if total_images > 0 else 0.0

        metrics.append(
            {
                "threshold": thresh,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "recall": float(recall),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1_score": float(f1),
                "labor_reduction": float(labor_reduction),
            }
        )

    return metrics, is_positive, max_confs


def main():
    all_binary_sweep = []
    plot_data = []

    # Find all model results folders
    folders = sorted(
        [
            f
            for f in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, f))
        ]
    )

    for model_folder in folders:
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        # ONLY look for test_full_seq raw predictions (the test unlabeled pool)
        filenames = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if f.endswith("_test_full_seq_raw.json")
            ]
        )

        for filename in filenames:
            parts = filename.replace("_raw.json", "").split("_")
            cycle = int(parts[1])
            variant = parts[2]
            dataset = "test_full_seq"

            mf_parts = model_folder.split("_")
            processing = mf_parts[-1]
            model_type = "_".join(mf_parts[:-1])

            print(
                f"Processing {model_type} ({processing}) | Variant: {variant} | Cycle: {cycle}"
            )
            filepath = os.path.join(folder_path, filename)
            results = load_predictions_from_json(filepath, is_full_seq=True)

            if not results:
                print(f"Warning: No images from {filename} found. Skipping.")
                continue

            # Calculate Sweep strictly focusing on WLT
            binary_sweep_data, binary_gt, binary_scores = compute_wlt_sweep(
                results, CONF_THRESHOLDS
            )

            # Calculate ROC-AUC for WLT
            if len(np.unique(binary_gt)) > 1:
                binary_fpr, binary_tpr, _ = roc_curve(binary_gt, binary_scores)
                binary_auc = auc(binary_fpr, binary_tpr)
            else:
                binary_fpr, binary_tpr = None, None
                binary_auc = np.nan

            for entry in binary_sweep_data:
                entry.update(
                    {
                        "model": model_type,
                        "processing": processing,
                        "cycle": cycle,
                        "variant": variant,
                        "dataset": dataset,
                        "auc": binary_auc,
                    }
                )
                all_binary_sweep.append(entry)

            if binary_fpr is not None:
                plot_data.append(
                    {
                        "model": model_type,
                        "processing": processing,
                        "variant": variant,
                        "cycle": cycle,
                        "fpr": binary_fpr,
                        "tpr": binary_tpr,
                        "auc": binary_auc,
                    }
                )

    # Save CSV
    os.makedirs(FILES_DIR, exist_ok=True)
    df_sweep = pd.DataFrame(all_binary_sweep)
    if not df_sweep.empty:
        sweep_path = os.path.join(FILES_DIR, "wlt_binary_threshold_sweep_test_pool.csv")
        df_sweep.to_csv(sweep_path, index=False)
        print(f"\nSaved CSV to {sweep_path}")
    else:
        print("\nNo data to save for threshold sweep.")

    # Generate Plots
    if not plot_data:
        print("No plot data found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    cycles = sorted(list(set(d["cycle"] for d in plot_data)))

    name_map = {
        "yolo": "YOLO",
        "faster_rcnn": "Faster R-CNN",
        "rtdetr": "RT-DETR",
    }

    for c in cycles:
        plt.figure(figsize=(10, 8))
        subset = [d for d in plot_data if d["cycle"] == c]
        for d in subset:
            disp_model = name_map.get(d["model"].lower(), d["model"])
            disp_processing = d["processing"].upper()
            disp_variant = d["variant"].capitalize()
            label = (
                f"{disp_model} {disp_variant} ({disp_processing}) - AUC: {d['auc']:.4f}"
            )
            plt.plot(d["fpr"], d["tpr"], label=label, linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Image-Level ROC Curves (WLT Focus) - Test Pool (Cycle {c})")
        plt.legend(loc="lower right")
        plt.gca().set_aspect("equal")

        plot_path_png = os.path.join(
            PLOTS_DIR, f"wlt_binary_roc_test_pool_cycle_{c}.png"
        )
        plot_path_pdf = os.path.join(
            PLOTS_DIR, f"wlt_binary_roc_test_pool_cycle_{c}.pdf"
        )
        plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved WLT ROC plot to {plot_path_png} and {plot_path_pdf}")


if __name__ == "__main__":
    main()
