import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, PLOTS_DIR
from eval_utils.data_utils import load_predictions_from_json
from eval_utils.metrics import calculate_detection_metrics


def get_exact_macro_pr(raw_json_path):
    if not os.path.exists(raw_json_path):
        return None, None, None, None

    results = load_predictions_from_json(raw_json_path, is_full_seq=False)

    metrics = calculate_detection_metrics(results, iou_threshold=0.5)
    class_curves = metrics.get("class_curves", {})
    if not class_curves:
        return None, None, None, None

    recalls = np.linspace(0, 1.0, 101)
    precisions = []

    for c, curve in class_curves.items():
        mrec = curve["recall"]
        mpre = curve["precision"]

        # numpy interp requires increasing x-values.
        # VOC mrec is [0.0, ..., 1.0] and monotonically increasing
        p_interp = np.interp(recalls, mrec, mpre)
        precisions.append(p_interp)

    if not precisions:
        return None, None, None, None

    macro_precisions = np.mean(precisions, axis=0)

    max_recalls = []
    for c, curve in class_curves.items():
        mrec = curve["recall"]
        if len(mrec) >= 2:
            max_recalls.append(mrec[-2])  # last point before [1.0] endpoint
        else:
            max_recalls.append(0.0)
    mar = np.mean(max_recalls) if max_recalls else 0.0

    mAP = metrics["mAP"]

    return recalls, macro_precisions, mAP, mar


def plot_pr_curves():
    architectures = ["yolo", "faster_rcnn", "rtdetr"]

    for arch in architectures:
        fig, ax = plt.subplots(figsize=(8, 8))

        has_data = False
        for proc in ["plain", "clahe"]:
            root_key = f"{arch}_{proc}"
            json_path = os.path.join(
                RESULTS_DIR, root_key, "cycle_0_pretrained_test_raw.json"
            )

            rec, prec, mAP, mar = get_exact_macro_pr(json_path)
            if rec is None:
                print(f"Skipping {root_key} - no data")
                continue

            has_data = True
            label_name = "CLAHE" if proc == "clahe" else proc.capitalize()
            label = f"{label_name} (mAP: {mAP:.3f}, mAR: {mar:.3f})"
            color = "#1f77b4" if proc == "plain" else "#ff7f0e"

            ax.plot(rec, prec, label=label, color=color, linewidth=3)

        if not has_data:
            plt.close()
            continue

        # ax.set_title(
        #     f"Precision-Recall Curve: {arch.upper()} (Cycle 0)",
        #     fontsize=16,
        #     fontweight="bold",
        # )
        ax.set_xlabel("Recall", fontsize=18)
        ax.set_ylabel("Precision", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(fontsize=16, loc="lower left")
        ax.grid(False)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_aspect("equal")

        ax.text(
            0.95,
            0.05,
            "All Classes (Macro)",
            fontsize=12,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.5,
        )

        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)

        png_path = os.path.join(PLOTS_DIR, f"pr_curve_{arch}_cycle0.png")
        pdf_path = os.path.join(PLOTS_DIR, f"pr_curve_{arch}_cycle0.pdf")

        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, dpi=300)
        print(f"Saved exact PR curve plot to: {png_path} and {pdf_path}")
        plt.close()


if __name__ == "__main__":
    plot_pr_curves()
