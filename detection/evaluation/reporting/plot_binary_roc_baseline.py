import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, PLOTS_DIR
from eval_utils.data_utils import refresh_results


def main():
    # Baseline configuration as defined in the architecture report
    CYCLE = 0
    VARIANT = "scratch"
    PROCESSING = "clahe"
    DATASET = "test_full_seq"

    plot_data = []
    models = ["yolo", "faster_rcnn", "rtdetr"]

    for m_type in models:
        model_folder = f"{m_type}_{PROCESSING}"
        folder_path = os.path.join(RESULTS_DIR, model_folder)

        filename = f"cycle_{CYCLE}_{VARIANT}_{DATASET}_raw.json"
        filepath = os.path.join(folder_path, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        print(f"Loading predictions for {m_type}...")
        with open(filepath, "r") as f:
            results = json.load(f)

        # Refresh ground truth from clean data
        is_full_seq = "full_seq" in DATASET
        results = refresh_results(results, is_full_seq=is_full_seq)

        binary_gt = np.array([res["is_positive"] for res in results])
        binary_scores = np.array(
            [max([p["conf"] for p in res["predictions"]] + [0.0]) for res in results]
        )

        if len(np.unique(binary_gt)) > 1:
            binary_fpr, binary_tpr, _ = roc_curve(binary_gt, binary_scores)
            binary_auc = auc(binary_fpr, binary_tpr)
        else:
            binary_fpr, binary_tpr, binary_auc = None, None, np.nan

        if binary_fpr is not None:
            plot_data.append(
                {
                    "model": m_type,
                    "fpr": binary_fpr,
                    "tpr": binary_tpr,
                    "auc": binary_auc,
                }
            )

    if not plot_data:
        print("No plot data found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 8))

    # Map model_type to exact requested legend strings
    name_map = {"yolo": "YOLO", "faster_rcnn": "Faster R-CNN", "rtdetr": "RT-DETR"}

    for d in plot_data:
        disp_name = name_map.get(d["model"], d["model"])
        label = f"{disp_name} - AUC: {d['auc']:.4f}"
        plt.plot(d["fpr"], d["tpr"], label=label, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

    # Bound the plot frame from 0 to 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Image-Level ROC Curves - Baseline Architectures (Test Unlabeled Pool)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.gca().set_aspect("equal")

    png_path = os.path.join(PLOTS_DIR, "binary_roc_baseline.png")
    pdf_path = os.path.join(PLOTS_DIR, "binary_roc_baseline.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
