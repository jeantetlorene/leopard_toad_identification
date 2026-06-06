import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, PLOTS_DIR
from eval_utils.data_utils import load_predictions_from_json


def main():
    CYCLE = 0
    DATASET = "test_full_seq"

    # Define models explicitly with their specific variant and processing types
    models = [
        {"type": "yolo", "processing": "plain", "variant": "scratch"},
        {"type": "faster_rcnn", "processing": "plain", "variant": "scratch"},
        {"type": "rtdetr", "processing": "plain", "variant": "scratch"},
        {"type": "megadetector", "processing": "plain", "variant": "pretrained"},
    ]

    # Pre-load predictions once to avoid redundant IO
    loaded_datasets = {}
    for m_info in models:
        m_type = m_info["type"]
        processing = m_info["processing"]
        variant = m_info["variant"]

        model_folder = f"{m_type}_{processing}"
        folder_path = os.path.join(RESULTS_DIR, model_folder)

        filename = f"cycle_{CYCLE}_{variant}_{DATASET}_raw.json"
        filepath = os.path.join(folder_path, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        is_full_seq = "full_seq" in DATASET
        loaded_datasets[m_type] = load_predictions_from_json(
            filepath, is_full_seq=is_full_seq
        )

    # Map model_type to exact requested legend strings
    name_map = {
        "yolo": "YOLO",
        "faster_rcnn": "Faster R-CNN",
        "rtdetr": "RT-DETR",
        "megadetector": "MegaDetector",
    }

    # Generate Agnostic and Toad-Specific Plots
    for is_specific in [False, True]:
        plot_data = []

        for m_type in loaded_datasets:
            results = loaded_datasets[m_type]
            is_md = m_type == "megadetector"

            # Determine ground truth
            if is_specific:
                # positive if WLT (class 2) is present
                binary_gt = np.array(
                    [any(gt["cls"] == 2 for gt in res["gt_boxes"]) for res in results],
                    dtype=bool,
                )
            else:
                # positive if any animal is present
                binary_gt = np.array(
                    [len(res["gt_boxes"]) > 0 for res in results], dtype=bool
                )

            # Determine scores
            binary_scores = []
            for res in results:
                preds = res["predictions"]
                if is_specific and not is_md:
                    scores = [p["conf"] for p in preds if p["cls"] == 2]
                else:
                    scores = [p["conf"] for p in preds]
                binary_scores.append(max(scores + [0.0]))
            binary_scores = np.array(binary_scores)

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
            print(f"No plot data found for specific={is_specific}.")
            continue

        plt.figure(figsize=(10, 8))

        for d in plot_data:
            disp_name = name_map.get(d["model"], d["model"])
            label = f"{disp_name} - AUC: {d['auc']:.4f}"
            plt.plot(d["fpr"], d["tpr"], label=label, linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

        # Bound the plot frame from 0 to 1
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.gca().set_aspect("equal")

        suffix = "_wlt" if is_specific else ""
        png_path = os.path.join(PLOTS_DIR, f"binary_roc_baseline{suffix}.png")
        pdf_path = os.path.join(PLOTS_DIR, f"binary_roc_baseline{suffix}.pdf")

        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
