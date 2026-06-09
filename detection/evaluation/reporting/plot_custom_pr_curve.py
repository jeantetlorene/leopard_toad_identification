import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import RESULTS_DIR, PLOTS_DIR
from eval_utils.data_utils import load_predictions_from_json
from eval_utils.metrics import calculate_detection_metrics

plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Outfit", "DejaVu Sans", "Arial"]

MODEL_COLORS = {"yolo": "#1f77b4", "faster_rcnn": "#ff7f0e", "rtdetr": "#2ca02c"}


def get_wlt_pr(raw_json_path):
    if not os.path.exists(raw_json_path):
        return None, None, None, None

    results = load_predictions_from_json(raw_json_path, is_full_seq=False)
    metrics = calculate_detection_metrics(results, iou_threshold=0.5)
    class_curves = metrics.get("class_curves", {})

    if 2 not in class_curves:
        return None, None, None, None

    curve = class_curves[2]
    mrec = curve["recall"]
    mpre = curve["precision"]

    ap = metrics.get("class_aps", {}).get(2, 0.0)
    ar = mrec[-2] if len(mrec) >= 2 else 0.0

    return mrec, mpre, ap, ar


def plot_custom_curve(json_paths, custom_labels, save_path_png, save_path_pdf):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    has_data = False

    for m_type in ["yolo", "faster_rcnn", "rtdetr"]:
        json_path = json_paths.get(m_type)
        if not json_path or not os.path.exists(json_path):
            print(f"Skipping {m_type}: Could not find {json_path}")
            continue

        mrec, mpre, ap, ar = get_wlt_pr(json_path)
        if mrec is None:
            print(f"Skipping {m_type}: No WLT class curve returned.")
            continue

        has_data = True
        label = f"{custom_labels[m_type]} (AP: {ap:.2f}, AR: {ar:.2f})"

        ax.plot(
            mrec,
            mpre,
            label=label,
            color=MODEL_COLORS.get(m_type, "black"),
            linewidth=3,
        )

    if not has_data:
        print("No valid data to plot. Closing.")
        plt.close()
        return

    ax.set_xlabel("Recall", fontsize=18)
    ax.set_ylabel("Precision", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=16, loc="lower left")
    ax.grid(False)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.05])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    fig.tight_layout()
    plt.savefig(save_path_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_path_pdf, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    out_base = os.path.join(PLOTS_DIR, "wlt_pr_curves")
    os.makedirs(out_base, exist_ok=True)

    # --- PLOT 1 ---
    # YOLO Cycle 2 (idx 1), Faster RCNN Cycle 3 (idx 2), RTDETR Cycle 5 (idx 4)
    json_paths_1 = {
        "yolo": os.path.join(
            RESULTS_DIR, "yolo_clahe", "cycle_1_pretrained_test_raw.json"
        ),
        "faster_rcnn": os.path.join(
            RESULTS_DIR, "faster_rcnn_clahe", "cycle_2_pretrained_test_raw.json"
        ),
        "rtdetr": os.path.join(
            RESULTS_DIR, "rtdetr_clahe", "cycle_4_pretrained_test_raw.json"
        ),
    }
    custom_labels_1 = {
        "yolo": "YOLO (Cycle 2)",
        "faster_rcnn": "Faster R-CNN (Cycle 3)",
        "rtdetr": "RT-DETR (Cycle 5)",
    }
    png_path_1 = os.path.join(out_base, "custom_clahe_pretrained_comparison_1.png")
    pdf_path_1 = os.path.join(out_base, "custom_clahe_pretrained_comparison_1.pdf")

    print("Generating custom mixed-cycle PR Curve #1...")
    plot_custom_curve(json_paths_1, custom_labels_1, png_path_1, pdf_path_1)
    print(f"Generated successfully: {png_path_1}")

    # --- PLOT 2 ---
    # YOLO Cycle 1 (idx 0), Faster RCNN Cycle 4 (idx 3), RTDETR Cycle 3 (idx 2)
    json_paths_2 = {
        "yolo": os.path.join(
            RESULTS_DIR, "yolo_clahe", "cycle_0_pretrained_test_raw.json"
        ),
        "faster_rcnn": os.path.join(
            RESULTS_DIR, "faster_rcnn_clahe", "cycle_3_pretrained_test_raw.json"
        ),
        "rtdetr": os.path.join(
            RESULTS_DIR, "rtdetr_clahe", "cycle_2_pretrained_test_raw.json"
        ),
    }
    custom_labels_2 = {
        "yolo": "YOLO (Cycle 1)",
        "faster_rcnn": "Faster R-CNN (Cycle 4)",
        "rtdetr": "RT-DETR (Cycle 3)",
    }
    png_path_2 = os.path.join(out_base, "custom_clahe_pretrained_comparison_2.png")
    pdf_path_2 = os.path.join(out_base, "custom_clahe_pretrained_comparison_2.pdf")

    print("\nGenerating custom mixed-cycle PR Curve #2...")
    plot_custom_curve(json_paths_2, custom_labels_2, png_path_2, pdf_path_2)
    print(f"Generated successfully: {png_path_2}")


if __name__ == "__main__":
    main()
