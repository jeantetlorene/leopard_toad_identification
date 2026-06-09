import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import RESULTS_DIR, PLOTS_DIR
from eval_utils.data_utils import load_predictions_from_json
from eval_utils.metrics import calculate_detection_metrics

plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Outfit", "DejaVu Sans", "Arial"]

MODEL_COLORS = {"yolo": "#1f77b4", "faster_rcnn": "#ff7f0e", "rtdetr": "#2ca02c"}

MODEL_NAMES = {"yolo": "YOLO", "faster_rcnn": "Faster R-CNN", "rtdetr": "RT-DETR"}


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


def plot_pr_curves(json_paths, save_path_png, save_path_pdf):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    has_data = False

    for m_type in ["yolo", "faster_rcnn", "rtdetr"]:
        json_path = json_paths.get(m_type)
        if not json_path or not os.path.exists(json_path):
            continue

        mrec, mpre, ap, ar = get_wlt_pr(json_path)
        if mrec is None:
            continue

        has_data = True
        label = f"{MODEL_NAMES[m_type]} (AP: {ap:.2f}, AR: {ar:.2f})"

        ax.plot(
            mrec,
            mpre,
            label=label,
            color=MODEL_COLORS.get(m_type, "black"),
            linewidth=3,
        )

    if not has_data:
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
    CYCLES = [0, 1, 2, 3, 4]

    CONFIGS = [
        {"process": "plain", "variant": "scratch", "name": "baseline"},
        {"process": "clahe", "variant": "scratch", "name": "clahe_scratch"},
        {"process": "plain", "variant": "pretrained", "name": "pretrained"},
        {"process": "clahe", "variant": "pretrained", "name": "clahe_pretrained"},
    ]

    out_base = os.path.join(PLOTS_DIR, "wlt_pr_curves")
    os.makedirs(out_base, exist_ok=True)

    generated = 0

    # Iterate with a simple counter/tracker for the user since JSON eval takes longer
    total_configs = len(CONFIGS) * len(CYCLES)
    with tqdm(total=total_configs, desc="Generating PR Curves") as pbar:
        for config in CONFIGS:
            process = config["process"]
            variant = config["variant"]
            conf_name = config["name"]

            config_dir = os.path.join(out_base, conf_name)
            os.makedirs(config_dir, exist_ok=True)

            for cycle in CYCLES:
                json_paths = {}
                for m_type in ["yolo", "faster_rcnn", "rtdetr"]:
                    model_folder = m_type if process == "plain" else f"{m_type}_clahe"
                    fname = f"cycle_{cycle}_{variant}_test_raw.json"
                    json_path = os.path.join(RESULTS_DIR, model_folder, fname)
                    json_paths[m_type] = json_path

                base_filename = f"pr_curve_cycle{cycle}_{conf_name}"
                png_path = os.path.join(config_dir, base_filename + ".png")
                pdf_path = os.path.join(config_dir, base_filename + ".pdf")

                plot_pr_curves(json_paths, png_path, pdf_path)
                generated += 1
                pbar.update(1)

    print(f"Generated {generated} PR curves in {out_base}")


if __name__ == "__main__":
    main()
