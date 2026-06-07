import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, PLOTS_DIR


def count_bounding_boxes(model, variant, cycle):
    """
    Counts the total number of bounding boxes (instances) across all classes
    in the active learning training dataset for a given configuration.
    """
    base_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection"
    labels_dir = os.path.join(
        base_dir,
        "active learning",
        "data",
        model,
        variant,
        f"cycle_{cycle}",
        "train",
        "labels",
    )
    if not os.path.exists(labels_dir):
        return 0

    total_boxes = 0
    try:
        for f in os.listdir(labels_dir):
            if f.endswith(".txt"):
                filepath = os.path.join(labels_dir, f)
                with open(filepath, "r") as file:
                    total_boxes += sum(1 for line in file if line.strip())
    except Exception as e:
        print(f"Error reading labels for {model} {variant} cycle {cycle}: {e}")
    return total_boxes


def main():
    print("Generating WLT Image-Level Active Learning Trajectory Plots...")

    # Paths
    sweep_csv = os.path.join(FILES_DIR, "wlt_binary_threshold_sweep_test_pool.csv")

    if not os.path.exists(sweep_csv):
        print(
            f"Error: Sweep CSV not found at {sweep_csv}. Please run the evaluation pipeline first."
        )
        return

    df = pd.read_csv(sweep_csv)
    if df.empty:
        print("Error: Sweep CSV is empty.")
        return

    CYCLES = [0, 1, 2, 3, 4]
    MODELS = ["yolo", "faster_rcnn", "rtdetr"]

    # The 4 requested configurations
    CONFIGS = [
        {"process": "plain", "variant": "scratch", "name": "Baseline (Plain+Scratch)"},
        {"process": "clahe", "variant": "scratch", "name": "CLAHE+Scratch"},
        {
            "process": "plain",
            "variant": "pretrained",
            "name": "Baseline+Pretrained (Plain)",
        },
        {"process": "clahe", "variant": "pretrained", "name": "CLAHE+Pretrained"},
    ]

    COLOR_FP = "#d62728"  # Red for False Positives
    COLOR_BBOX = "#2ca02c"  # Green for Bounding Boxes

    METRICS = [
        {"id": "max_f1", "name": "max_f1"},
        {"id": "max_f2", "name": "max_f2"},
        {"id": "min_roc", "name": "min_roc_dist"},
        {"id": "recall_95", "name": "recall_95_target"},
    ]

    for m_type in MODELS:
        for config in CONFIGS:
            process = config["process"]
            variant = config["variant"]
            conf_name = config["name"]

            bbox_history = []
            for cycle in CYCLES:
                bbox_history.append(count_bounding_boxes(m_type, variant, cycle))

            for metric in METRICS:
                fp_history = []
                has_data = True

                for cycle in CYCLES:
                    # Query the sweep data for this cycle, model, process, variant
                    df_cycle = df[
                        (df["model"] == m_type)
                        & (df["processing"] == process)
                        & (df["variant"] == variant)
                        & (df["cycle"] == cycle)
                        & (df["dataset"] == "test_full_seq")
                    ]

                    if df_cycle.empty:
                        print(
                            f"Warning: No data for {m_type} {process} {variant} cycle {cycle}. Skipping trajectory plot."
                        )
                        has_data = False
                        break

                    if metric["id"] == "max_f1":
                        best_row = df_cycle.loc[df_cycle["f1_score"].idxmax()]
                    elif metric["id"] == "max_f2":
                        if "f2_score" in df_cycle.columns:
                            best_row = df_cycle.loc[df_cycle["f2_score"].idxmax()]
                        else:
                            best_row = df_cycle.loc[df_cycle["f1_score"].idxmax()]
                    elif metric["id"] == "min_roc":
                        best_row = df_cycle.loc[df_cycle["roc_distance"].idxmin()]
                    elif metric["id"] == "recall_95":
                        df_95 = df_cycle[df_cycle["recall"] >= 0.95]
                        if not df_95.empty:
                            best_row = df_95.loc[df_95["threshold"].idxmax()]
                        else:
                            best_row = df_cycle.loc[df_cycle["recall"].idxmax()]

                    fp_history.append(best_row["fp"])

                if not has_data:
                    continue

                # PlotTrajectory Dual Axis
                fig, ax_fp = plt.subplots(figsize=(7, 5))
                cycles_plot = [c + 1 for c in CYCLES]

                # Primary Axis: False Positives
                (line_fp,) = ax_fp.plot(
                    cycles_plot,
                    fp_history,
                    marker="o",
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    linewidth=2.5,
                    color=COLOR_FP,
                    label="False Positives",
                )

                ax_fp.set_xlabel(
                    "Active Learning Cycle",
                    fontsize=11,
                    fontweight="medium",
                    labelpad=8,
                )
                ax_fp.set_ylabel(
                    "False Positives",
                    fontsize=11,
                    fontweight="medium",
                    color=COLOR_FP,
                    labelpad=8,
                )
                ax_fp.tick_params(axis="y", colors=COLOR_FP)

                # Set Y limit with some headroom
                max_fp_val = max(fp_history) if fp_history else 10
                ax_fp.set_ylim(0, max(10, max_fp_val * 1.15))

                # Secondary Axis: Bounding Box Count
                ax_bbox = ax_fp.twinx()
                (line_bbox,) = ax_bbox.plot(
                    cycles_plot,
                    bbox_history,
                    marker="s",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    linewidth=2.5,
                    linestyle="-",
                    color=COLOR_BBOX,
                    label="Bounding Boxes",
                )

                ax_bbox.set_ylabel(
                    "Number of Annotations",
                    fontsize=11,
                    fontweight="medium",
                    color=COLOR_BBOX,
                    labelpad=8,
                )
                ax_bbox.tick_params(axis="y", colors=COLOR_BBOX)

                max_bbox_val = max(bbox_history) if bbox_history else 100
                ax_bbox.set_ylim(0, max(100, max_bbox_val * 1.15))

                ax_fp.set_xticks(cycles_plot)
                ax_fp.set_xticklabels([str(c) for c in cycles_plot], fontsize=9)
                ax_fp.set_xlim(0.6, 5.4)

                # Despine
                ax_fp.spines["top"].set_visible(False)
                ax_bbox.spines["top"].set_visible(False)

                # Legends
                lines = [line_fp, line_bbox]
                labels = [l.get_label() for l in lines]
                ax_fp.legend(
                    lines,
                    labels,
                    loc="lower left",
                    frameon=True,
                    fontsize=10,
                    ncol=1,
                )

                plt.tight_layout()

                # Save
                safe_conf_name = (
                    conf_name.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("+", "_")
                    .lower()
                )

                metric_dir = os.path.join(PLOTS_DIR, "wlt_trajectories", metric["name"])
                os.makedirs(metric_dir, exist_ok=True)

                png_path = os.path.join(
                    metric_dir,
                    f"al_wlt_binary_trajectory_{m_type}_{safe_conf_name}.png",
                )
                pdf_path = os.path.join(
                    metric_dir,
                    f"al_wlt_binary_trajectory_{m_type}_{safe_conf_name}.pdf",
                )

                plt.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved trajectory plots to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
