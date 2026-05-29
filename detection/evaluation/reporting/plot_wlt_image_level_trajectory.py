import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, PLOTS_DIR


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

    # Filter settings
    PROCESS = "clahe"
    VARIANT = "pretrained"
    CYCLES = [0, 1, 2, 3, 4]
    MODELS = ["yolo", "faster_rcnn", "rtdetr"]

    COLOR_AUC = "#1f77b4"  # Blue
    COLOR_F1 = "#ff7f0e"  # Orange
    COLOR_FP = "#6f42c1"  # Purple

    for m_type in MODELS:
        auc_history = []
        f1_history = []
        fp_history = []

        has_data = True
        for cycle in CYCLES:
            # Query the sweep data for this cycle and model
            df_cycle = df[
                (df["model"] == m_type)
                & (df["processing"] == PROCESS)
                & (df["variant"] == VARIANT)
                & (df["cycle"] == cycle)
                & (df["dataset"] == "test_full_seq")
            ]

            if df_cycle.empty:
                print(
                    f"Warning: No data for {m_type} cycle {cycle}. Skipping trajectory plot."
                )
                has_data = False
                break

            # Find the optimal threshold operating point maximizing F1-Score
            idx_max_f1 = df_cycle["f1_score"].idxmax()
            best_row = df_cycle.loc[idx_max_f1]

            auc_history.append(best_row["auc"])
            f1_history.append(best_row["f1_score"])
            fp_history.append(best_row["fp"])

        if not has_data:
            continue

        # PlotTrajectory
        fig, ax = plt.subplots(figsize=(6, 4))
        cycles_plot = [c + 1 for c in CYCLES]

        # 1. Plot AUC and F1-Score on the primary y-axis (Metric Value, scale 0 to 1.12)
        (line_auc,) = ax.plot(
            cycles_plot,
            auc_history,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=2.5,
            color=COLOR_AUC,
            label="ROC-AUC",
        )
        (line_f1,) = ax.plot(
            cycles_plot,
            f1_history,
            marker="s",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=2.5,
            color=COLOR_F1,
            label="Optimal F$_1$",
        )

        # 2. Add second scale to the right for FP count
        ax_fp = ax.twinx()
        (line_fp,) = ax_fp.plot(
            cycles_plot,
            fp_history,
            marker="d",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=2.0,
            linestyle="-",
            color=COLOR_FP,
            label="False Positives",
        )

        # Max AUC annotation calculation
        max_auc = max(auc_history)
        max_auc_idx = auc_history.index(max_auc)
        max_auc_x = cycles_plot[max_auc_idx]

        # Max F1 annotation calculation
        max_f1 = max(f1_history)
        max_f1_idx = f1_history.index(max_f1)
        max_f1_x = cycles_plot[max_f1_idx]

        # Draw red circle around max AUC value
        ax.plot(
            max_auc_x,
            max_auc,
            marker="o",
            markersize=14,
            markeredgecolor="red",
            markerfacecolor="none",
            markeredgewidth=1.5,
            linestyle="",
        )

        # Draw red circle around max F1 value
        ax.plot(
            max_f1_x,
            max_f1,
            marker="o",
            markersize=14,
            markeredgecolor="red",
            markerfacecolor="none",
            markeredgewidth=1.5,
            linestyle="",
        )

        # Intelligently set annotation offsets to avoid overlapping
        offset_auc = 0.03
        va_auc = "bottom"
        offset_f1 = -0.05
        va_f1 = "top"

        if max_auc_x == max_f1_x:
            if max_auc >= max_f1:
                offset_auc = 0.03
                va_auc = "bottom"
                offset_f1 = -0.05
                va_f1 = "top"
            else:
                offset_auc = -0.05
                va_auc = "top"
                offset_f1 = 0.03
                va_f1 = "bottom"
        else:
            offset_auc = -0.05 if max_auc > 0.95 else 0.03
            va_auc = "top" if max_auc > 0.95 else "bottom"

            offset_f1 = -0.05 if max_f1 > 0.95 else 0.03
            va_f1 = "top" if max_f1 > 0.95 else "bottom"

        # Annotate with just the value in red
        ax.text(
            max_auc_x,
            max_auc + offset_auc,
            f"{max_auc:.2f}",
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va=va_auc,
        )

        ax.text(
            max_f1_x,
            max_f1 + offset_f1,
            f"{max_f1:.2f}",
            color="red",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va=va_f1,
        )

        # Minimalist Axis Styling
        ax.set_xlabel(
            "Active Learning Cycle", fontsize=11, fontweight="medium", labelpad=8
        )
        ax.set_ylabel("Metric Value", fontsize=11, fontweight="medium", labelpad=8)
        ax.set_xticks(cycles_plot)
        ax.set_xticklabels([str(c) for c in cycles_plot], fontsize=9)
        ax.set_xlim(0.6, 5.4)
        ax.set_ylim(0.0, 1.12)

        ax_fp.set_ylabel(
            "Number of Image-Level False Positives",
            fontsize=11,
            fontweight="medium",
            color=COLOR_FP,
            labelpad=8,
        )
        ax_fp.tick_params(axis="y", colors=COLOR_FP, labelsize=9, width=1.2, length=4)
        ax_fp.spines["right"].set_color(COLOR_FP)
        ax_fp.spines["right"].set_linewidth(1.2)

        max_fp_val = max(fp_history)
        ax_fp.set_ylim(0, max(10, max_fp_val * 1.15))

        # Despine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax_fp.spines["top"].set_visible(False)
        ax_fp.spines["left"].set_visible(False)
        ax_fp.spines["bottom"].set_visible(False)
        ax_fp.spines["right"].set_visible(True)

        ax.tick_params(axis="both", which="major", labelsize=9, width=1.2, length=4)

        # Legend
        lines = [line_auc, line_f1, line_fp]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="lower right", frameon=False, fontsize=10)

        # Ensure no grid
        ax.grid(False)

        plt.tight_layout()
        fig.subplots_adjust(left=0.12, right=0.88)

        png_path = os.path.join(PLOTS_DIR, f"al_wlt_binary_trajectory_{m_type}.png")
        pdf_path = os.path.join(PLOTS_DIR, f"al_wlt_binary_trajectory_{m_type}.pdf")

        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, dpi=300)
        plt.close()
        print(f"Saved exact trajectory plot to {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
