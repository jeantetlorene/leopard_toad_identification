import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from config import FILES_DIR, PLOTS_DIR

SWEEP_CSV = os.path.join(FILES_DIR, "per_class_threshold_sweep.csv")


def plot_pr_curves():
    if not os.path.exists(SWEEP_CSV):
        print(f"Error: {SWEEP_CSV} not found.")
        return

    df_sweep = pd.read_csv(SWEEP_CSV)

    # Filter for test dataset at Cycle 0, clahe processing
    df_sweep = df_sweep[
        (df_sweep["cycle"] == 0)
        & (df_sweep["dataset"] == "test")
        & (df_sweep["processing"] == "clahe")
    ]

    architectures = df_sweep["model"].unique()

    for arch in architectures:
        fig, ax = plt.subplots(figsize=(8, 8))

        df_arch = df_sweep[df_sweep["model"] == arch]

        for variant in ["scratch", "pretrained"]:
            df_var = df_arch[df_arch["variant"] == variant]
            if df_var.empty:
                continue

            # Macro-average Precision and Recall at each threshold
            mean_metrics = (
                df_var.groupby("threshold")
                .agg({"precision": "mean", "recall": "mean"})
                .sort_index(ascending=False)
            )

            mean_metrics = mean_metrics.sort_values("recall")
            rec = mean_metrics["recall"].values
            prec = mean_metrics["precision"].values.copy()

            # Interpolate to make it monotonic (typical PR curve)
            for i in range(len(prec) - 2, -1, -1):
                prec[i] = np.maximum(prec[i], prec[i + 1])

            # Add endpoints to bound the curve [0, 1]
            rec = np.concatenate(([0.0], rec, [1.0]))
            prec = np.concatenate(([prec[0]], prec, [0.0]))

            # Calculate AP-like area (Trapezoidal integration)
            ap = np.sum((rec[1:] - rec[:-1]) * (prec[1:] + prec[:-1]) / 2)
            max_recall = mean_metrics["recall"].max()

            label = f"{variant.capitalize()} (mAP: {ap:.3f}, AR: {max_recall:.3f})"
            color = "#1f77b4" if variant == "scratch" else "#ff7f0e"
            ax.plot(
                rec,
                prec,
                label=label,
                color=color,
                linewidth=3,
            )
            ax.fill_between(rec, prec, alpha=0.1, color=color)

        ax.set_title(
            f"Precision-Recall Curve: {arch.upper()} Transfer Learning (Cycle 0)",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Recall", fontsize=14)
        ax.set_ylabel("Precision", fontsize=14)
        ax.legend(fontsize=12, loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1.05])
        ax.set_aspect("equal")

        ax.text(
            0.95,
            0.05,
            "All Classes (Macro) - Test Set",
            fontsize=12,
            color="gray",
            ha="right",
            va="bottom",
            alpha=0.5,
        )

        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)

        png_path = os.path.join(
            PLOTS_DIR, f"pr_curve_transfer_learning_{arch}_cycle0.png"
        )
        pdf_path = os.path.join(
            PLOTS_DIR, f"pr_curve_transfer_learning_{arch}_cycle0.pdf"
        )

        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, dpi=300)
        print(f"Saved PR curve plot to: {png_path} and {pdf_path}")
        plt.close()


if __name__ == "__main__":
    plot_pr_curves()
