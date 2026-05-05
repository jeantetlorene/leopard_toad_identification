import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from config import FILES_DIR, PLOTS_DIR

UNIFIED_CSV = os.path.join(FILES_DIR, "unified_model_evaluation.csv")
SWEEP_CSV = os.path.join(FILES_DIR, "per_class_threshold_sweep.csv")


def generate_report():
    if not os.path.exists(UNIFIED_CSV) or not os.path.exists(SWEEP_CSV):
        print(
            "Error: Evaluation CSVs not found. Please run evaluate_all_models.py first."
        )
        return

    # 1. Load Data
    df_unified = pd.read_csv(UNIFIED_CSV)
    df_sweep = pd.read_csv(SWEEP_CSV)

    # Filter for Cycle 0 and Test set
    df_unified = df_unified[
        (df_unified["cycle"] == 0) & (df_unified["dataset"] == "test")
    ]
    df_sweep = df_sweep[(df_sweep["cycle"] == 0) & (df_sweep["dataset"] == "test")]

    # 2. Calculate Average Recall (AR)
    # AR is defined as the macro-average of max recall per class
    ar_data = []
    for (model, proc, var), group in df_sweep.groupby(
        ["model", "processing", "variant"]
    ):
        # For each class, find max recall
        class_recalls = group.groupby("class_name")["recall"].max()
        ar = class_recalls.mean()
        ar_data.append({"model": model, "processing": proc, "variant": var, "AR": ar})
    df_ar = pd.DataFrame(ar_data)

    # 3. Merge with mAP
    df_metrics = df_unified[["model", "processing", "variant", "mAP"]]
    df_final = pd.merge(df_metrics, df_ar, on=["model", "processing", "variant"])

    # 4. Generate Table (Markdown)
    report_path = os.path.join(FILES_DIR, "preprocessing_results.md")
    with open(report_path, "w") as f:
        f.write("# Results: Effect of Preprocessing (Cycle 0)\n\n")
        f.write(
            "This report summarizes the impact of CLAHE on Cycle 0 model performance.\n\n"
        )
        f.write("### Comparative Performance Table (Cycle 0, Test Set)\n")
        f.write("| Architecture | Variant | Processing | mAP | Average Recall |\n")
        f.write("|--------------|---------|------------|-----|----------------|\n")

        # Sort for consistent display
        df_final = df_final.sort_values(["model", "variant", "processing"])

        for _, row in df_final.iterrows():
            f.write(
                f"| {row['model'].upper()} | {row['variant'].capitalize()} | {row['processing'].capitalize()} | {row['mAP']:.4f} | {row['AR']:.4f} |\n"
            )

        f.write("\n### Precision-Recall Visualizations\n\n")
        for arch in df_final["model"].unique():
            f.write(f"#### {arch.upper()}\n")
            f.write(
                f"![{arch.upper()} PR Curve](../plots/pr_curve_{arch}_cycle0.png)\n\n"
            )

    print(f"Report saved to: {report_path}")

    # 5. Plot Precision-Recall Curves
    plot_pr_curves(df_sweep)


def plot_pr_curves(df_sweep):
    architectures = df_sweep["model"].unique()

    for arch in architectures:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Focus on 'pretrained' variant for the primary plot
        df_arch = df_sweep[
            (df_sweep["model"] == arch) & (df_sweep["variant"] == "pretrained")
        ]

        for proc in ["plain", "clahe"]:
            df_proc = df_arch[df_arch["processing"] == proc]
            if df_proc.empty:
                continue

            # Macro-average Precision and Recall at each threshold
            mean_metrics = (
                df_proc.groupby("threshold")
                .agg({"precision": "mean", "recall": "mean"})
                .sort_index(ascending=False)
            )

            # Sort by recall for better plotting
            mean_metrics = mean_metrics.sort_values("recall")

            rec = mean_metrics["recall"].values
            prec = mean_metrics["precision"].values.copy()

            # Interpolate to make it monotonic (typical PR curve)
            # p(r) = max_{r' >= r} p(r')
            for i in range(len(prec) - 2, -1, -1):
                prec[i] = np.maximum(prec[i], prec[i + 1])

            # Add endpoints to bound the curve [0, 1]
            # Ensure it starts at recall 0 and ends at recall 1
            rec = np.concatenate(([0.0], rec, [1.0]))
            prec = np.concatenate(([prec[0]], prec, [0.0]))

            # Calculate AP-like area (Trapezoidal integration on interpolated curve)
            ap = np.sum((rec[1:] - rec[:-1]) * (prec[1:] + prec[:-1]) / 2)
            max_recall = mean_metrics["recall"].max()

            label = f"{proc.capitalize()} (mAP: {ap:.3f}, AR: {max_recall:.3f})"
            color = "#1f77b4" if proc == "plain" else "#ff7f0e"
            ax.plot(
                rec,
                prec,
                label=label,
                color=color,
                linewidth=3,
            )
            # Fill area under curve
            ax.fill_between(rec, prec, alpha=0.1, color=color)

        ax.set_title(
            f"Precision-Recall Curve: {arch.upper()} (Cycle 0)",
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

        # Add "all classes" watermark or similar style
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
        plot_path = os.path.join(PLOTS_DIR, f"pr_curve_{arch}_cycle0.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved PR curve plot to: {plot_path}")
        plt.close()


if __name__ == "__main__":
    generate_report()
