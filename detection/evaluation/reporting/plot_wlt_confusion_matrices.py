import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import RESULTS_DIR, PLOTS_DIR

plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Outfit", "DejaVu Sans", "Arial"]


def plot_cm(tn, fp, fn, tp, save_path_png, save_path_pdf, normalized=False):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    matrix = np.array([[tn, fp], [fn, tp]], dtype=float)

    if normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze() > 0
        matrix[valid_rows] = matrix[valid_rows] / row_sums[valid_rows]
    labels = ["Background", "WLT"]

    # Plot heatmap without the colorbar as requested
    im = ax.imshow(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1.0 if normalized else None)

    # Show all ticks and label them with large fonts
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16, rotation=90, va="center")

    # Let the horizontal axes labeling appear on bottom
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Loop over data dimensions and create text annotations with very large fonts.
    fmt = ".2f" if normalized else ".0f"
    thresh = (matrix.max() + matrix.min()) / 2.0 if not normalized else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=26,
            )

    # ax.set_title("WLT vs Background", fontsize=20, pad=15)
    ax.set_xlabel("Predicted", fontsize=18, labelpad=10)
    ax.set_ylabel("Ground truth", fontsize=18, labelpad=10)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1)

    fig.tight_layout()
    plt.savefig(save_path_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_path_pdf, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    sweep_csv = os.path.join(
        RESULTS_DIR, "files", "wlt_binary_threshold_sweep_test_pool.csv"
    )
    if not os.path.exists(sweep_csv):
        print("Sweep CSV not found at", sweep_csv)
        return

    df = pd.read_csv(sweep_csv)

    CYCLES = [0, 1, 2, 3, 4]
    MODELS = ["yolo", "faster_rcnn", "rtdetr"]

    CONFIGS = [
        {"process": "plain", "variant": "scratch", "name": "baseline"},
        {"process": "clahe", "variant": "scratch", "name": "clahe_scratch"},
        {"process": "plain", "variant": "pretrained", "name": "pretrained"},
        {"process": "clahe", "variant": "pretrained", "name": "clahe_pretrained"},
    ]

    METRICS = [
        {"id": "max_f1", "name": "max_f1"},
        {"id": "max_f2", "name": "max_f2"},
        {"id": "min_roc", "name": "min_roc_dist"},
        {"id": "recall_95", "name": "recall_95_target"},
    ]

    out_base = os.path.join(PLOTS_DIR, "wlt_confusion_matrices")

    generated = 0
    for metric in METRICS:
        metric_dir = os.path.join(out_base, metric["name"])
        for m_type in MODELS:
            model_dir = os.path.join(metric_dir, m_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                process = config["process"]
                variant = config["variant"]
                conf_name = config["name"]

                for cycle in CYCLES:
                    df_cycle = df[
                        (df["model"] == m_type)
                        & (df["processing"] == process)
                        & (df["variant"] == variant)
                        & (df["cycle"] == cycle)
                        & (df["dataset"] == "test_full_seq")
                    ]

                    if df_cycle.empty:
                        continue

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

                    tn = best_row["tn"]
                    fp = best_row["fp"]
                    fn = best_row["fn"]
                    tp = best_row["tp"]

                    base_filename = f"cm_{m_type}_cycle{cycle}_{conf_name}"
                    png_path = os.path.join(model_dir, base_filename + ".png")
                    pdf_path = os.path.join(model_dir, base_filename + ".pdf")

                    # Unnormalized
                    plot_cm(tn, fp, fn, tp, png_path, pdf_path, normalized=False)

                    # Normalized
                    norm_png_path = os.path.join(
                        model_dir, base_filename + "_normalized.png"
                    )
                    norm_pdf_path = os.path.join(
                        model_dir, base_filename + "_normalized.pdf"
                    )
                    plot_cm(
                        tn, fp, fn, tp, norm_png_path, norm_pdf_path, normalized=True
                    )

                    generated += 2

    print(f"Generated {generated} binary confusion matrices in {out_base}")


if __name__ == "__main__":
    main()
