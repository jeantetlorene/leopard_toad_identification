import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, PLOTS_DIR, RESULTS_DIR
from eval_utils.data_utils import refresh_results
from eval_utils.metrics import calculate_detection_metrics

UNIFIED_CSV = os.path.join(FILES_DIR, "active_learning_unified_evaluation.csv")
CLASSES = ["Western_Leopard_Toad", "Small_Mammal", "Other_Amphibian"]
BUDGET_MAP = {0: 130, 1: 230, 2: 330, 3: 430, 4: 530}


def load_data():
    if not os.path.exists(UNIFIED_CSV):
        print("Error: Unified CSV not found.")
        return None
    df = pd.read_csv(UNIFIED_CSV)
    return df[df["dataset"] == "test"]


def plot_trajectories(df):
    models = df["model"].unique()

    for m_type in models:
        # Baseline is Cycle 0, plain, scratch
        df_base = df[
            (df["model"] == m_type)
            & (df["processing"] == "plain")
            & (df["variant"] == "scratch")
            & (df["cycle"] == 0)
        ]

        # AL progression (we'll plot clahe pretrained as the main AL pipeline)
        df_al = df[
            (df["model"] == m_type)
            & (df["processing"] == "clahe")
            & (df["variant"] == "pretrained")
        ].sort_values("cycle")

        if df_al.empty:
            continue

        budgets = [BUDGET_MAP.get(c, 0) for c in df_al["cycle"]]

        # 1. mAP Progression
        plt.figure(figsize=(10, 6))
        for cls in CLASSES:
            y_al = df_al[f"{cls}_ap"]
            plt.plot(budgets, y_al, marker="o", label=f"{cls} (AL)")

            if not df_base.empty:
                base_val = df_base[f"{cls}_ap"].values[0]
                plt.axhline(
                    y=base_val, linestyle="--", alpha=0.6, label=f"{cls} (Baseline)"
                )

        plt.xlabel("Cumulative Labelled Images")
        plt.ylabel("Average Precision (AP)")
        plt.title(f"{m_type.upper()} Class-Specific AP Trajectory")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"al_trajectory_ap_{m_type}.png"))
        plt.savefig(os.path.join(PLOTS_DIR, f"al_trajectory_ap_{m_type}.pdf"))
        plt.close()

        # 2. Recall Progression
        plt.figure(figsize=(10, 6))
        for cls in CLASSES:
            y_al = df_al[f"{cls}_recall_optimal"]
            plt.plot(budgets, y_al, marker="s", label=f"{cls} (AL)")

            if not df_base.empty:
                base_val = df_base[f"{cls}_recall_optimal"].values[0]
                plt.axhline(
                    y=base_val, linestyle="--", alpha=0.6, label=f"{cls} (Baseline)"
                )

        plt.xlabel("Cumulative Labelled Images")
        plt.ylabel("Recall (Optimal Threshold)")
        plt.title(f"{m_type.upper()} Class-Specific Recall Trajectory")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"al_trajectory_recall_{m_type}.png"))
        plt.savefig(os.path.join(PLOTS_DIR, f"al_trajectory_recall_{m_type}.pdf"))
        plt.close()


def plot_confidence_distributions(df):
    cycle = 4
    MODELS = ["yolo", "faster_rcnn", "rtdetr"]
    PROCESSINGS = ["clahe", "plain"]
    VARIANTS = ["pretrained", "scratch"]

    def box_iou(box1, box2):
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
        inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
        return inter_area / union_area if union_area > 0 else 0

    CLASS_MAP = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}

    for m_type in MODELS:
        for proc in PROCESSINGS:
            for var in VARIANTS:
                df_u = df[
                    (df["model"] == m_type)
                    & (df["processing"] == proc)
                    & (df["variant"] == var)
                    & (df["cycle"] == cycle)
                ]

                raw_json_path = os.path.join(
                    RESULTS_DIR,
                    f"{m_type}_{proc}",
                    f"cycle_{cycle}_{var}_test_raw.json",
                )
                if not os.path.exists(raw_json_path):
                    continue

                with open(raw_json_path, "r") as f:
                    results = json.load(f)

                # Refresh ground truth from clean data
                results = refresh_results(results, is_full_seq=False)

                tp_scores = {c: [] for c in CLASS_MAP.values()}
                fp_scores = {c: [] for c in CLASS_MAP.values()}

                for res in results:
                    preds = res["predictions"]
                    gts = res["gt_boxes"]

                    gt_matched = [False] * len(gts)
                    preds.sort(key=lambda x: x["conf"], reverse=True)

                    for p in preds:
                        c_name = CLASS_MAP[p["cls"]]
                        best_iou = -1
                        best_gt_idx = -1

                        for i, gt in enumerate(gts):
                            if gt["cls"] == p["cls"] and not gt_matched[i]:
                                iou = box_iou(p["bbox"], gt["bbox"])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = i

                        if best_iou >= 0.5:
                            tp_scores[c_name].append(p["conf"])
                            gt_matched[best_gt_idx] = True
                        else:
                            fp_scores[c_name].append(p["conf"])

                # Now plot KDE for each class
                for cls in CLASS_MAP.values():
                    plt.figure(figsize=(8, 5))

                    if tp_scores[cls]:
                        try:
                            sns.kdeplot(
                                tp_scores[cls],
                                fill=True,
                                color="green",
                                label="True Positives",
                                alpha=0.5,
                                bw_adjust=0.5,
                            )
                        except Exception:
                            pass

                    if fp_scores[cls]:
                        try:
                            sns.kdeplot(
                                fp_scores[cls],
                                fill=True,
                                color="red",
                                label="False Positives",
                                alpha=0.5,
                                bw_adjust=0.5,
                            )
                        except Exception:
                            pass

                    opt_thresh_col = f"{cls}_optimal_threshold"
                    if not df_u.empty and opt_thresh_col in df_u.columns:
                        opt_thresh = df_u[opt_thresh_col].values[0]
                        if pd.notna(opt_thresh):
                            plt.axvline(
                                x=opt_thresh,
                                color="blue",
                                # linestyle="--",
                                linewidth=2,
                                label="Optimal validation threshold",
                            )

                    plt.xlabel("Confidence Score")
                    plt.ylabel("Density")
                    # plt.title(
                    #     f"Overlapping Confidence Score Distribution ({m_type.upper()} {proc} {var} Cycle 4)\nClass: {cls}"
                    # )
                    plt.xlim(0, 1)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            PLOTS_DIR,
                            f"al_confidence_kde_cycle4_{m_type}_{proc}_{var}_{cls}.png",
                        )
                    )
                    plt.savefig(
                        os.path.join(
                            PLOTS_DIR,
                            f"al_confidence_kde_cycle4_{m_type}_{proc}_{var}_{cls}.pdf",
                        )
                    )
                    plt.close()


def plot_ap_ar_trajectories():
    MODELS = ["yolo", "faster_rcnn", "rtdetr"]
    PROCESS = "clahe"
    VARIANT = "pretrained"
    CYCLES = [0, 1, 2, 3, 4]
    CLASS_IDS = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}

    # Use a premium color palette for publication-quality aesthetic
    COLOR_AP = "#1f77b4"  # Elegant blue for Average Precision
    COLOR_AR = "#ff7f0e"  # Elegant orange for Absolute Recall

    for m_type in MODELS:
        ap_history = {c: [] for c in CLASS_IDS.values()}
        ar_history = {c: [] for c in CLASS_IDS.values()}

        for cycle in CYCLES:
            root_key = f"{m_type}_{PROCESS}"
            raw_json_path = os.path.join(
                RESULTS_DIR, root_key, f"cycle_{cycle}_{VARIANT}_test_raw.json"
            )

            if os.path.exists(raw_json_path):
                with open(raw_json_path, "r") as f:
                    raw_results = json.load(f)
                raw_results = refresh_results(raw_results, is_full_seq=False)
                det_metrics = calculate_detection_metrics(raw_results)

                for cls_id, cls_name in CLASS_IDS.items():
                    ap = det_metrics["class_aps"].get(cls_id, 0.0)
                    curves = det_metrics["class_curves"].get(cls_id, {})
                    mrec = curves.get("recall", [])
                    ar = mrec[-2] if len(mrec) >= 2 else 0.0

                    ap_history[cls_name].append(ap)
                    ar_history[cls_name].append(ar)
            else:
                for cls_name in CLASS_IDS.values():
                    ap_history[cls_name].append(0.0)
                    ar_history[cls_name].append(0.0)

        # Plot for each class: Clean, simple, no grid or title
        for cls_name in CLASS_IDS.values():
            fig, ax = plt.subplots(figsize=(6, 4))

            cycles_plot = [c + 1 for c in CYCLES]

            # Clean, premium line styles: wider lines, crisp markers with white borders
            ax.plot(
                cycles_plot,
                ap_history[cls_name],
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                linewidth=2.5,
                color=COLOR_AP,
                label="AP$_{50}$",
            )
            ax.plot(
                cycles_plot,
                ar_history[cls_name],
                marker="s",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                linewidth=2.5,
                color=COLOR_AR,
                label="AR$_{50}$",
            )

            # Max AP annotation calculation
            max_ap = max(ap_history[cls_name])
            max_ap_idx = ap_history[cls_name].index(max_ap)
            max_ap_x = cycles_plot[max_ap_idx]

            # Max AR annotation calculation
            max_ar = max(ar_history[cls_name])
            max_ar_idx = ar_history[cls_name].index(max_ar)
            max_ar_x = cycles_plot[max_ar_idx]

            # Draw red circle around max AP value
            ax.plot(
                max_ap_x,
                max_ap,
                marker="o",
                markersize=14,
                markeredgecolor="red",
                markerfacecolor="none",
                markeredgewidth=1.5,
                linestyle="",
                # label="Max Value",
            )

            # Draw red circle around max AR value (without duplicate label)
            ax.plot(
                max_ar_x,
                max_ar,
                marker="o",
                markersize=14,
                markeredgecolor="red",
                markerfacecolor="none",
                markeredgewidth=1.5,
                linestyle="",
            )

            # Intelligently set annotation offsets to avoid text box overlap
            offset_ap = 0.03
            va_ap = "bottom"
            offset_ar = -0.05
            va_ar = "top"

            if max_ap_x == max_ar_x:
                if max_ap >= max_ar:
                    offset_ap = 0.03
                    va_ap = "bottom"
                    offset_ar = -0.05
                    va_ar = "top"
                else:
                    offset_ap = -0.05
                    va_ap = "top"
                    offset_ar = 0.03
                    va_ar = "bottom"
            else:
                # If they are not in the same cycle, adjust text positions to not cross limits
                offset_ap = -0.05 if max_ap > 0.95 else 0.03
                va_ap = "top" if max_ap > 0.95 else "bottom"

                offset_ar = -0.05 if max_ar > 0.95 else 0.03
                va_ar = "top" if max_ar > 0.95 else "bottom"

            # Annotate with just the value in red
            ax.text(
                max_ap_x,
                max_ap + offset_ap,
                f"{max_ap:.3f}",
                color="red",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va=va_ap,
            )

            ax.text(
                max_ar_x,
                max_ar + offset_ar,
                f"{max_ar:.3f}",
                color="red",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va=va_ar,
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

            # Despine: Remove top and right borders for an ultra-clean academic look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            # Tick styling
            ax.tick_params(axis="both", which="major", labelsize=9, width=1.2, length=4)

            # Elegant legend with no frame border
            ax.legend(loc="lower right", frameon=False, fontsize=10)

            # Ensure no grid
            ax.grid(False)

            plt.tight_layout()

            plt.savefig(
                os.path.join(PLOTS_DIR, f"al_ap_ar_trajectory_{m_type}_{cls_name}.png"),
                dpi=300,
            )
            plt.savefig(
                os.path.join(PLOTS_DIR, f"al_ap_ar_trajectory_{m_type}_{cls_name}.pdf"),
                dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_trajectories(df)
        plot_confidence_distributions(df)
        plot_ap_ar_trajectories()
        print("Active learning plots generated successfully in results/plots/")
