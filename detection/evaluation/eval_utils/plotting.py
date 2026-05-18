import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Premium, modern visual style configurations
plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Outfit", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

CLASSES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}


def box_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h] in normalized coordinates."""
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def generate_validation_plots(results, output_dir):
    """
    Generates the 6 standard Ultralytics-style validation plots for Faster R-CNN:
    - BoxPR_curve.png
    - BoxP_curve.png
    - BoxR_curve.png
    - BoxF1_curve.png
    - confusion_matrix.png
    - confusion_matrix_normalized.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse and group predictions & ground truths
    all_preds = []
    all_gts = []
    for res in results:
        img_id = res["path"]
        for p in res["predictions"]:
            all_preds.append(
                {
                    "conf": p["conf"],
                    "cls": p["cls"],
                    "bbox": p["bbox"],
                    "img_id": img_id,
                }
            )
        for gt in res["gt_boxes"]:
            all_gts.append(
                {
                    "cls": gt["cls"],
                    "bbox": gt["bbox"],
                    "img_id": img_id,
                    "matched": False,
                }
            )

    # Distinct, premium line colors matching Ultralytics YOLO/RT-DETR style
    colors = ["#FF3838", "#FF9D2E", "#2C82C9"]
    mean_color = "#333333"

    # Define confidence threshold steps
    thresholds = np.linspace(0.0, 1.0, 100)
    recall_grid = np.linspace(0.0, 1.0, 101)

    class_curves = {}
    class_p_sweeps = {}
    class_r_sweeps = {}
    class_f1_sweeps = {}

    for c, cls_name in CLASSES.items():
        cls_preds = [p for p in all_preds if p["cls"] == c]
        cls_gts = [g for g in all_gts if g["cls"] == c]
        n_gt = len(cls_gts)

        if n_gt == 0:
            continue

        # Sort predictions by confidence
        cls_preds.sort(key=lambda x: x["conf"], reverse=True)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        gts_by_img = defaultdict(list)
        for g in cls_gts:
            gts_by_img[g["img_id"]].append(g)

        for i, pred in enumerate(cls_preds):
            img_gts = gts_by_img.get(pred["img_id"], [])
            best_iou = -1
            best_gt = None

            for gt in img_gts:
                if gt["matched"]:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= 0.5 and best_gt is not None:
                tp[i] = 1
                best_gt["matched"] = True
            else:
                fp[i] = 1

        # Reset GT match state for subsequent steps
        for g in cls_gts:
            g["matched"] = False

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        # Raw precision and recall corresponding to sorted confidence cuts
        raw_recalls = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        raw_precisions = (
            tp_cum / (tp_cum + fp_cum) if len(tp_cum) > 0 else np.zeros_like(tp_cum)
        )
        raw_confs = np.array([p["conf"] for p in cls_preds])

        # VOC monotonic padding for standard PR Curve
        mrec = np.concatenate(([0.0], raw_recalls, [1.0]))
        mpre = np.concatenate(([1.0], raw_precisions, [0.0]))
        for j in range(mpre.size - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])

        # Interpolate precision on a uniform recall grid for macro averaging
        p_interp = np.interp(recall_grid, mrec, mpre)
        class_curves[c] = p_interp

        # Compute P, R, F1 for the 100 confidence threshold steps
        p_sweep = []
        r_sweep = []
        f1_sweep = []

        for t in thresholds:
            active_mask = raw_confs >= t
            if not np.any(active_mask):
                p_val, r_val, f1_val = 1.0, 0.0, 0.0
            else:
                last_idx = np.where(active_mask)[0][-1]
                p_val = raw_precisions[last_idx]
                r_val = raw_recalls[last_idx]
                f1_val = (
                    2 * (p_val * r_val) / (p_val + r_val)
                    if (p_val + r_val) > 0
                    else 0.0
                )

            p_sweep.append(p_val)
            r_sweep.append(r_val)
            f1_sweep.append(f1_val)

        class_p_sweeps[c] = np.array(p_sweep)
        class_r_sweeps[c] = np.array(r_sweep)
        class_f1_sweeps[c] = np.array(f1_sweep)

    # Compute macro-average curves
    mean_pr = np.mean(list(class_curves.values()), axis=0)
    mean_p_sweep = np.mean(list(class_p_sweeps.values()), axis=0)
    mean_r_sweep = np.mean(list(class_r_sweeps.values()), axis=0)
    mean_f1_sweep = np.mean(list(class_f1_sweeps.values()), axis=0)

    # Calculate mAP@0.5 under VOC scheme
    mAP_50 = np.mean(mean_pr[:-1])

    # 2. Draw Precision-Recall Curve (BoxPR_curve.png)
    plt.figure(figsize=(7, 6.5), dpi=300)
    for c, cls_name in CLASSES.items():
        if c in class_curves:
            plt.plot(
                recall_grid,
                class_curves[c],
                label=f"{cls_name} {np.mean(class_curves[c][:-1]):.3f}",
                color=colors[c],
                linewidth=1.8,
            )
    plt.plot(
        recall_grid,
        mean_pr,
        label=f"all classes {mAP_50:.3f} mAP@0.5",
        color=mean_color,
        linewidth=2.8,
        linestyle="-",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="medium")
    plt.ylabel("Precision", fontsize=11, fontweight="medium")
    plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    plt.gca().set_aspect("equal")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "BoxPR_curve.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 3. Draw Precision vs Confidence Sweep (BoxP_curve.png)
    plt.figure(figsize=(7, 6.5), dpi=300)
    for c, cls_name in CLASSES.items():
        if c in class_p_sweeps:
            plt.plot(
                thresholds,
                class_p_sweeps[c],
                label=f"{cls_name}",
                color=colors[c],
                linewidth=1.8,
            )
    plt.plot(
        thresholds,
        mean_p_sweep,
        label="all classes",
        color=mean_color,
        linewidth=2.8,
        linestyle="-",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Confidence", fontsize=11, fontweight="medium")
    plt.ylabel("Precision", fontsize=11, fontweight="medium")
    plt.title("Precision-Confidence Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "BoxP_curve.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 4. Draw Recall vs Confidence Sweep (BoxR_curve.png)
    plt.figure(figsize=(7, 6.5), dpi=300)
    for c, cls_name in CLASSES.items():
        if c in class_r_sweeps:
            plt.plot(
                thresholds,
                class_r_sweeps[c],
                label=f"{cls_name}",
                color=colors[c],
                linewidth=1.8,
            )
    plt.plot(
        thresholds,
        mean_r_sweep,
        label="all classes",
        color=mean_color,
        linewidth=2.8,
        linestyle="-",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Confidence", fontsize=11, fontweight="medium")
    plt.ylabel("Recall", fontsize=11, fontweight="medium")
    plt.title("Recall-Confidence Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "BoxR_curve.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 5. Draw F1 vs Confidence Sweep (BoxF1_curve.png)
    plt.figure(figsize=(7, 6.5), dpi=300)
    peak_idx = np.argmax(mean_f1_sweep)
    peak_f1 = mean_f1_sweep[peak_idx]
    peak_conf = thresholds[peak_idx]

    for c, cls_name in CLASSES.items():
        if c in class_f1_sweeps:
            plt.plot(
                thresholds,
                class_f1_sweeps[c],
                label=f"{cls_name}",
                color=colors[c],
                linewidth=1.8,
            )
    plt.plot(
        thresholds,
        mean_f1_sweep,
        label=f"all classes {peak_f1:.2f} at {peak_conf:.2f}",
        color=mean_color,
        linewidth=2.8,
        linestyle="-",
    )
    plt.axvline(
        x=peak_conf,
        color="grey",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Confidence", fontsize=11, fontweight="medium")
    plt.ylabel("F1", fontsize=11, fontweight="medium")
    plt.title("F1-Confidence Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(
        loc="lower center",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=9,
    )
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "BoxF1_curve.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 6. Compute Confusion Matrix (Raw and Normalized)
    # Background index is 3
    confusion_matrix = np.zeros((4, 4))

    for res in results:
        # Ground truths for this image
        gts_img = [
            {"cls": gt["cls"], "bbox": gt["bbox"], "matched": False}
            for gt in res["gt_boxes"]
        ]
        # Predictions with confidence >= 0.25 (Ultralytics standard)
        preds_img = [
            {"cls": p["cls"], "bbox": p["bbox"]}
            for p in res["predictions"]
            if p["conf"] >= 0.25
        ]

        # Greedy match predictions to GT
        for pred in preds_img:
            best_iou = -1
            best_gt = None
            for gt in gts_img:
                if gt["matched"]:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= 0.5 and best_gt is not None:
                best_gt["matched"] = True
                confusion_matrix[best_gt["cls"], pred["cls"]] += 1
            else:
                # Unmatched prediction -> predicted class on background (FP)
                confusion_matrix[3, pred["cls"]] += 1

        # Unmatched ground truths -> true class on background (FN)
        for gt in gts_img:
            if not gt["matched"]:
                confusion_matrix[gt["cls"], 3] += 1

    # Labels for display
    cm_labels = [
        "Other_Amphibian",
        "Small_Mammal",
        "Western_Leopard_Toad",
        "background",
    ]

    # Plot Confusion Matrix (Raw Counts)
    plot_confusion_matrix_heatmap(
        confusion_matrix,
        cm_labels,
        os.path.join(output_dir, "confusion_matrix.png"),
        normalized=False,
    )

    # Plot Normalized Confusion Matrix
    # Normalize by row sums (true classes) to get fractional distributions
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = np.zeros_like(confusion_matrix)
    valid_rows = row_sums > 0
    normalized_cm[valid_rows[:, 0]] = (
        confusion_matrix[valid_rows[:, 0]] / row_sums[valid_rows[:, 0]]
    )

    plot_confusion_matrix_heatmap(
        normalized_cm,
        cm_labels,
        os.path.join(output_dir, "confusion_matrix_normalized.png"),
        normalized=True,
    )


def plot_confusion_matrix_heatmap(matrix, labels, save_path, normalized=False):
    """Draws a premium-quality confusion matrix heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)

    # Custom premium Blue colormap
    im = ax.imshow(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1.0 if normalized else None)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, fontweight="medium", rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9, fontweight="medium")

    # Let the horizontal axes labeling appear on top
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalized else ".0f"
    thresh = (matrix.max() + matrix.min()) / 2.0 if not normalized else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Don't show text in background-background box (meaningless)
            if i == 3 and j == 3:
                continue

            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_title(
        "Confusion Matrix" + (" (Normalized)" if normalized else ""),
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted class", fontsize=11, fontweight="medium", labelpad=10)
    ax.set_ylabel("True class", fontsize=11, fontweight="medium", labelpad=10)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
