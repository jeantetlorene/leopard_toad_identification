import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, FILES_DIR, PLOTS_DIR, CONF_THRESHOLDS
from eval_utils.data_utils import refresh_results


def compute_sweep(results, thresholds, is_toad_specific=False, is_megadetector=False):
    """
    Compute binary classification metrics across all confidence thresholds.
    """
    # 1. Determine ground truth for each image
    if is_toad_specific:
        # positive if there is at least one ground truth box of class 2 (toad)
        is_positive = np.array(
            [any(gt["cls"] == 2 for gt in res["gt_boxes"]) for res in results],
            dtype=bool,
        )
    else:
        # positive if there is any ground truth box of any class
        is_positive = np.array(
            [len(res["gt_boxes"]) > 0 for res in results], dtype=bool
        )

    # 2. Determine maximum confidence prediction score for each image
    max_confs = []
    for res in results:
        preds = res["predictions"]
        if is_toad_specific and not is_megadetector:
            # only consider predictions of class 2 (Western Leopard Toad)
            scores = [p["conf"] for p in preds if p["cls"] == 2]
        else:
            # consider all predictions class-agnostically
            scores = [p["conf"] for p in preds]
        max_confs.append(max(scores + [0.0]))
    max_confs = np.array(max_confs)

    # 3. Sweep thresholds
    metrics = []
    total_images = len(results)

    for thresh in thresholds:
        has_detection = max_confs >= thresh
        tp = np.sum(has_detection & is_positive)
        fn = np.sum(~has_detection & is_positive)
        fp = np.sum(has_detection & ~is_positive)
        tn = np.sum(~has_detection & ~is_positive)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        labor_reduction = (tn + fn) / total_images if total_images > 0 else 0.0

        metrics.append(
            {
                "threshold": thresh,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "recall": float(recall),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1_score": float(f1),
                "labor_reduction": float(labor_reduction),
            }
        )

    return metrics, is_positive, max_confs


def generate_image_level_report():
    print("Generating Comprehensive Image-Level Evaluation Report...")

    # Paths
    report_md = os.path.join(FILES_DIR, "image_level_results.md")

    # Define baselines explicitly
    baselines = [
        {
            "model": "yolo",
            "processing": "plain",
            "variant": "scratch",
            "display_name": "YOLO",
        },
        {
            "model": "faster_rcnn",
            "processing": "plain",
            "variant": "scratch",
            "display_name": "Faster R-CNN",
        },
        {
            "model": "rtdetr",
            "processing": "plain",
            "variant": "scratch",
            "display_name": "RT-DETR",
        },
        {
            "model": "megadetector",
            "processing": "plain",
            "variant": "pretrained",
            "display_name": "MegaDetector",
        },
    ]

    # Pre-load and refresh prediction files once to save processing time
    loaded_datasets = {}
    for b in baselines:
        raw_filename = f"cycle_0_{b['variant']}_test_full_seq_raw.json"
        raw_filepath = os.path.join(
            RESULTS_DIR, f"{b['model']}_{b['processing']}", raw_filename
        )

        if not os.path.exists(raw_filepath):
            print(f"Warning: {raw_filepath} not found. Skipping.")
            continue

        print(f"Loading and refreshing ground truth for {b['display_name']}...")
        with open(raw_filepath, "r") as f:
            raw_results = json.load(f)

        loaded_datasets[b["model"]] = refresh_results(raw_results, is_full_seq=True)

    # Store report results
    tables_data = {"class_agnostic": [], "class_specific": []}

    for is_specific in [False, True]:
        table_key = "class_specific" if is_specific else "class_agnostic"

        for b in baselines:
            model_key = b["model"]
            if model_key not in loaded_datasets:
                continue

            results = loaded_datasets[model_key]
            is_md = model_key == "megadetector"

            # Compute metric sweep
            metrics, is_pos, scores = compute_sweep(
                results,
                CONF_THRESHOLDS,
                is_toad_specific=is_specific,
                is_megadetector=is_md,
            )

            # Compute precise ROC-AUC
            auc_val = np.nan
            if len(np.unique(is_pos)) > 1:
                fpr, tpr, _ = roc_curve(is_pos, scores)
                auc_val = auc(fpr, tpr)

            # Find peak F1-Score operating point
            max_f1_op = max(metrics, key=lambda x: x["f1_score"])

            # Find High-Recall operating point (recall >= 0.95)
            high_rec_candidates = [m for m in metrics if m["recall"] >= 0.95]
            if high_rec_candidates:
                # Maximize specificity/labor reduction by choosing the highest threshold
                high_rec_op = max(high_rec_candidates, key=lambda x: x["threshold"])
            else:
                high_rec_op = None

            # Find Moderate-Recall operating point (recall >= 0.85)
            rec_85_candidates = [m for m in metrics if m["recall"] >= 0.85]
            if rec_85_candidates:
                rec_85_op = max(rec_85_candidates, key=lambda x: x["threshold"])
            else:
                rec_85_op = None

            # Calculate absolute maximum recall achieved
            max_recall = max(m["recall"] for m in metrics)

            tables_data[table_key].append(
                {
                    "display_name": b["display_name"],
                    "auc": auc_val,
                    "max_f1_op": max_f1_op,
                    "high_rec_op": high_rec_op,
                    "rec_85_op": rec_85_op,
                    "max_recall": max_recall,
                }
            )

    # Write report markdown
    with open(report_md, "w") as f:
        f.write(
            "# Results: Image-Level Binary Classification & Labor Reduction (Cycle 0)\n\n"
        )
        f.write(
            "This report documents the image-level binary filtering performance and manual annotation labor "
            "reduction achieved by baseline models on the unlabelled test pool (147,351 frames).\n\n"
        )

        f.write(
            "For each model, we evaluate performance at three separate operating points:\n"
            "1.  **Optimal $F_1$-Score Operating Point**: Maximizes the geometric mean of Precision and Recall. "
            "Recommended for general machine learning model comparisons.\n"
            "2.  **High-Recall Safety Operating Point (Target $\\ge 95\\%$)**: Restricts the search space to "
            "configurations that guarantee at least $95\\%$ target recall, and maximizes specificity. "
            "Recommended for risk-averse ecological deployment where target species must not be missed.\n"
            "3.  **Moderate High-Recall Operating Point (Target $\\ge 85\\%$)**: Restricts the search space to "
            "configurations that guarantee at least $85\\%$ target recall, and maximizes specificity. "
            "Offers a balanced compromise with higher specificity and labor savings where acceptable.\n\n"
        )

        # TABLE 1: Class-Agnostic Evaluation
        f.write(
            "## 1. Class-Agnostic Evaluation (General Animal vs. Empty Background)\n\n"
        )
        f.write(
            "In this configuration, any animal detection of any taxon is treated as positive. This is highly robust "
            "to night-time taxonomic misclassification and allows a direct, fair benchmark against **MegaDetector v5a**.\n\n"
        )

        f.write(
            "| Baseline Model | Area Under ROC (AUC) | Metric Focus | Optimal Conf. Threshold | Achieved F1-Score | Achieved Recall (Sensitivity) | Achieved Specificity (TNR) | Achieved Precision | Manual Labor Saved |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for r in tables_data["class_agnostic"]:
            name = r["display_name"]
            auc_str = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "N/A"
            f1_op = r["max_f1_op"]
            rec_op = r["high_rec_op"]
            rec_85_op = r["rec_85_op"]

            # Row 1: Max F1
            f.write(
                f"| **{name}** | **{auc_str}** | Max $F_1$-Score | {f1_op['threshold']:.2f} | {f1_op['f1_score']:.4f} | {f1_op['recall'] * 100:.2f}% | {f1_op['specificity'] * 100:.2f}% | {f1_op['precision'] * 100:.2f}% | **{f1_op['labor_reduction'] * 100:.2f}%** |\n"
            )
            # Row 2: High-Recall Target (95%)
            if rec_op is not None:
                f.write(
                    f"| | | High-Recall ($\\ge 95\\%$) | {rec_op['threshold']:.2f} | {rec_op['f1_score']:.4f} | {rec_op['recall'] * 100:.2f}% | {rec_op['specificity'] * 100:.2f}% | {rec_op['precision'] * 100:.2f}% | **{rec_op['labor_reduction'] * 100:.2f}%** |\n"
                )
            else:
                f.write(
                    f"| | | High-Recall ($\\ge 95\\%$) | N/A | N/A | N/A | N/A | N/A | **N/A (Max Rec: {r['max_recall'] * 100:.1f}%)** |\n"
                )
            # Row 3: High-Recall Target (85%)
            if rec_85_op is not None:
                f.write(
                    f"| | | High-Recall ($\\ge 85\\%$) | {rec_85_op['threshold']:.2f} | {rec_85_op['f1_score']:.4f} | {rec_85_op['recall'] * 100:.2f}% | {rec_85_op['specificity'] * 100:.2f}% | {rec_85_op['precision'] * 100:.2f}% | **{rec_85_op['labor_reduction'] * 100:.2f}%** |\n"
                )
            else:
                f.write(
                    f"| | | High-Recall ($\\ge 85\\%$) | N/A | N/A | N/A | N/A | N/A | **N/A (Max Rec: {r['max_recall'] * 100:.1f}%)** |\n"
                )
            f.write("| | | | | | | | | |\n")  # Divider row

        f.write("\n### Class-Agnostic ROC Curve Visualization\n\n")
        f.write(
            "![Baseline Architectures Bounded ROC Curve (Class-Agnostic)](../plots/binary_roc_baseline.png)\n\n"
        )

        # TABLE 2: Class-Specific Evaluation
        f.write(
            "\n## 2. Class-Specific Evaluation (Western Leopard Toad vs. Background/Other Taxa)\n\n"
        )
        f.write(
            "In this configuration, only annotations containing the target species (**Western Leopard Toad**) are "
            "treated as positive. For fine-tuned custom models, only predictions of Class 2 (`Western_Leopard_Toad`) "
            "trigger a positive flag. For the zero-shot **MegaDetector**, all animal detections trigger a positive "
            "flag (as it is class-agnostic), but performance is measured strictly against target toad presence.\n\n"
        )

        f.write(
            "| Baseline Model | Area Under ROC (AUC) | Metric Focus | Optimal Conf. Threshold | Achieved F1-Score | Achieved Recall (Sensitivity) | Achieved Specificity (TNR) | Achieved Precision | Manual Labor Saved |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for r in tables_data["class_specific"]:
            name = r["display_name"]
            auc_str = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "N/A"
            f1_op = r["max_f1_op"]
            rec_op = r["high_rec_op"]
            rec_85_op = r["rec_85_op"]

            # Row 1: Max F1
            f.write(
                f"| **{name}** | **{auc_str}** | Max $F_1$-Score | {f1_op['threshold']:.2f} | {f1_op['f1_score']:.4f} | {f1_op['recall'] * 100:.2f}% | {f1_op['specificity'] * 100:.2f}% | {f1_op['precision'] * 100:.2f}% | **{f1_op['labor_reduction'] * 100:.2f}%** |\n"
            )
            # Row 2: High-Recall Target (95%)
            if rec_op is not None:
                f.write(
                    f"| | | High-Recall ($\\ge 95\\%$) | {rec_op['threshold']:.2f} | {rec_op['f1_score']:.4f} | {rec_op['recall'] * 100:.2f}% | {rec_op['specificity'] * 100:.2f}% | {rec_op['precision'] * 100:.2f}% | **{rec_op['labor_reduction'] * 100:.2f}%** |\n"
                )
            else:
                f.write(
                    f"| | | High-Recall ($\\ge 95\\%$) | N/A | N/A | N/A | N/A | N/A | **N/A (Max Rec: {r['max_recall'] * 100:.1f}%)** |\n"
                )
            # Row 3: High-Recall Target (85%)
            if rec_85_op is not None:
                f.write(
                    f"| | | High-Recall ($\\ge 85\\%$) | {rec_85_op['threshold']:.2f} | {rec_85_op['f1_score']:.4f} | {rec_85_op['recall'] * 100:.2f}% | {rec_85_op['specificity'] * 100:.2f}% | {rec_85_op['precision'] * 100:.2f}% | **{rec_85_op['labor_reduction'] * 100:.2f}%** |\n"
                )
            else:
                f.write(
                    f"| | | High-Recall ($\\ge 85\\%$) | N/A | N/A | N/A | N/A | N/A | **N/A (Max Rec: {r['max_recall'] * 100:.1f}%)** |\n"
                )
            f.write("| | | | | | | | | |\n")  # Divider row

        f.write("\n### WLT-Specific ROC Curve Visualization\n\n")
        f.write(
            "![Baseline Architectures Bounded ROC Curve (WLT-Specific)](../plots/binary_roc_baseline_wlt.png)\n\n"
        )

    print(f"Dual-table report successfully compiled and saved to: {report_md}")


if __name__ == "__main__":
    generate_image_level_report()
