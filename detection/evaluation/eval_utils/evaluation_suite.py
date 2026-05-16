import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.metrics import (
    calculate_detection_metrics,
    calculate_image_level_metrics,
)
from eval_utils.config import (
    RESULTS_DIR,
    FILES_DIR,
    PLOTS_DIR,
    CLASSES,
    CONF_THRESHOLDS,
)


def get_model_info(folder_name):
    """Extract model type and processing from folder name."""
    parts = folder_name.split("_")
    processing = parts[-1]
    model_type = "_".join(parts[:-1])
    return model_type, processing


def get_eval_info(filename):
    """Extract cycle, variant, and dataset from filename."""
    parts = filename.replace("_raw.json", "").split("_")
    cycle = int(parts[1])
    variant = parts[2]
    dataset = "_".join(parts[3:])
    return cycle, variant, dataset


def save_plot_data(model, processing, variant, dataset, per_class_results, plot_data):
    for cls_id, res in per_class_results.items():
        if res["fpr"] is not None:
            plot_data.append(
                {
                    "model": model,
                    "processing": processing,
                    "variant": variant,
                    "dataset": dataset,
                    "class_id": cls_id,
                    "class_name": CLASSES[cls_id],
                    "fpr": res["fpr"],
                    "tpr": res["tpr"],
                    "auc": res["auc"],
                }
            )


def generate_roc_plots(plot_data):
    df_plot = pd.DataFrame(plot_data)
    if df_plot.empty:
        return

    for dataset in df_plot["dataset"].unique():
        for cls_id, cls_name in CLASSES.items():
            subset = df_plot[
                (df_plot["dataset"] == dataset) & (df_plot["class_id"] == cls_id)
            ]
            if subset.empty:
                continue

            plt.figure(figsize=(10, 8))

            for _, row in subset.iterrows():
                label = f"{row['model'].upper()} {row['variant'].capitalize()} ({row['processing']}) - AUC: {row['auc']:.4f}"
                plt.plot(row["fpr"], row["tpr"], label=label, linewidth=2)

            plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Recall)")
            plt.title(f"ROC Curves - {cls_name} (Cycle 4, {dataset.capitalize()} Set)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.gca().set_aspect("equal")

            plot_filename = f"final_roc_{cls_name.lower()}_{dataset}.png"
            plot_path = os.path.join(PLOTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"ROC plot saved to {plot_path}")


def run_evaluation_suite(
    target_models=None, target_processing=None, target_cycles=None, target_variants=None
):
    all_metrics = []
    all_binary_sweep = []
    plot_data = []

    # Get all model folders
    folders = sorted(
        [
            f
            for f in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, f))
        ]
    )

    for model_folder in folders:
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        filenames = sorted(
            [f for f in os.listdir(folder_path) if f.endswith("_raw.json")],
            key=lambda x: (0 if "_val_raw.json" in x else 1, x),
        )
        if not filenames:
            continue

        model_type, processing = get_model_info(model_folder)
        if target_models and model_type not in target_models:
            continue
        if target_processing and processing not in target_processing:
            continue

        print(f"\n>>> Evaluating {model_type} ({processing})...")

        for filename in tqdm(filenames, desc=f"Models in {model_folder}"):
            cycle, variant, dataset = get_eval_info(filename)
            if target_cycles and cycle not in target_cycles:
                continue
            if target_variants and variant not in target_variants:
                continue

            with open(os.path.join(folder_path, filename), "r") as f:
                results = json.load(f)

            is_full_seq = "full_seq" in dataset

            # Refresh ground truth from clean data
            from eval_utils.data_utils import get_clean_ground_truth

            refreshed_results = []
            for res in results:
                is_positive, gt_boxes, split = get_clean_ground_truth(res["path"])
                if split is not None:
                    # Found in clean mapping, update GT
                    res["is_positive"] = is_positive
                    res["gt_boxes"] = gt_boxes
                    refreshed_results.append(res)
                elif is_full_seq:
                    # Not in clean mapping but we are in full sequence mode
                    # Treat as negative (empty background)
                    res["is_positive"] = False
                    res["gt_boxes"] = []
                    refreshed_results.append(res)

            if not refreshed_results:
                print(
                    f"      Warning: No images from {filename} found in clean mapping. Skipping."
                )
                continue

            results = refreshed_results

            if not is_full_seq:
                # 1. Detection-level Metrics (mAP and BB-level optimal thresholds)
                det_metrics = calculate_detection_metrics(results)
                mAP = det_metrics["mAP"]
                class_aps = det_metrics["class_aps"]
                class_optimal = det_metrics.get("class_optimal", {})
                class_curves = det_metrics.get("class_curves", {})
            else:
                mAP = np.nan
                class_aps = {}
                class_optimal = {}
                class_curves = {}

            # 2. Image-level Binary Metrics (Any Animal vs None)
            binary_gt = np.array([res["is_positive"] for res in results])
            binary_scores = np.array(
                [
                    max([p["conf"] for p in res["predictions"]] + [0.0])
                    for res in results
                ]
            )

            if is_full_seq:
                # Full Sweep for Binary
                binary_sweep_data = calculate_image_level_metrics(
                    results, CONF_THRESHOLDS
                )
                for entry in binary_sweep_data:
                    entry.update(
                        {
                            "model": model_type,
                            "processing": processing,
                            "cycle": cycle,
                            "variant": variant,
                            "dataset": dataset,
                        }
                    )
                    all_binary_sweep.append(entry)

                # ROC-AUC using all unique thresholds
                binary_fpr, binary_tpr, _ = roc_curve(binary_gt, binary_scores)
                binary_auc = auc(binary_fpr, binary_tpr)

                # Metrics at 0.1 for summary (legacy)
                binary_preds_01 = binary_scores >= 0.1
                binary_recall_01 = recall_score(
                    binary_gt, binary_preds_01, zero_division=0
                )
                binary_precision_01 = precision_score(
                    binary_gt, binary_preds_01, zero_division=0
                )
                binary_f1_01 = f1_score(binary_gt, binary_preds_01, zero_division=0)
            else:
                binary_auc = np.nan
                binary_recall_01 = np.nan
                binary_precision_01 = np.nan
                binary_f1_01 = np.nan

            # 3. Per-Class Image-level Metrics
            per_class_results = {}
            for cls_id, cls_name in CLASSES.items():
                cls_gt = np.array(
                    [
                        any(gt["cls"] == cls_id for gt in res["gt_boxes"])
                        for res in results
                    ]
                )
                cls_scores = np.array(
                    [
                        max(
                            [
                                p["conf"]
                                for p in res["predictions"]
                                if p["cls"] == cls_id
                            ]
                            + [0.0]
                        )
                        for res in results
                    ]
                )

                best_thresh = np.nan
                opt_recall, opt_precision, opt_f1, opt_specificity = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                def_recall, def_precision, def_f1 = np.nan, np.nan, np.nan
                cls_auc, fpr, tpr = np.nan, None, None

                if not is_full_seq:
                    # Retrieve bounding-box level optimal metrics calculated analytically
                    opt = class_optimal.get(cls_id, {})
                    best_thresh = opt.get("best_thresh", np.nan)
                    opt_recall = opt.get("best_recall", np.nan)
                    opt_precision = opt.get("best_precision", np.nan)
                    opt_f1 = (
                        2 * (opt_precision * opt_recall) / (opt_precision + opt_recall)
                        if (opt_precision + opt_recall) > 0
                        else 0
                    )

                    # Calculate image-level specificity at this optimal bounding-box threshold
                    if pd.notna(best_thresh):
                        opt_preds = cls_scores >= best_thresh
                        tn = np.sum(~opt_preds & ~cls_gt)
                        fp_img = np.sum(opt_preds & ~cls_gt)
                        opt_specificity = (
                            tn / (tn + fp_img) if (tn + fp_img) > 0 else 0.0
                        )

                    # We still compute default (0.5) bounding box metrics by extracting from the curve
                    # For simplicity, we just save the optimal ones. If def is needed, we could extract it from the curve too.
                    # Setting def_f1 etc to np.nan as optimal is the new standard
                    def_recall, def_precision, def_f1 = np.nan, np.nan, np.nan

                if is_full_seq:
                    # ROC-AUC using all unique thresholds
                    if len(np.unique(cls_gt)) > 1:
                        fpr, tpr, _ = roc_curve(cls_gt, cls_scores)
                        cls_auc = auc(fpr, tpr)

                # Extract absolute maximum recall (AR) for this class
                cls_curve = class_curves.get(cls_id, {})
                mrec = cls_curve.get("recall", [])
                cls_ar = mrec[-2] if len(mrec) >= 2 else 0.0

                per_class_results[cls_id] = {
                    "auc": cls_auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "ap": class_aps.get(cls_id, np.nan) if not is_full_seq else np.nan,
                    "ar": cls_ar,  # Absolute Max Recall
                    "f1_default": def_f1,
                    "recall_default": def_recall,
                    "precision_default": def_precision,
                    "f1_optimal": opt_f1,
                    "recall_optimal": opt_recall,
                    "precision_optimal": opt_precision,
                    "specificity_optimal": opt_specificity,
                    "optimal_threshold": best_thresh,
                }

            # Calculate mean Average Recall (mAR) exactly from the PR curves
            mar = np.nan
            if not is_full_seq and class_curves:
                max_recalls = []
                for c, curve in class_curves.items():
                    mrec = curve.get("recall", [])
                    if len(mrec) >= 2:
                        max_recalls.append(
                            mrec[-2]
                        )  # mrec ends with [1.0] from VOC interpolation
                    else:
                        max_recalls.append(0.0)
                if max_recalls:
                    mar = float(np.mean(max_recalls))

            # Record metrics
            row = {
                "model": model_type,
                "processing": processing,
                "cycle": cycle,
                "variant": variant,
                "dataset": dataset,
                "mAP": mAP,
                "mAR": mar,
                "binary_auc": binary_auc,
                "binary_recall_0.1": binary_recall_01,
                "binary_precision_0.1": binary_precision_01,
                "binary_f1_0.1": binary_f1_01,
            }
            # Add per-class columns
            for cls_id, cls_name in CLASSES.items():
                res = per_class_results[cls_id]
                row[f"{cls_name}_auc"] = res["auc"]
                row[f"{cls_name}_ap"] = res["ap"]
                row[f"{cls_name}_ar"] = res["ar"]
                row[f"{cls_name}_f1_default"] = res["f1_default"]
                row[f"{cls_name}_recall_default"] = res["recall_default"]
                row[f"{cls_name}_precision_default"] = res["precision_default"]
                row[f"{cls_name}_f1_optimal"] = res["f1_optimal"]
                row[f"{cls_name}_recall_optimal"] = res["recall_optimal"]
                row[f"{cls_name}_precision_optimal"] = res["precision_optimal"]
                row[f"{cls_name}_specificity_optimal"] = res.get(
                    "specificity_optimal", np.nan
                )
                row[f"{cls_name}_optimal_threshold"] = res["optimal_threshold"]

            all_metrics.append(row)

            # Store FPR/TPR for plotting (only for Cycle 4)
            if cycle == 4 and is_full_seq:
                save_plot_data(
                    model_type,
                    processing,
                    variant,
                    dataset,
                    per_class_results,
                    plot_data,
                )

    # Save Summaries
    os.makedirs(FILES_DIR, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(FILES_DIR, "unified_model_evaluation.csv"), index=False
    )
    pd.DataFrame(all_binary_sweep).to_csv(
        os.path.join(FILES_DIR, "binary_threshold_sweep.csv"), index=False
    )

    print(f"\nEvaluation files saved to {FILES_DIR}")

    # Generate Plots
    print("\n>>> Generating ROC Plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    generate_roc_plots(plot_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Evaluation Suite")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Target models (e.g., yolo rtdetr faster_rcnn)",
    )
    parser.add_argument(
        "--processing",
        nargs="+",
        default=None,
        help="Target processing (e.g., plain clahe)",
    )
    parser.add_argument(
        "--cycles", type=int, nargs="+", default=None, help="Target cycles (e.g., 0 4)"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Target variants (e.g., pretrained scratch)",
    )
    args = parser.parse_args()

    run_evaluation_suite(
        target_models=args.models,
        target_processing=args.processing,
        target_cycles=args.cycles,
        target_variants=args.variants,
    )
