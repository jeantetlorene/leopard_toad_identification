import os
import orjson
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import calculate_detection_metrics, box_iou
from config import RESULTS_DIR, CONF_THRESHOLDS

CLASS_NAMES = {0: "Other_Amphibian", 1: "Small_Mammal", 2: "Western_Leopard_Toad"}


def calculate_per_class_image_metrics(results, thresholds):
    """
    Calculate image-level Recall and Specificity for each class independently.
    Optimized version using NumPy.
    """
    class_metrics = {}

    for cls_id in CLASS_NAMES.keys():
        is_gt_positives = []
        max_confs = []

        for res in results:
            is_gt_positives.append(any(gt["cls"] == cls_id for gt in res["gt_boxes"]))
            cls_confs = [p["conf"] for p in res["predictions"] if p["cls"] == cls_id]
            max_confs.append(max(cls_confs) if cls_confs else 0.0)

        is_gt_positives = np.array(is_gt_positives)
        max_confs = np.array(max_confs)

        n_pos = np.sum(is_gt_positives)
        n_neg = len(is_gt_positives) - n_pos

        metrics = []
        for thresh in thresholds:
            has_detection = max_confs >= thresh
            tp = np.sum(has_detection & is_gt_positives)
            fp = np.sum(has_detection & ~is_gt_positives)
            tn = n_neg - fp
            fn = n_pos - tp

            recall = tp / n_pos if n_pos > 0 else 0
            specificity = tn / n_neg if n_neg > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

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
                }
            )
        class_metrics[cls_id] = metrics

    return class_metrics


def analyze_all_results():
    all_summary_rows = []
    full_sweep_rows = []

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
            [f for f in os.listdir(folder_path) if f.endswith("_raw.json")]
        )
        if not filenames:
            continue

        print(f"\n>>> Processing {model_folder}...")

        for filename in tqdm(filenames, desc=f"Models in {model_folder}"):
            parts = filename.replace("_raw.json", "").split("_")
            cycle = int(parts[1])
            variant = parts[2]
            dataset = parts[3]

            folder_parts = model_folder.split("_")
            processing = folder_parts[-1]
            model_type = "_".join(folder_parts[:-1])

            with open(os.path.join(folder_path, filename), "rb") as f:
                # Using orjson for much faster loading of large files
                results = orjson.loads(f.read())

            det_metrics = calculate_detection_metrics(results)
            class_aps = det_metrics["class_aps"]
            image_class_metrics = calculate_per_class_image_metrics(
                results, CONF_THRESHOLDS
            )

            for cls_id, cls_name in CLASS_NAMES.items():
                ap = class_aps.get(cls_id, 0.0)
                metrics_sweep = image_class_metrics[cls_id]

                # Record all thresholds in full_sweep_rows
                for m in metrics_sweep:
                    full_sweep_rows.append(
                        {
                            "model": model_type,
                            "processing": processing,
                            "cycle": cycle,
                            "variant": variant,
                            "dataset": dataset,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "threshold": m["threshold"],
                            "recall": m["recall"],
                            "specificity": m["specificity"],
                            "precision": m["precision"],
                            "f1_score": m["f1_score"],
                            "tp": m["tp"],
                            "fp": m["fp"],
                            "tn": m["tn"],
                            "fn": m["fn"],
                        }
                    )

                idx_01 = np.argmin([abs(m["threshold"] - 0.1) for m in metrics_sweep])
                m01 = metrics_sweep[idx_01]

                max_recall = max(m["recall"] for m in metrics_sweep)
                best_thresh_for_recall = 0.0
                best_spec_at_max_recall = 0.0
                if max_recall > 0:
                    candidates = [m for m in metrics_sweep if m["recall"] >= max_recall]
                    candidates.sort(
                        key=lambda x: (x["specificity"], x["threshold"]), reverse=True
                    )
                    best_m = candidates[0]
                    best_thresh_for_recall = best_m["threshold"]
                    best_spec_at_max_recall = best_m["specificity"]

                all_summary_rows.append(
                    {
                        "model": model_type,
                        "processing": processing,
                        "cycle": cycle,
                        "variant": variant,
                        "dataset": dataset,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "AP": ap,
                        "recall_0.1": m01["recall"],
                        "specificity_0.1": m01["specificity"],
                        "precision_0.1": m01["precision"],
                        "f1_0.1": m01["f1_score"],
                        "max_recall": max_recall,
                        "best_thresh": best_thresh_for_recall,
                        "spec_at_best_thresh": best_spec_at_max_recall,
                    }
                )

    if all_summary_rows:
        # Save Summary
        summary_df = pd.DataFrame(all_summary_rows)
        csv_path = os.path.join(RESULTS_DIR, "per_class_models_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to {csv_path}")

        # Save Full Sweep
        sweep_df = pd.DataFrame(full_sweep_rows)
        sweep_path = os.path.join(RESULTS_DIR, "per_class_threshold_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Full sweep data saved to {sweep_path}")

        # Display highlights for Toad class (2)
        print("\n--- Toad Performance Highlights (Cycle 4, Test Set) ---")
        highlights = summary_df[
            (summary_df["cycle"] == 4)
            & (summary_df["dataset"] == "test")
            & (summary_df["class_id"] == 2)
        ]
        print(
            highlights[
                [
                    "model",
                    "processing",
                    "AP",
                    "recall_0.1",
                    "specificity_0.1",
                    "best_thresh",
                    "spec_at_best_thresh",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    analyze_all_results()
