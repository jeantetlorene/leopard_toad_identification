import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, RESULTS_DIR, CLASSES
from eval_utils.data_utils import refresh_results
from eval_utils.metrics import calculate_detection_metrics

UNIFIED_CSV = os.path.join(FILES_DIR, "active_learning_unified_evaluation.csv")


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


def compute_precision_recall_at_threshold(raw_results, cls_id, thresh):
    tp_count = 0
    fp_count = 0
    n_gt = 0

    for res in raw_results:
        preds = [p for p in res.get("predictions", []) if p["cls"] == cls_id]
        gts = [g for g in res.get("gt_boxes", []) if g["cls"] == cls_id]
        n_gt += len(gts)

        preds.sort(key=lambda x: x["conf"], reverse=True)
        gt_matched = [False] * len(gts)

        for p in preds:
            best_iou = -1
            best_gt_idx = -1
            for i, gt in enumerate(gts):
                if not gt_matched[i]:
                    iou = box_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            is_tp = False
            if best_iou >= 0.5 and best_gt_idx >= 0:
                is_tp = True
                gt_matched[best_gt_idx] = True

            if p["conf"] >= thresh:
                if is_tp:
                    tp_count += 1
                else:
                    fp_count += 1

    recall = tp_count / n_gt if n_gt > 0 else 0.0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    return precision, recall


def main():
    if not os.path.exists(UNIFIED_CSV):
        print(f"Error: Unified CSV not found at {UNIFIED_CSV}")
        return

    df = pd.read_csv(UNIFIED_CSV)
    print(f"Loaded {len(df)} rows from {UNIFIED_CSV}")

    # Filter for YOLO/RT-DETR models
    ultralytics_mask = df["model"].isin(["yolo", "rtdetr"])
    combos = df[ultralytics_mask][
        ["model", "processing", "cycle", "variant"]
    ].drop_duplicates()
    print(f"Found {len(combos)} unique Ultralytics model combinations to optimize.")

    updated_count = 0

    for _, combo in combos.iterrows():
        m_type = combo["model"]
        proc = combo["processing"]
        cycle = int(combo["cycle"])
        var = combo["variant"]

        root_key = f"{m_type}_{proc}"
        val_json_path = os.path.join(
            RESULTS_DIR, root_key, f"cycle_{cycle}_{var}_val_raw.json"
        )

        if not os.path.exists(val_json_path):
            print(
                f"  Warning: Validation file {val_json_path} does not exist. Skipping."
            )
            continue

        # 1. Load validation predictions and run F1 sweep to find optimal threshold
        with open(val_json_path, "r") as f:
            val_results = json.load(f)
        val_results = refresh_results(val_results, is_full_seq=False)
        det_metrics_val = calculate_detection_metrics(val_results)

        # Determine optimal validation threshold for each class
        opt_thresholds = {}
        for cls_id, cls_name in CLASSES.items():
            opt_info = det_metrics_val["class_optimal"].get(cls_id)
            if opt_info and opt_info["best_thresh"] > 0:
                opt_thresholds[cls_name] = opt_info["best_thresh"]
            else:
                opt_thresholds[cls_name] = (
                    0.25  # default fallback if no predictions or F1 is 0
                )

        print(
            f"  {m_type}_{proc} (Cycle {cycle}, {var}) optimal validation thresholds: {opt_thresholds}"
        )

        # 2. Update CSV rows for this combination (both dataset='test' and dataset='val')
        for split in ["test", "val"]:
            split_mask = (
                (df["model"] == m_type)
                & (df["processing"] == proc)
                & (df["cycle"] == cycle)
                & (df["variant"] == var)
                & (df["dataset"] == split)
            )

            if not df[split_mask].empty:
                split_json_path = os.path.join(
                    RESULTS_DIR, root_key, f"cycle_{cycle}_{var}_{split}_raw.json"
                )
                if os.path.exists(split_json_path):
                    with open(split_json_path, "r") as f:
                        split_results = json.load(f)
                    split_results = refresh_results(split_results, is_full_seq=False)

                    # Update columns for each class
                    for cls_id, cls_name in CLASSES.items():
                        thresh = opt_thresholds[cls_name]
                        precision_opt, recall_opt = (
                            compute_precision_recall_at_threshold(
                                split_results, cls_id, thresh
                            )
                        )

                        df.loc[split_mask, f"{cls_name}_optimal_threshold"] = thresh
                        df.loc[split_mask, f"{cls_name}_precision_optimal"] = (
                            precision_opt
                        )
                        df.loc[split_mask, f"{cls_name}_recall_optimal"] = recall_opt

                    updated_count += 1
                else:
                    # If raw JSON split file is not found, just assign thresholds
                    for cls_id, cls_name in CLASSES.items():
                        df.loc[split_mask, f"{cls_name}_optimal_threshold"] = (
                            opt_thresholds[cls_name]
                        )
                    updated_count += 1

    # Save back to CSV
    df.to_csv(UNIFIED_CSV, index=False)
    print(
        f"Successfully optimized and saved thresholds for {updated_count} Ultralytics dataset rows in {UNIFIED_CSV}!"
    )


if __name__ == "__main__":
    main()
