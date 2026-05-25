import numpy as np


def calculate_image_level_metrics(results, thresholds):
    """Calculate Recall and Specificity for binary classification (image-level)."""
    if not results:
        return []

    max_confs = np.array(
        [max([p["conf"] for p in res["predictions"]] + [0.0]) for res in results]
    )
    is_positive = np.array([res["is_positive"] for res in results], dtype=bool)

    metrics = []
    for thresh in thresholds:
        has_detection = max_confs >= thresh
        tp = np.sum(has_detection & is_positive)
        fn = np.sum(~has_detection & is_positive)
        fp = np.sum(has_detection & ~is_positive)
        tn = np.sum(~has_detection & ~is_positive)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
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
    return metrics


from collections import defaultdict


def box_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h] in normalized coordinates."""
    # Convert to x1, y1, x2, y2
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


def calculate_detection_metrics(results, iou_threshold=0.5):
    """Calculate AP per class and mAP@0.5 across the dataset."""
    # Group results by class
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

    classes = np.unique([g["cls"] for g in all_gts])
    class_aps = {}
    class_curves = {}
    class_optimal = {}

    for c in classes:
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
            # Find GTs in same image
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

            if best_iou >= iou_threshold and best_gt is not None:
                tp[i] = 1
                best_gt["matched"] = True
            else:
                fp[i] = 1

        # Reset matched
        for g in cls_gts:
            g["matched"] = False

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        precision = (
            tp_cum / (tp_cum + fp_cum)
            if (tp_cum + fp_cum).any()
            else np.zeros_like(tp_cum)
        )

        # VOC all-points interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for j in range(mpre.size - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        class_aps[int(c)] = ap
        class_curves[int(c)] = {"recall": mrec, "precision": mpre}

        # Calculate optimal threshold based on maximum F1-Score
        confs = np.array([p["conf"] for p in cls_preds])
        best_recall = 0.0
        best_precision = 0.0
        best_thresh = 0.0

        if len(recall) > 0:
            f1_scores = np.zeros_like(recall)
            valid = (precision + recall) > 0
            f1_scores[valid] = (
                2
                * (precision[valid] * recall[valid])
                / (precision[valid] + recall[valid])
            )

            best_idx = np.argmax(f1_scores)

            best_recall = float(recall[best_idx])
            best_precision = float(precision[best_idx])
            best_thresh = float(confs[best_idx])
            if best_thresh > 0.5:
                best_thresh = best_thresh - 0.1

        class_optimal[int(c)] = {
            "best_recall": best_recall,
            "best_precision": best_precision,
            "best_thresh": best_thresh,
        }

    mAP = np.mean(list(class_aps.values())) if class_aps else 0.0
    return {
        "mAP": mAP,
        "class_aps": class_aps,
        "class_curves": class_curves,
        "class_optimal": class_optimal,
    }


def calculate_map50_95(results):
    """Calculate mAP@0.5:0.95 by averaging mAP across IoU thresholds from 0.5 to 0.95 with step 0.05."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = []
    for iou in iou_thresholds:
        det_metrics = calculate_detection_metrics(results, iou_threshold=iou)
        maps.append(det_metrics["mAP"])

    return float(np.mean(maps)) if maps else 0.0
