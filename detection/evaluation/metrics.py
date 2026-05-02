import numpy as np


def calculate_image_level_metrics(results, thresholds):
    """Calculate Recall and Specificity for binary classification (image-level)."""
    metrics = []
    for thresh in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for res in results:
            has_detection = any(p["conf"] >= thresh for p in res["predictions"])
            if res["is_positive"]:
                if has_detection:
                    tp += 1
                else:
                    fn += 1
            else:
                if has_detection:
                    fp += 1
                else:
                    tn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.append(
            {
                "threshold": thresh,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "recall": recall,
                "specificity": specificity,
            }
        )
    return metrics


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
    """Calculate mAP@0.5 across the dataset."""
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
    aps = []

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

        for i, pred in enumerate(cls_preds):
            # Find GTs in same image
            img_gts = [g for g in cls_gts if g["img_id"] == pred["img_id"]]
            best_iou = -1
            best_gt = None

            for gt in img_gts:
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= iou_threshold and not best_gt["matched"]:
                tp[i] = 1
                best_gt["matched"] = True
            else:
                fp[i] = 1

        # Reset matched for next eval (if any) - not strictly needed here as we filter cls_gts
        for g in cls_gts:
            g["matched"] = False

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)

        # VOC all-points interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for j in range(mpre.size - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        aps.append(ap)

    return np.mean(aps) if aps else 0.0
