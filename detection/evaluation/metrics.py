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


# Note: mAP calculation is complex, usually we'd use a library.
# For now, we prioritize image-level metrics as requested in EVALUATION.md.
