import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from train_faster_rcnn import get_model, YoloToFasterRCNNDataset, collate_fn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def calculate_pr_and_cm(
    model, data_loader, device, class_names, iou_threshold=0.5, conf_threshold=0.5
):
    """
    Evaluates the model and computes proper PR curve metrics using IoU matching
    and confusion matrix properly associating predictions with ground truths.
    """
    model.eval()
    num_classes = len(class_names)

    # Store data for PR curve calculation per class
    pr_data = {
        c: {"scores": [], "matches": [], "num_gt": 0} for c in range(1, num_classes + 1)
    }

    # Target and predicted labels for confusion matrix
    cm_y_true = []
    cm_y_pred = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                pred_boxes = out["boxes"].cpu()
                pred_labels = out["labels"].cpu()
                pred_scores = out["scores"].cpu()

                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()

                for c in range(1, num_classes + 1):
                    pr_data[c]["num_gt"] += (gt_labels == c).sum().item()

                if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                    continue

                # -----------------------
                # 1. Matching for PR Curve
                # -----------------------
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
                    matched_gt = set()

                    # Sort predictions by score descending
                    sorted_idx = torch.argsort(pred_scores, descending=True)

                    for i in sorted_idx:
                        p_label = pred_labels[i].item()
                        p_score = pred_scores[i].item()

                        # Find best matching ground truth of the SAME class
                        best_iou = 0
                        best_gt_idx = -1
                        for j in range(len(gt_boxes)):
                            if j not in matched_gt and gt_labels[j].item() == p_label:
                                iou = ious[i, j].item()
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = j

                        if best_iou >= iou_threshold:
                            matched_gt.add(best_gt_idx)
                            pr_data[p_label]["scores"].append(p_score)
                            pr_data[p_label]["matches"].append(1)  # True Positive
                        else:
                            pr_data[p_label]["scores"].append(p_score)
                            pr_data[p_label]["matches"].append(0)  # False Positive

                elif len(pred_boxes) > 0:
                    # All predictions are false positives because no ground truth
                    for lbl, score in zip(pred_labels, pred_scores):
                        pr_data[lbl.item()]["scores"].append(score.item())
                        pr_data[lbl.item()]["matches"].append(0)

                # -----------------------------
                # 2. Matching for Confusion Matrix
                # -----------------------------
                # Keep only predictions over the confidence threshold
                conf_mask = pred_scores >= conf_threshold
                conf_bboxes = pred_boxes[conf_mask]
                conf_labels = pred_labels[conf_mask]

                if len(gt_boxes) == 0:
                    for l in conf_labels:
                        cm_y_true.append(0)  # true was background
                        cm_y_pred.append(l.item())
                    continue

                if len(conf_bboxes) == 0:
                    for l in gt_labels:
                        cm_y_true.append(l.item())
                        cm_y_pred.append(0)  # predicted background (False Negative)
                    continue

                ious = box_iou(gt_boxes, conf_bboxes)  # [num_gts, num_preds]
                matched_preds = set()

                # Match each Ground Truth to the best predicted generic box
                for i in range(len(gt_boxes)):
                    best_iou = 0
                    best_match = -1
                    for j in range(len(conf_bboxes)):
                        if j not in matched_preds:
                            iou = ious[i, j].item()
                            if iou > best_iou:
                                best_iou = iou
                                best_match = j

                    if best_iou >= iou_threshold:
                        matched_preds.add(best_match)
                        cm_y_true.append(gt_labels[i].item())
                        cm_y_pred.append(conf_labels[best_match].item())
                    else:
                        cm_y_true.append(gt_labels[i].item())
                        cm_y_pred.append(0)  # FN

                # Any unmatched confident prediction is a false positive
                for j in range(len(conf_bboxes)):
                    if j not in matched_preds:
                        cm_y_true.append(0)
                        cm_y_pred.append(conf_labels[j].item())

    return pr_data, cm_y_true, cm_y_pred


def plot_results(pr_data, cm_y_true, cm_y_pred, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("default")

    # 1. PR Curve
    fig, ax = plt.subplots(figsize=(8, 6))

    px = np.linspace(0, 1, 1000)
    all_py = []
    ap_list = []

    for c, data in pr_data.items():
        cls_name = class_names[c - 1]
        scores = data["scores"]
        matches = data["matches"]
        num_gt = data["num_gt"]

        if num_gt == 0:
            continue

        if len(scores) == 0:
            ax.plot([0, 1], [0, 0], label=f"{cls_name} 0.000", linewidth=1.1)
            all_py.append(np.zeros(1000))
            ap_list.append(0.0)
            continue

        # Sort scores and matching descending
        sorted_indices = np.argsort(scores)[::-1]
        matches = np.array(matches)[sorted_indices]

        tp_cumsum = np.cumsum(matches)
        fp_cumsum = np.cumsum(1 - matches)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Add start point (0, 1) and extend envelope to (1, 0)
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))

        # Smooth PR curve (envelope) according to COCO/PASCAL VOC 11-point logic
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        ap_list.append(ap)

        # Interpolate precision at 1000 recall points for mean AP line
        py = np.interp(px, recalls, precisions)
        all_py.append(py)

        ax.plot(recalls, precisions, label=f"{cls_name} {ap:.3f}", linewidth=1.1)

    if len(all_py) > 0:
        mean_py = np.mean(all_py, axis=0)
        mean_ap = np.mean(ap_list)
        ax.plot(
            px,
            mean_py,
            label=f"all classes {mean_ap:.3f} mAP@0.5",
            color="blue",
            linewidth=3,
        )

    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(False)

    # Place legend outside the plot box, matching YOLO style
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    plt.savefig(
        os.path.join(save_dir, "PR_curve_fixed.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(cm_y_true, cm_y_pred, labels=range(len(class_names) + 1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["background"] + class_names,
        yticklabels=["background"] + class_names,
    )
    plt.title("Confusion Matrix (Conf=0.5, IoU=0.5)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.grid(False)
    plt.savefig(
        os.path.join(save_dir, "confusion_matrix_fixed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset"
    class_names = ["Frog", "Mouse", "Snail"]

    weights_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn/train_resnet50/weights/best.pt"
    save_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn/train_resnet50"

    print("Loading test and validation datasets...")
    test_loader = DataLoader(
        YoloToFasterRCNNDataset(dataset_path, "test"),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        YoloToFasterRCNNDataset(dataset_path, "val"),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = get_model(len(class_names)).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Weights not found at {weights_path}")
        return

    print("\n--- Evaluating on Test Set ---")
    pr_data, cm_y_true, cm_y_pred = calculate_pr_and_cm(
        model, test_loader, device, class_names
    )
    plot_results(pr_data, cm_y_true, cm_y_pred, class_names, save_dir)
    print(f"Evaluation complete. Plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
