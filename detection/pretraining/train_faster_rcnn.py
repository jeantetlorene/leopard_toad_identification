import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import copy

from torchvision.ops import box_iou


class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, dataset_dir, split="train", img_size=640, augment=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.img_dir = os.path.join(dataset_dir, "images", split)
        self.lbl_dir = os.path.join(dataset_dir, "labels", split)

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_files = [
            f for f in self.img_files if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        # Normalized center to absolute pixel coordinates
                        xmin = (xc - w / 2) * self.img_size
                        ymin = (yc - h / 2) * self.img_size
                        xmax = (xc + w / 2) * self.img_size
                        ymax = (yc + h / 2) * self.img_size
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id + 1)

        if self.augment:
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_boxes = []
                for b in boxes:
                    # x_new = width - x_old
                    new_boxes.append(
                        [self.img_size - b[2], b[1], self.img_size - b[0], b[3]]
                    )
                boxes = new_boxes

            import PIL.ImageEnhance as ImageEnhance

            if np.random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(np.random.uniform(0.8, 1.2))

            # Slight Rotation (+/- 5 degrees) - Safe even without box updates for very small angles
            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                img = img.rotate(angle, resample=Image.BILINEAR)

        # 2. Resize and Format
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            final_boxes = []
            final_labels = []
            for i, b in enumerate(boxes):
                x1 = max(0.0, min(b[0], float(self.img_size)))
                y1 = max(0.0, min(b[1], float(self.img_size)))
                x2 = max(0.0, min(b[2], float(self.img_size)))
                y2 = max(0.0, min(b[3], float(self.img_size)))
                if (x2 > x1 + 1) and (y2 > y1 + 1):  # Filter boxes smaller than 1px
                    final_boxes.append([x1, y1, x2, y2])
                    final_labels.append(labels[i])

            if not final_boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
                labels = torch.as_tensor(final_labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# --- 2. Model Initialization ---
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


# --- 3. Evaluation ---
def calculate_pr_and_cm(
    model, data_loader, device, class_names, iou_threshold=0.5, conf_threshold=0.5
):
    """
    Evaluates the model and computes proper PR curve metrics using IoU matching
    and confusion matrix following the robust evaluate_faster_rcnn.py logic.
    """
    model.eval()
    num_classes = len(class_names)
    pr_data = {
        c: {"scores": [], "matches": [], "num_gt": 0} for c in range(1, num_classes + 1)
    }
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

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()
                    sorted_idx = torch.argsort(pred_scores, descending=True)

                    for i in sorted_idx:
                        p_label = pred_labels[i].item()
                        p_score = pred_scores[i].item()
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
                            pr_data[p_label]["matches"].append(1)
                        else:
                            pr_data[p_label]["scores"].append(p_score)
                            pr_data[p_label]["matches"].append(0)

                elif len(pred_boxes) > 0:
                    for lbl, score in zip(pred_labels, pred_scores):
                        pr_data[lbl.item()]["scores"].append(score.item())
                        pr_data[lbl.item()]["matches"].append(0)

                # Confusion Matrix logic
                conf_mask = pred_scores >= conf_threshold
                conf_bboxes = pred_boxes[conf_mask]
                conf_labels = pred_labels[conf_mask]

                if len(gt_boxes) == 0:
                    for l in conf_labels:
                        cm_y_true.append(0)
                        cm_y_pred.append(l.item())
                    continue
                if len(conf_bboxes) == 0:
                    for l in gt_labels:
                        cm_y_true.append(l.item())
                        cm_y_pred.append(0)
                    continue

                ious_cm = box_iou(gt_boxes, conf_bboxes)
                matched_preds = set()
                for i in range(len(gt_boxes)):
                    idx = (ious_cm[i] >= iou_threshold).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        best_match = idx[torch.argmax(ious_cm[i][idx])].item()
                        if best_match not in matched_preds:
                            matched_preds.add(best_match)
                            cm_y_true.append(gt_labels[i].item())
                            cm_y_pred.append(conf_labels[best_match].item())
                        else:
                            cm_y_true.append(gt_labels[i].item())
                            cm_y_pred.append(0)
                    else:
                        cm_y_true.append(gt_labels[i].item())
                        cm_y_pred.append(0)
                for j in range(len(conf_bboxes)):
                    if j not in matched_preds:
                        cm_y_true.append(0)
                        cm_y_pred.append(conf_labels[j].item())

    return pr_data, cm_y_true, cm_y_pred


def generate_visual_results(pr_data, cm_y_true, cm_y_pred, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("default")

    # 1. PR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    px = np.linspace(0, 1, 1000)
    all_py = []
    ap_list = []

    for c, data in pr_data.items():
        cls_name = class_names[c - 1]
        scores, matches, num_gt = data["scores"], data["matches"], data["num_gt"]
        if num_gt == 0:
            continue
        if len(scores) == 0:
            ax.plot([0, 1], [0, 0], label=f"{cls_name} 0.000")
            all_py.append(np.zeros(1000))
            ap_list.append(0.0)
            continue

        indices = np.argsort(scores)[::-1]
        matches = np.array(matches)[indices]
        tp_cumsum = np.cumsum(matches)
        fp_cumsum = np.cumsum(1 - matches)
        rec = tp_cumsum / num_gt
        prec = tp_cumsum / (tp_cumsum + fp_cumsum)
        rec = np.concatenate(([0.0], rec, [1.0]))
        prec = np.concatenate(([1.0], prec, [0.0]))
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        ap = np.sum((rec[1:] - rec[:-1]) * prec[1:])
        ap_list.append(ap)
        py = np.interp(px, rec, prec)
        all_py.append(py)
        ax.plot(rec, prec, label=f"{cls_name} {ap:.3f}")

    if all_py:
        mean_ap = np.mean(ap_list)
        ax.plot(
            px,
            np.mean(all_py, axis=0),
            label=f"all classes {mean_ap:.3f} mAP@0.5",
            color="blue",
            linewidth=3,
        )

    ax.set_title("Precision-Recall Curve (Fixed)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(False)
    plt.savefig(os.path.join(save_dir, "PR_curve_fixed.png"), bbox_inches="tight")
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
    plt.title("Confusion Matrix (Fixed)")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.grid(False)
    plt.savefig(
        os.path.join(save_dir, "confusion_matrix_fixed.png"), bbox_inches="tight"
    )
    plt.close()


# --- 4. Helper Classes ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# --- 5. Main Script ---
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset_path = "/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset"
    class_names = ["Frog", "Mouse", "Snail"]

    # YOLO style run directory organization
    project_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn"
    run_name = "train_resnet50"
    run_dir = os.path.join(project_dir, run_name)
    weights_dir = os.path.join(run_dir, "weights")
    results_dir = os.path.join(run_dir, "results")

    for d in [weights_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # Enable manual augmentations for the training set
    train_loader = DataLoader(
        YoloToFasterRCNNDataset(dataset_path, "train", augment=True),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        YoloToFasterRCNNDataset(dataset_path, "val"),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        YoloToFasterRCNNDataset(dataset_path, "test"),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )
    num_epochs = 100
    model = get_model(len(class_names)).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.0001, weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    early_stopping = EarlyStopping(patience=10)

    history = []
    best_loss = float("inf")

    print(f"Fine-tuning Faster R-CNN. Results will be saved to: {run_dir}")
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        train_comps = {
            "loss_classifier": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            for k in train_comps.keys():
                if k in loss_dict:
                    train_comps[k] += loss_dict[k].item()

            pbar.set_postfix(loss=losses.item())

        # --- Validation ---
        model.train()
        val_loss = 0
        val_comps = {
            "loss_classifier": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
        }

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()
                for k in val_comps.keys():
                    if k in loss_dict:
                        val_comps[k] += loss_dict[k].item()

                pbar_val.set_postfix(val_loss=losses.item())

        # Step the scheduler
        lr_scheduler.step()

        # Average losses
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}"
        )

        epoch_data = {
            "epoch": epoch + 1,
            "train_total_loss": avg_train,
            "train_loss_classifier": train_comps["loss_classifier"] / len(train_loader),
            "train_loss_box_reg": train_comps["loss_box_reg"] / len(train_loader),
            "train_loss_objectness": train_comps["loss_objectness"] / len(train_loader),
            "train_loss_rpn_box_reg": train_comps["loss_rpn_box_reg"]
            / len(train_loader),
            "val_total_loss": avg_val,
            "val_loss_classifier": val_comps["loss_classifier"] / len(val_loader),
            "val_loss_box_reg": val_comps["loss_box_reg"] / len(val_loader),
            "val_loss_objectness": val_comps["loss_objectness"] / len(val_loader),
            "val_loss_rpn_box_reg": val_comps["loss_rpn_box_reg"] / len(val_loader),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_data)

        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            print(f"New best model saved at epoch {epoch + 1}")

        early_stopping(avg_val)
        if early_stopping.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "results.csv"), index=False)

    print("Running final evaluation on Test set with IoU matching...")
    model.load_state_dict(torch.load(os.path.join(weights_dir, "best.pt")))
    pr_data, cm_y_true, cm_y_pred = calculate_pr_and_cm(
        model, test_loader, device, class_names
    )
    generate_visual_results(pr_data, cm_y_true, cm_y_pred, class_names, run_dir)
    print(f"Training complete. Best model and fixed plots saved in: {run_dir}")


if __name__ == "__main__":
    main()
