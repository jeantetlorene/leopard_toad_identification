import os
import glob
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.ops.boxes as box_ops
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc

class YoloSubsetDataset(Dataset):
    def __init__(self, dataset_dir):
        self.image_paths = glob.glob(os.path.join(dataset_dir, "images", "*.jpg")) + \
                           glob.glob(os.path.join(dataset_dir, "images", "*.JPG")) + \
                           glob.glob(os.path.join(dataset_dir, "images", "*.png"))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Apply augmentation (ColorJitter preserves bounding box coordinates)
        augment = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        img = augment(img)
        
        w, h = img.size
        
        label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        xmin = (cx - bw/2) * w
                        ymin = (cy - bh/2) * h
                        xmax = (cx + bw/2) * w
                        ymax = (cy + bh/2) * h
                        
                        # torchvision expects xmax > xmin and ymax > ymin
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            # torchvision labels are 1-indexed (0 is background)
                            labels.append(class_id + 1)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        img_tensor = F.to_tensor(img)
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_and_plot_metrics(model, val_loader, device, save_dir, iou_threshold=0.5):
    print("Evaluating model to generate plots...")
    model.eval()
    
    classes = [1, 2, 3]
    class_names = ['Background', 'Other Amphibian', 'Small Mammal', 'WLT']
    
    gt_counts = {c: 0 for c in classes}
    predictions_data = {c:[] for c in classes}
    
    all_gt_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                gt_boxes = target['boxes'].to(device)
                gt_labels = target['labels'].to(device)
                
                pred_boxes = output['boxes'].to(device)
                pred_labels = output['labels'].to(device)
                pred_scores = output['scores'].to(device)
                
                keep_idx = pred_scores > 0.25
                filt_pred_boxes = pred_boxes[keep_idx]
                filt_pred_labels = pred_labels[keep_idx]
                
                for c in classes:
                    gt_counts[c] += (gt_labels == c).sum().item()
                    c_gt_boxes = gt_boxes[gt_labels == c]
                    c_pred_mask = pred_labels == c
                    c_pred_boxes = pred_boxes[c_pred_mask]
                    c_pred_scores = pred_scores[c_pred_mask]
                    
                    sort_idx = torch.argsort(c_pred_scores, descending=True)
                    c_pred_boxes = c_pred_boxes[sort_idx]
                    c_pred_scores = c_pred_scores[sort_idx]
                    
                    matched_gt = set()
                    for i in range(len(c_pred_boxes)):
                        score = c_pred_scores[i].item()
                        is_tp = 0
                        if len(c_gt_boxes) > 0:
                            ious = box_ops.box_iou(c_pred_boxes[i].unsqueeze(0), c_gt_boxes)[0]
                            max_iou, max_idx = ious.max(0)
                            max_idx = max_idx.item()
                            if max_iou >= iou_threshold and max_idx not in matched_gt:
                                is_tp = 1
                                matched_gt.add(max_idx)
                        predictions_data[c].append((score, is_tp))
                        
                gt_claimed = [False] * len(gt_boxes)
                for i, pbox in enumerate(filt_pred_boxes):
                    plabel = filt_pred_labels[i].item()
                    best_iou = 0
                    best_gt_idx = -1
                    for j, gbox in enumerate(gt_boxes):
                        if not gt_claimed[j]:
                            iou = box_ops.box_iou(pbox.unsqueeze(0), gbox.unsqueeze(0))[0][0].item()
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = j
                    if best_iou >= iou_threshold:
                        glabel = gt_labels[best_gt_idx].item()
                        all_gt_labels.append(glabel)
                        all_pred_labels.append(plabel)
                        gt_claimed[best_gt_idx] = True
                    else:
                        all_gt_labels.append(0)
                        all_pred_labels.append(plabel)
                        
                for j, claimed in enumerate(gt_claimed):
                    if not claimed:
                        glabel = gt_labels[j].item()
                        all_gt_labels.append(glabel)
                        all_pred_labels.append(0)
                        
    # 1. PR Curve Plot
    fig, ax_pr = plt.subplots(figsize=(8, 6))
    colors = {1: 'blue', 2: 'orange', 3: 'green'}
    names = {1: 'Other_Amphibian', 2: 'Small_Mammal', 3: 'WLT'}
    mean_ap = 0.0
    f1_curves = {}
    
    for c in classes:
        if gt_counts[c] == 0: continue
        data = sorted(predictions_data[c], key=lambda x: x[0], reverse=True)
        scores = [x[0] for x in data]
        tps = [x[1] for x in data]
        fps = [1 - x for x in tps]
        
        cum_tps = np.cumsum(tps)
        cum_fps = np.cumsum(fps)
        
        recalls = cum_tps / gt_counts[c]
        precisions = cum_tps / (cum_tps + cum_fps + 1e-16)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
        f1_curves[c] = (scores, f1_scores)
        
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        
        ap = auc(recalls, precisions)
        mean_ap += ap
        
        ax_pr.plot(recalls, precisions, color=colors[c], lw=2, label=f'{names[c]} {ap:.3f}')
        
    ax_pr.set_title(f'PR Curve (mAP@0.5 = {mean_ap/len(classes):.3f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True)
    plt.savefig(os.path.join(save_dir, 'PR_curve.png'))
    plt.close()
    
    # 2. F1 Curve Plot
    fig, ax_f1 = plt.subplots(figsize=(8, 6))
    for c in classes:
        if c in f1_curves and len(f1_curves[c][0]) > 0:
            ax_f1.plot(f1_curves[c][0], f1_curves[c][1], color=colors[c], lw=2, label=f'{names[c]}')
    ax_f1.set_title('F1-Confidence Curve')
    ax_f1.set_xlabel('Confidence')
    ax_f1.set_ylabel('F1')
    ax_f1.legend(loc="lower left")
    ax_f1.grid(True)
    plt.savefig(os.path.join(save_dir, 'F1_curve.png'))
    plt.close()
    
    # 3. Confusion Matrix
    if len(all_gt_labels) > 0:
        cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=[0, 1, 2, 3])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

def main():
    dataset_dir = "/home/Joshua/Downloads/leopard_toad_identification/dataset/detect_2/dataset_clahe"
    classes = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
    num_classes = len(classes) + 1 # +1 for background
    
    dataset = YoloSubsetDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    print("Initializing Faster R-CNN model...")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    
    best_weights = "/home/Joshua/Downloads/OIDv4_ToolKit/runs/faster_rcnn/train_resnet50/weights/best.pt"
    if os.path.exists(best_weights):
        print(f"Loading best weights from {best_weights}...")
        state_dict = torch.load(best_weights, map_location="cpu")
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: {best_weights} not found. Starting from scratch.")
        
    # FREEZE BACKBONE for partial fine-tuning
    print("Freezing backbone layers...")
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Only optimize parameters that require gradients (the top layers / heads)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    save_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/faster_rcnn_finetune"
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)
    save_path = os.path.join(save_dir, "weights", "best_partial.pt")
    
    num_epochs = 10
    train_losses = []
    
    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            valid_batches += 1
            
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float("inf")
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved partially fine-tuned model to {save_path}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss', color='blue')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'results.png'))
    plt.close()
    
    # Run evaluation and generate PR, F1, and Confusion Matrix plots
    evaluate_and_plot_metrics(model, dataloader, device, save_dir, iou_threshold=0.5)

if __name__ == "__main__":
    main()
