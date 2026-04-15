import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

# --- 1. Dataset Configuration ---
class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, dataset_dir, split='train', img_size=640):
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.img_dir = os.path.join(dataset_dir, 'images', split)
        self.lbl_dir = os.path.join(dataset_dir, 'labels', split)
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_files = [f for f in self.img_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        
        # Resize image for consistency
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

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
                        # Convert to absolute pixel coordinates (x1, y1, x2, y2)
                        xmin = (xc - w/2) * self.img_size
                        ymin = (yc - h/2) * self.img_size
                        xmax = (xc + w/2) * self.img_size
                        ymax = (yc + h/2) * self.img_size
                        # Faster R-CNN uses 1-based indexing for foreground (0 is background)
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id + 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

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

# --- 3. Evaluation & YOLO-style Plotting ---
def calculate_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i in range(len(outputs)):
                all_preds.append({k: v.cpu().numpy() for k, v in outputs[i].items()})
                all_targets.append({k: v.cpu().numpy() for k, v in targets[i].items()})
    return all_preds, all_targets

def generate_visual_results(all_preds, all_targets, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    
    # 1. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        cls_id = cls_idx + 1
        y_true, y_scores = [], []
        for p, t in zip(all_preds, all_targets):
            mask_p = p['labels'] == cls_id
            mask_t = t['labels'] == cls_id
            if mask_t.any():
                y_true.extend([1] * mask_t.sum())
                scores = p['scores'][mask_p]
                if len(scores) > 0:
                    y_scores.extend(scores[:mask_t.sum()])
                    if len(scores) < mask_t.sum(): y_true = y_true[:-(mask_t.sum() - len(scores))]
                else: y_true = y_true[:-mask_t.sum()]
        if y_scores:
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            plt.plot(rec, prec, label=f"{cls_name} {average_precision_score(y_true, y_scores):.3f}")

    plt.title("Precision-Recall Curve", fontsize=16); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.grid(False)
    plt.savefig(os.path.join(save_dir, "PR_curve.png"), dpi=300); plt.close()

    # 2. Confusion Matrix
    y_true_all, y_pred_all = [], []
    for p, t in zip(all_preds, all_targets):
        if len(t['labels']) > 0:
            y_true_all.extend(t['labels'])
            top_preds = p['labels'][p['scores'] > 0.5]
            if len(top_preds) >= len(t['labels']): y_pred_all.extend(top_preds[:len(t['labels'])])
            else:
                y_pred_all.extend(top_preds)
                y_pred_all.extend([0] * (len(t['labels']) - len(top_preds)))
        elif (p['scores'] > 0.5).any():
             y_true_all.extend([0]); y_pred_all.extend([p['labels'][p['scores'] > 0.5][0]])

    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(class_names)+1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['background'] + class_names, yticklabels=['background'] + class_names)
    plt.title("Confusion Matrix"); plt.grid(False)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300); plt.close()

def plot_training_history(csv_path, save_dir):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    
    # Mimic YOLO's 2-row layout: Train on top, Val on bottom
    # We will plot: cls_loss, box_reg_loss, objectness_loss, rpn_box_loss, total_loss
    metrics = [
        ('train_loss_classifier', 'val_loss_classifier', 'cls_loss'),
        ('train_loss_box_reg', 'val_loss_box_reg', 'box_loss'),
        ('train_loss_objectness', 'val_loss_objectness', 'obj_loss'),
        ('train_loss_rpn_box_reg', 'val_loss_rpn_box_reg', 'rpn_box_loss'),
        ('train_total_loss', 'val_total_loss', 'total_loss'),
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.style.use('default')
    
    for i, (train_col, val_col, title) in enumerate(metrics):
        if train_col in df.columns:
            ax = axes[0, i]
            ax.plot(epochs, df[train_col], marker='.', label='results', color='tab:blue', linewidth=2)
            ax.set_title(f'train/{title}')
            ax.grid(False)
        if val_col in df.columns:
            ax = axes[1, i]
            ax.plot(epochs, df[val_col], marker='.', label='results', color='tab:blue', linewidth=2)
            ax.set_title(f'val/{title}')
            ax.grid(False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- 4. Main Script ---
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = "/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset"
    class_names = ["Frog", "Mouse", "Snail"]
    
    # YOLO style run directory organization
    project_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/runs/faster_rcnn"
    run_name = "train_resnet50"
    run_dir = os.path.join(project_dir, run_name)
    weights_dir = os.path.join(run_dir, "weights")
    results_dir = os.path.join(run_dir, "results")
    
    for d in [weights_dir, results_dir]: os.makedirs(d, exist_ok=True)

    train_loader = DataLoader(YoloToFasterRCNNDataset(dataset_path, 'train'), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(YoloToFasterRCNNDataset(dataset_path, 'val'), batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(YoloToFasterRCNNDataset(dataset_path, 'test'), batch_size=16, shuffle=False, collate_fn=collate_fn)
    num_epochs = 100
    model = get_model(len(class_names)).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    
    # Implementation of Cosine Annealing LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = []
    best_loss = float('inf')

    print(f"Fine-tuning Faster R-CNN. Results will be saved to: {run_dir}")
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        train_comps = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad(); losses.backward(); optimizer.step()
            
            train_loss += losses.item()
            for k in train_comps.keys():
                if k in loss_dict: train_comps[k] += loss_dict[k].item()
                
            pbar.set_postfix(loss=losses.item())
        
        # --- Validation ---
        model.train() # Faster R-CNN needs to be in train mode to compute losses
        val_loss = 0
        val_comps = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
                for k in val_comps.keys():
                    if k in loss_dict: val_comps[k] += loss_dict[k].item()
                    
                pbar_val.set_postfix(val_loss=losses.item())
        
        # Step the scheduler
        lr_scheduler.step()
        
        # Average losses
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")
        
        epoch_data = {
            "epoch": epoch+1, 
            "train_total_loss": avg_train,
            "train_loss_classifier": train_comps['loss_classifier'] / len(train_loader),
            "train_loss_box_reg": train_comps['loss_box_reg'] / len(train_loader),
            "train_loss_objectness": train_comps['loss_objectness'] / len(train_loader),
            "train_loss_rpn_box_reg": train_comps['loss_rpn_box_reg'] / len(train_loader),
            "val_total_loss": avg_val,
            "val_loss_classifier": val_comps['loss_classifier'] / len(val_loader),
            "val_loss_box_reg": val_comps['loss_box_reg'] / len(val_loader),
            "val_loss_objectness": val_comps['loss_objectness'] / len(val_loader),
            "val_loss_rpn_box_reg": val_comps['loss_rpn_box_reg'] / len(val_loader),
            "lr": optimizer.param_groups[0]['lr']
        }
        history.append(epoch_data)
        
        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            
        # Periodically save plots and csv
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "results.csv"), index=False)
        plot_training_history(os.path.join(run_dir, "results.csv"), run_dir)

    print("Running final evaluation on Test set...")
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(weights_dir, "best.pt")))
    preds, targets = calculate_metrics(model, test_loader, device)
    generate_visual_results(preds, targets, class_names, run_dir)
    print(f"All plots, weights, and results.csv saved in: {run_dir}")

if __name__ == "__main__":
    main()
