import os
import glob
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

class YolosYoloSubsetDataset(Dataset):
    def __init__(self, dataset_dir, classes, image_processor):
        self.dataset_dir = dataset_dir
        self.image_processor = image_processor
        self.image_paths = glob.glob(os.path.join(dataset_dir, "images", "*.jpg")) + \
                           glob.glob(os.path.join(dataset_dir, "images", "*.JPG")) + \
                           glob.glob(os.path.join(dataset_dir, "images", "*.png"))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Apply augmentation (ColorJitter avoids altering bounding boxes)
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
                        
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_id)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
        target = {
            "image_id": torch.tensor([idx]),
            "annotations": [
                {
                    "image_id": idx,
                    "category_id": l.item(),
                    "bbox": [b[0].item(), b[1].item(), (b[2]-b[0]).item(), (b[3]-b[1]).item()],
                    "area": a.item(),
                    "iscrowd": 0
                } for b, l, a in zip(boxes, labels, area)
            ]
        }
        
        encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        return pixel_values, target

def main():
    dataset_dir = "/home/Joshua/Downloads/leopard_toad_identification/dataset/detect_2/dataset_clahe"
    classes = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
    
    id2label = {i: name for i, name in enumerate(classes)}
    label2id = {name: i for i, name in enumerate(classes)}
    
    print("Loading YOLOS processor and model from standard ViT object detection repository...")
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
    model = YolosForObjectDetection.from_pretrained(
        "hustvl/yolos-small",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Check if a custom ViT best weight exists from previous runs and load it
    best_weights = "/home/Joshua/Downloads/OIDv4_ToolKit/runs/vit_model/weights/best.bin"
    if os.path.exists(best_weights):
        print(f"Loading custom specific ViT weights from {best_weights}...")
        model.load_state_dict(torch.load(best_weights, map_location="cpu"))
    
    # FREEZE ViT BASE LAYERS for partial fine-tuning
    print("Freezing ViT base layers (encoder)...")
    for param in model.vit.parameters():
        param.requires_grad = False
        
    global image_processor_ref
    image_processor_ref = processor

    def custom_collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = image_processor_ref.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }
    
    dataset = YolosYoloSubsetDataset(dataset_dir, classes, processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimize only parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    
    num_epochs = 10
    print("Starting partial fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float("inf")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    save_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/vit_finetune/model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved partially fine-tuned ViT model to {save_dir}")

if __name__ == "__main__":
    main()
