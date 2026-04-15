import os
import torch
import cv2
import yaml
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, YolosForObjectDetection, TrainingArguments, Trainer
import albumentations as A

# SOTA ViT model (YOLOS - You Only Look at One Sequence) training script
# This script adapts YOLO format dataset to Hugging Face Transformers for ViT Object Detection

class YOLODataset(Dataset):
    def __init__(self, yaml_path, split="train", processor=None):
        with open(yaml_path, 'r') as f:
            self.data_info = yaml.safe_load(f)
        
        self.root_dir = self.data_info['path']
        self.image_dir = os.path.join(self.root_dir, self.data_info[split])
        self.label_dir = self.image_dir.replace("images", "labels")
        
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.processor = processor
        
        # We apply basic augmentations. YOLOS image processor will handle resizing and normalization.
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=['class_labels']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, x, y, bw, bh = map(float, parts)
                        boxes.append([x, y, bw, bh])
                        classes.append(int(c))
                        
        if self.transform and len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=classes)
            image = transformed['image']
            boxes = transformed['bboxes']
            classes = transformed['class_labels']
            
        # Convert YOLO [x_center, y_center, width, height] (normalized) 
        # to COCO [x_min, y_min, width, height] (unnormalized)
        coco_boxes = []
        for (xc, yc, bw, bh) in boxes:
            xmin = (xc - bw / 2) * w
            ymin = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            coco_boxes.append([xmin, ymin, box_w, box_h])
            
        annotations = []
        for i, box in enumerate(coco_boxes):
            annotations.append({
                "image_id": idx,
                "category_id": classes[i],
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            })
            
        target = {
            "image_id": idx,
            "annotations": annotations
        }
        
        # The processor handles resizing and creating the pixel_values and labels tensors
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Remove batch dimension added by the processor
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        return {"pixel_values": pixel_values, "labels": target}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # We pad the images to the maximum size in the batch
    batch_dict = processor.pad(pixel_values, return_tensors="pt")
    batch_dict["labels"] = labels
    return batch_dict

def main(args):
    global processor
    
    print(f"Loading processor and model for {args.model_name}...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    print("Preparing datasets...")
    train_dataset = YOLODataset(args.data_yaml, split="train", processor=processor)
    val_dataset = YOLODataset(args.data_yaml, split="val", processor=processor)
    
    id2label = {int(k): str(v) for k, v in train_dataset.data_info['names'].items()}
    label2id = {str(v): int(k) for k, v in id2label.items()}
    
    print(f"Classes found: {id2label}")
    
    model = YolosForObjectDetection.from_pretrained(
        args.model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        save_steps=1000,
        eval_strategy="epoch",  # evaluates at the end of each epoch
        save_strategy="epoch",  # saves at the end of each epoch
        logging_steps=50,
        learning_rate=args.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none" # Disable wandb/tensorboard for simplicity unless requested
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the final model and processor
    model_save_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(model_save_path)
    processor.save_pretrained(model_save_path)
    print(f"Training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a SOTA ViT Model (YOLOS) on a YOLO-format dataset.")
    parser.add_argument("--data_yaml", type=str, default="/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model_name", type=str, default="hustvl/yolos-small", help="Hugging Face ViT model name (e.g., hustvl/yolos-small, hustvl/yolos-base)")
    parser.add_argument("--output_dir", type=str, default="./yolos_output", help="Directory to save weights and logs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    main(args)
