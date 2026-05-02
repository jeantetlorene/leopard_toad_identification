import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


class YoloToFasterRCNNDatasetCLAHE(Dataset):
    def __init__(self, dataset_dir, split="train", img_size=640, augment=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        custom_img_dir = os.path.join(dataset_dir, split, "images")
        custom_lbl_dir = os.path.join(dataset_dir, split, "labels")
        yolo_img_dir = os.path.join(dataset_dir, "images", split)
        yolo_lbl_dir = os.path.join(dataset_dir, "labels", split)

        if os.path.exists(custom_img_dir):
            self.img_dir = custom_img_dir
            self.lbl_dir = custom_lbl_dir
        else:
            self.img_dir = yolo_img_dir
            self.lbl_dir = yolo_lbl_dir

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_files = [
            f
            for f in self.img_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")

            # --- Apply CLAHE ---
            # Convert to OpenCV format (BGR)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            img_clahe_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # Convert back to PIL Image (RGB)
            img = Image.fromarray(cv2.cvtColor(img_clahe_cv, cv2.COLOR_BGR2RGB))
            # -------------------

        except Exception as e:
            print(f"Warning: Skipping corrupted image {img_path}: {e}")
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

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
                        xmin = (xc - w / 2) * self.img_size
                        ymin = (yc - h / 2) * self.img_size
                        xmax = (xc + w / 2) * self.img_size
                        ymax = (yc + h / 2) * self.img_size

                        xmin = max(0.0, xmin)
                        ymin = max(0.0, ymin)
                        xmax = min(float(self.img_size), xmax)
                        ymax = min(float(self.img_size), ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id + 1)

        if self.augment:
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes) > 0:
                    new_boxes = []
                    for b in boxes:
                        new_boxes.append(
                            [self.img_size - b[2], b[1], self.img_size - b[0], b[3]]
                        )
                    boxes = new_boxes

            import PIL.ImageEnhance as ImageEnhance

            if np.random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(np.random.uniform(0.8, 1.2))

            if np.random.random() > 0.5:
                angle = np.random.uniform(-5, 5)
                img = img.rotate(angle, resample=Image.BILINEAR)

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
                if (x2 > x1 + 1) and (y2 > y1 + 1):
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
