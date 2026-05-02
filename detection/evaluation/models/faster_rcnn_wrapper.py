import os
import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from models.base import BaseModel


class FasterRCNNWrapper(BaseModel):
    def __init__(self, model_path, num_classes=3, device="cpu"):
        self.device = device
        # Replicate model creation logic from trainer.py
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

    def predict_batch(self, images):
        # Convert BGR images to RGB tensors
        inputs = []
        for img in images:
            img_rgb = img[:, :, ::-1]  # BGR to RGB
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)  # C, H, W
            inputs.append(img_tensor.to(self.device))

        with torch.no_grad():
            outputs = self.model(inputs)

        batch_preds = []
        for out in outputs:
            preds = []
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            # Faster R-CNN outputs [x1, y1, x2, y2] in pixels
            # We need to normalize them to [x_center, y_center, w, h] for consistency
            img_h, img_w = images[0].shape[:2]

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i]
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                x_center = (x1 + x2) / (2 * img_w)
                y_center = (y1 + y2) / (2 * img_h)

                preds.append(
                    {
                        "cls": int(labels[i]) - 1,  # 1-indexed to 0-indexed
                        "conf": float(scores[i]),
                        "bbox": [float(x_center), float(y_center), float(w), float(h)],
                    }
                )
            batch_preds.append(preds)
        return batch_preds


import os  # Needed for os.path.exists
