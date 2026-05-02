from ultralytics import YOLO, RTDETR
from models.base import BaseModel
import torch


class UltralyticsWrapper(BaseModel):
    def __init__(self, model_type, model_path, imgsz=640, device="cpu"):
        if model_type == "yolo":
            self.model = YOLO(model_path)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.imgsz = imgsz
        self.device = device

    def predict_batch(self, images):
        results = self.model(
            images, imgsz=self.imgsz, conf=0.001, device=self.device, verbose=False
        )
        batch_preds = []
        for res in results:
            preds = []
            for box in res.boxes:
                preds.append(
                    {
                        "cls": int(box.cls[0]),
                        "conf": float(box.conf[0]),
                        "bbox": box.xywhn[0].tolist(),
                    }
                )
            batch_preds.append(preds)
        return batch_preds
