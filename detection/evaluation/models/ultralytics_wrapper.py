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

    def predict_batch(self, images, **kwargs):
        sub_batch = kwargs.get("sub_batch_size", 128)
        results = self.model(
            images, imgsz=self.imgsz, conf=0.01, device=self.device, verbose=False,
            stream=True, batch=sub_batch
        )
        batch_preds = []
        for res in results:
            preds = []
            for box in res.boxes:
                preds.append(
                    {
                        "cls": int(box.cls[0]),
                        "conf": round(float(box.conf[0]), 4),
                        "bbox": [round(x, 4) for x in box.xywhn[0].tolist()],
                    }
                )
            batch_preds.append(preds)
        return batch_preds
