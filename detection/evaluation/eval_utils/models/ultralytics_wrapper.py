from ultralytics import YOLO, RTDETR
from .base import BaseModel
import torch
from eval_utils.config import MIN_CONF_THRESHOLD


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
        # Cap sub_batch to a safe value to prevent OOM
        sub_batch = min(sub_batch, 256)
        half_precision = "cuda" in str(self.device)
        batch_preds = []

        # Process in chunks of sub_batch to prevent OOM
        for i in range(0, len(images), sub_batch):
            chunk = images[i : i + sub_batch]
            results = self.model(
                chunk,
                imgsz=self.imgsz,
                conf=MIN_CONF_THRESHOLD,
                device=self.device,
                verbose=False,
                stream=True,
                batch=sub_batch,
                half=half_precision,
            )
            for res in results:
                preds = []
                if res.boxes is not None and len(res.boxes) > 0:
                    # Move all predictions to CPU at once to prevent GPU-CPU sync bottleneck
                    classes = res.boxes.cls.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    xywhns = res.boxes.xywhn.cpu().numpy()
                    for cls_val, conf_val, bbox_val in zip(classes, confs, xywhns):
                        preds.append(
                            {
                                "cls": int(cls_val),
                                "conf": round(float(conf_val), 4),
                                "bbox": [
                                    round(float(bbox_val[0]), 4),
                                    round(float(bbox_val[1]), 4),
                                    round(float(bbox_val[2]), 4),
                                    round(float(bbox_val[3]), 4),
                                ],
                            }
                        )
                batch_preds.append(preds)

        # Clear cache once at the end to free GPU memory
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        return batch_preds
