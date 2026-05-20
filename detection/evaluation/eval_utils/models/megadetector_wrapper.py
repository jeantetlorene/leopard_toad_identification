import torch
from .base import BaseModel


class MegaDetectorWrapper(BaseModel):
    def __init__(self, model_path, imgsz=640, device="cpu"):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, device=device
        )
        self.model.conf = 0.001
        self.imgsz = imgsz
        self.device = device

        if "cuda" in str(device):
            self.model.half()

    def predict_batch(self, images, **kwargs):
        sub_batch = 32
        batch_preds = []

        # Process in chunks of sub_batch
        for i in range(0, len(images), sub_batch):
            chunk = images[i : i + sub_batch]
            results = self.model(chunk, size=self.imgsz)

            for result_tensor in results.xywhn:
                preds = []
                for box in result_tensor:
                    cls_idx = int(box[5].item())
                    if cls_idx == 0:  # 0 is animal
                        preds.append(
                            {
                                "cls": 0,
                                "conf": round(float(box[4].item()), 4),
                                "bbox": [round(float(x), 4) for x in box[:4].tolist()],
                            }
                        )
                batch_preds.append(preds)

        return batch_preds
