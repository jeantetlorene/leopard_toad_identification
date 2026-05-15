from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from models.base import BaseModel
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os


class SAHIWrapper(BaseModel):
    def __init__(
        self, model_type, model_path, device="cpu", confidence_threshold=0.001
    ):
        self.device = device
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        if "yolo" in model_type:
            sahi_type = "yolov8"
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
        elif "rtdetr" in model_type:
            sahi_type = "rtdetr"
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=sahi_type,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
            )
        elif "faster_rcnn" in model_type:
            # For Faster R-CNN, we need to instantiate our specific model and wrap it
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                box_score_thresh=confidence_threshold,
                min_size=640,
                max_size=640,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features,
                4,  # 3 classes + 1 background
            )

            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)

            model.to(device)
            model.eval()

            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type="torchvision",
                model=model,
                confidence_threshold=confidence_threshold,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported model type for SAHI: {model_type}")

    def predict_batch(self, images, **kwargs):
        """
        Perform SAHI tiled inference.
        Note: SAHI's get_sliced_prediction is typically per-image.
        We loop over the 'batch' of images provided.
        """
        batch_preds = []
        for img in images:
            # img is a numpy array (BGR from cv2.imread)
            # SAHI handles numpy arrays.

            # Use slice size 640x640 (standard for our models)
            # and 0.2 overlap as recommended in the Ultralytics SAHI guide.
            result = get_sliced_prediction(
                img,
                self.detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )

            preds = []
            h, w = img.shape[:2]
            for obj in result.object_prediction_list:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()

                # Normalize coordinates to [0, 1]
                # x_center, y_center, width, height (normalized)
                xn = (x1 + x2) / (2 * w)
                yn = (y1 + y2) / (2 * h)
                wn = (x2 - x1) / w
                hn = (y2 - y1) / h

                # Correct class ID if needed (for Faster R-CNN it might be 1-indexed in result if not careful)
                # But SAHI's category.id should match what the model returns.
                # For our YOLO/RTDETR, it's 0, 1, 2.
                # For Faster R-CNN, the model returns 1, 2, 3 internally,
                # but our FasterRCNNWrapper subtracts 1.
                # We need to check what SAHI does.

                cls_id = int(obj.category.id)
                if "faster_rcnn" in self.model_type:
                    # If SAHI uses the raw labels from torchvision model, they are 1-indexed.
                    # Our project uses 0-indexed: {0: 'Other_Amphibian', 1: 'Small_Mammal', 2: 'Western_Leopard_Toad'}
                    # So if cls_id is 1, 2, 3, we subtract 1.
                    cls_id -= 1

                preds.append(
                    {
                        "cls": cls_id,
                        "conf": round(float(obj.score.value), 4),
                        "bbox": [
                            round(float(xn), 4),
                            round(float(yn), 4),
                            round(float(wn), 4),
                            round(float(hn), 4),
                        ],
                    }
                )
            batch_preds.append(preds)
        return batch_preds
