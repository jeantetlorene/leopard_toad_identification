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
        self,
        model_type,
        model_path,
        device="cpu",
        confidence_threshold=0.001,
        sahi_batch_size=32,
        no_standard_prediction=True,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    ):
        self.device = device
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.sahi_batch_size = sahi_batch_size
        self.no_standard_prediction = no_standard_prediction
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.half = device != "cpu"

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
        Perform SAHI tiled inference with native GPU batching for slices.
        """
        from sahi.slicing import slice_image
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import GreedyNMMPostprocess

        # 1. Generate all slices for all images in the batch (Parallelized)
        all_slices = []
        slice_meta = []  # (image_idx, offset_x, offset_y, full_h, full_w)

        from concurrent.futures import ThreadPoolExecutor

        def process_img(args):
            idx, img = args
            h_in, w_in = img.shape[:2]
            res = slice_image(
                img,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
            )
            slices = list(res.images)
            metas = [
                {"img_idx": idx, "offset": p, "full_shape": [h_in, w_in]}
                for p in res.starting_pixels
            ]
            if not self.no_standard_prediction:
                slices.append(img)
                metas.append(
                    {"img_idx": idx, "offset": [0, 0], "full_shape": [h_in, w_in]}
                )
            return slices, metas

        print(f"  [SAHI] Slicing {len(images)} images...")
        with ThreadPoolExecutor(
            max_workers=min(len(images), os.cpu_count() or 4)
        ) as executor:
            results_slicing = list(executor.map(process_img, enumerate(images)))

        for slices, metas in results_slicing:
            all_slices.extend(slices)
            slice_meta.extend(metas)

        if not all_slices:
            return [[] for _ in images]

        # 2. Perform batch inference on all slices in chunks to avoid memory spikes
        all_raw_preds = []
        if "yolo" in self.model_type or "rtdetr" in self.model_type:
            num_slices = len(all_slices)
            print(
                f"  [SAHI] Inferencing {num_slices} slices (batch_size={self.sahi_batch_size})..."
            )
            # Run in chunks of sahi_batch_size to be safe and responsive
            for i in range(0, len(all_slices), self.sahi_batch_size):
                batch_slices = all_slices[i : i + self.sahi_batch_size]
                batch_slices_rgb = [s[:, :, ::-1] for s in batch_slices]
                results = self.detection_model.model(
                    batch_slices_rgb,
                    batch=len(batch_slices),
                    half=self.half,
                    verbose=False,
                    conf=self.confidence_threshold,
                )
                all_raw_preds.extend([r.boxes.data.cpu().numpy() for r in results])
        else:
            # Fallback for other models
            for i in range(0, len(all_slices), self.sahi_batch_size):
                batch_slices = all_slices[i : i + self.sahi_batch_size]
                for s in batch_slices:
                    self.detection_model.perform_inference(s)
                    all_raw_preds.append(
                        self.detection_model.original_predictions[0]
                        if isinstance(self.detection_model.original_predictions, list)
                        else self.detection_model.original_predictions
                    )

        # 3. Convert raw predictions to ObjectPrediction and group by image
        predictions_per_image = [[] for _ in images]
        category_mapping = self.detection_model.category_mapping

        for raw_preds, meta in zip(all_raw_preds, slice_meta):
            img_idx = meta["img_idx"]
            offset = meta["offset"]
            full_shape = meta["full_shape"]

            for pred in raw_preds:
                x1, y1, x2, y2, conf, cls_id = pred[:6]
                if conf < self.confidence_threshold:
                    continue

                # Shift and normalize
                # pred is [x1, y1, x2, y2] relative to slice
                x1_full = x1 + offset[0]
                y1_full = y1 + offset[1]
                x2_full = x2 + offset[0]
                y2_full = y2 + offset[1]

                h, w = full_shape
                xn = (x1_full + x2_full) / (2 * w)
                yn = (y1_full + y2_full) / (2 * h)
                wn = (x2_full - x1_full) / w
                hn = (y2_full - y1_full) / h

                cls_id = int(cls_id)
                if "faster_rcnn" in self.model_type:
                    cls_id -= 1

                predictions_per_image[img_idx].append(
                    {
                        "cls": cls_id,
                        "conf": round(float(conf), 4),
                        "bbox": [
                            round(float(xn), 4),
                            round(float(yn), 4),
                            round(float(wn), 4),
                            round(float(hn), 4),
                        ],
                        # Store raw box for merging
                        "_raw_bbox": [x1_full, y1_full, x2_full, y2_full],
                    }
                )

        # 4. Post-process (merge) predictions for each image
        # We use SAHI's GreedyNMM logic but on our converted boxes
        # Alternatively, we can use SAHI's ObjectPrediction and PostprocessPredictions
        final_batch_preds = []
        postprocess = GreedyNMMPostprocess(
            match_threshold=0.5,
            match_metric="IOS",
            class_agnostic=False,
        )

        for img_idx, image_preds in enumerate(predictions_per_image):
            if not image_preds:
                final_batch_preds.append([])
                continue

            # Convert to SAHI ObjectPrediction for postprocess
            sahi_preds = []
            for p in image_preds:
                sahi_preds.append(
                    ObjectPrediction(
                        bbox=p["_raw_bbox"],
                        category_id=p["cls"],
                        score=p["conf"],
                        category_name=category_mapping[
                            str(
                                p["cls"]
                                + (1 if "faster_rcnn" in self.model_type else 0)
                            )
                        ],
                        full_shape=meta["full_shape"],
                    )
                )

            merged_sahi_preds = postprocess(sahi_preds)

            # Convert back to our project's format
            h, w = images[img_idx].shape[:2]
            formatted_preds = []
            for obj in merged_sahi_preds:
                x1, y1, x2, y2 = obj.bbox.to_xyxy()
                xn = (x1 + x2) / (2 * w)
                yn = (y1 + y2) / (2 * h)
                wn = (x2 - x1) / w
                hn = (y2 - y1) / h
                formatted_preds.append(
                    {
                        "cls": int(obj.category.id),
                        "conf": round(float(obj.score.value), 4),
                        "bbox": [
                            round(float(xn), 4),
                            round(float(yn), 4),
                            round(float(wn), 4),
                            round(float(hn), 4),
                        ],
                    }
                )
            final_batch_preds.append(formatted_preds)

        return final_batch_preds
