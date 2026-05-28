#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from tqdm import tqdm

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

# Import central configurations
from config import (
    CLASSES,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    DEFAULT_PRETRAINED_RESNET_WEIGHTS,
    FALLBACK_PRETRAINED_RESNET_WEIGHTS,
    DEFAULT_CURATION_CONF_THRESHOLD,
    BUDGET_ALLOCATION_WLT,
    BUDGET_ALLOCATION_HARD_NEGS,
    BUDGET_ALLOCATION_OTHER_FAUNA,
    DEFAULT_CURATION_BUDGET,
)


class CropDataset(Dataset):
    """
    Dataset class that loads images and returns cropped patches
    based on bounding box coordinates (xmin, ymin, xmax, ymax).
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        try:
            img = Image.open(image_path).convert("RGB")
            # Crop using bounding box
            xmin, ymin = max(0, int(row["xmin"])), max(0, int(row["ymin"]))
            xmax, ymax = (
                min(img.width, int(row["xmax"])),
                min(img.height, int(row["ymax"])),
            )

            # Handle invalid boxes gracefully
            if xmax <= xmin or ymax <= ymin:
                crop = img.resize((224, 224))
            else:
                crop = img.crop((xmin, ymin, xmax, ymax))

            if self.transform:
                crop = self.transform(crop)
            return crop, idx
        except Exception as e:
            # Return a zero tensor if image is unreadable
            if self.transform:
                return torch.zeros((3, 224, 224)), idx
            return None, idx


class DomainFeatureExtractor(torch.nn.Module):
    """
    Feature extractor module that wraps a trained Faster R-CNN ResNet50 FPN backbone.
    Pools the final feature maps (Layer 4) to output a 2048-dimensional embedding.
    """

    def __init__(self, resnet_body):
        super().__init__()
        self.resnet_body = resnet_body
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # resnet_body (IntermediateLayerGetter) returns a dict of FPN layer outputs.
        # Layer 4 output corresponds to key '3' (size: [B, 2048, H, W])
        features_dict = self.resnet_body(x)
        features = features_dict["3"]
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1)  # shape: [B, 2048]


def extract_features(model, dataloader, device):
    """
    Extracts deep embeddings from crops in the loader using the feature extractor.
    """
    model.eval()
    features = []
    indices = []

    with torch.no_grad():
        for batch_imgs, batch_idx in tqdm(
            dataloader, desc="Extracting Domain Features"
        ):
            batch_imgs = batch_imgs.to(device)
            out = model(batch_imgs)
            features.append(out.cpu().numpy())
            indices.append(batch_idx.numpy())

    return np.concatenate(features, axis=0), np.concatenate(indices, axis=0)


def extract_camera_id(subfolder):
    """
    Extracts camera ID from the subfolder string (e.g. '6R/108MEDIA' -> '6R').
    """
    if pd.isna(subfolder):
        return "unknown"
    parts = str(subfolder).split("/")
    return parts[0]


def flag_static_triggers(df, iou_threshold=0.7, occurrence_threshold=15):
    """
    Clusters bounding boxes spatially across fixed cameras.
    Identifies boxes that trigger repeatedly in the same spot as static triggers.
    """
    if len(df) == 0:
        return df

    df = df.copy()
    df["camera_id"] = df["subfolder"].apply(extract_camera_id)
    df["is_static_trigger"] = False

    # Group by stationary camera
    for cam_id, group in df.groupby("camera_id"):
        if cam_id in ["unknown", "Observed", "Seen"]:
            continue

        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values
        indices = group.index.values
        n = len(boxes)

        clusters = []  # list of dicts: {'rep': [xmin, ymin, xmax, ymax], 'indices': []}

        for i in range(n):
            box = boxes[i]
            idx = indices[i]

            matched = False
            for cluster in clusters:
                rep = cluster["rep"]

                # Spatial IoU Calculation
                inter_x1 = max(box[0], rep[0])
                inter_y1 = max(box[1], rep[1])
                inter_x2 = min(box[2], rep[2])
                inter_y2 = min(box[3], rep[3])

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (rep[2] - rep[0]) * (rep[3] - rep[1])
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou >= iou_threshold:
                    cluster["indices"].append(idx)
                    m = len(cluster["indices"])
                    cluster["rep"] = [
                        (rep[0] * (m - 1) + box[0]) / m,
                        (rep[1] * (m - 1) + box[1]) / m,
                        (rep[2] * (m - 1) + box[2]) / m,
                        (rep[3] * (m - 1) + box[3]) / m,
                    ]
                    matched = True
                    break
            if not matched:
                clusters.append({"rep": list(box), "indices": [idx]})

        # Suppress static triggers
        for cluster in clusters:
            count = len(cluster["indices"])
            if count > occurrence_threshold:
                df.loc[cluster["indices"], "is_static_trigger"] = True

    df = df.drop(columns=["camera_id"])
    return df


def perform_diversity_sampling(
    curation_df, budget, feature_extractor, transform, batch_size, device, category_name
):
    """
    Extracts deep features and runs K-Means++ to select diverse representative cases inside a sub-pool.
    """
    if len(curation_df) == 0:
        return pd.DataFrame()

    curation_df = curation_df.copy()
    curation_df["is_representative"] = False

    n_samples = min(budget, len(curation_df))
    if n_samples <= 0:
        return curation_df

    dataset = CropDataset(curation_df, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    embeddings, idxs = extract_features(feature_extractor, dataloader, device)

    # Ensure ordering aligns perfectly
    order = np.argsort(idxs)
    embeddings = embeddings[order]

    # K-Means++ Clustering
    print(
        f"Running K-Means++ clustering on '{category_name}' with {n_samples} clusters..."
    )
    kmeans = KMeans(n_clusters=n_samples, init="k-means++", n_init=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    curation_df["cluster_id"] = cluster_labels

    # Select candidate with the highest uncertainty per cluster
    curation_df = curation_df.sort_values(
        by=["cluster_id", "uncertainty"], ascending=[True, False]
    )
    representative_idx = curation_df.groupby("cluster_id").head(1).index
    curation_df.loc[representative_idx, "is_representative"] = True

    curation_df = curation_df.drop(columns=["cluster_id"])
    return curation_df


def main():
    parser = argparse.ArgumentParser(
        description="Active Curation Pipeline with False Positive & WLT splits."
    )
    parser.add_argument(
        "--consensus_csv",
        type=str,
        required=True,
        help="Path to predictions/consensus CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output curation_priority.csv.",
    )
    parser.add_argument(
        "--resnet_weights",
        type=str,
        default=DEFAULT_PRETRAINED_RESNET_WEIGHTS,
        help=f"Path to domain-pretrained ResNet50/Faster R-CNN weights (.pt). Defaults to '{DEFAULT_PRETRAINED_RESNET_WEIGHTS}'.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CURATION_CONF_THRESHOLD,
        help=f"Confidence threshold to filter curation candidates (default: {DEFAULT_CURATION_CONF_THRESHOLD}).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_CURATION_BUDGET,
        help=f"Total number of representative samples to select (human curation budget) (default: {DEFAULT_CURATION_BUDGET}).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for spatial filter box clustering (default: {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help=f"Triggers count threshold for identifying static triggers (default: {DEFAULT_OCCURRENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel feature extraction.",
    )

    args = parser.parse_args()

    # Load and clean prediction data
    print(f"Loading predictions from {args.consensus_csv}...")
    df = pd.read_csv(args.consensus_csv)

    # 1. Compute uncertainty using entropy (uncertain classification) and bbox_variance (uncertain localization)
    # Ensure they exist in the input DataFrame. If missing, fall back gracefully
    if "entropy" not in df.columns:
        df["entropy"] = 1.0 - df["confidence"]
    if "bbox_variance" not in df.columns:
        df["bbox_variance"] = 0.0

    scaler = MinMaxScaler()
    df[["norm_entropy", "norm_bbox_var"]] = scaler.fit_transform(
        df[["entropy", "bbox_variance"]]
    )
    df["uncertainty"] = 0.5 * df["norm_entropy"] + 0.5 * df["norm_bbox_var"]
    df = df.drop(columns=["norm_entropy", "norm_bbox_var"])

    # 2. Flag Static Triggers using Spatial Bounding Box Filter
    print("\n[Curation Filter] Running spatial filter to flag static triggers...")
    df = flag_static_triggers(df, args.iou_threshold, args.occurrence_threshold)

    # 3. Categorize candidates to resolve the class imbalance and high-confidence background triggers
    #   - Category A (Hard Negatives): Stationary boxes triggered >= occurrence_threshold times (regardless of conf)
    #   - Category B (WLT Candidates): Positives predicted as WLT, not static, conf < 0.7 (requiring manual curation)
    #   - Category C (Other Fauna): Predicted as Other Fauna, not static, conf < args.conf_threshold

    hard_negs_df = df[df["is_static_trigger"] == True].copy()

    wlt_mask = (df["class_name"] == "Western_Leopard_Toad") & (
        df["is_static_trigger"] == False
    )
    wlt_df = df[
        wlt_mask & (df["confidence"] < 0.70) & (df["confidence"] >= 0.25)
    ].copy()

    other_mask = (df["class_name"] != "Western_Leopard_Toad") & (
        df["is_static_trigger"] == False
    )
    other_df = df[other_mask & (df["confidence"] < args.conf_threshold)].copy()

    print(f"\nCandidates breakdown:")
    print(f"  - Hard Negatives (Static triggers):    {len(hard_negs_df)}")
    print(f"  - WLT Candidates (Target positive):   {len(wlt_df)}")
    print(f"  - Other Fauna (Classifier support):    {len(other_df)}")

    # 4. Proportionally Split Human Curation Budget (n_clusters)
    wlt_budget = int(args.n_clusters * BUDGET_ALLOCATION_WLT)
    hard_negs_budget = int(args.n_clusters * BUDGET_ALLOCATION_HARD_NEGS)
    other_budget = args.n_clusters - wlt_budget - hard_negs_budget

    print(f"\nCuration budget allocations:")
    print(f"  - WLT budget:    {wlt_budget} samples ({BUDGET_ALLOCATION_WLT:.0%})")
    print(
        f"  - Hard Negatives: {hard_negs_budget} samples ({BUDGET_ALLOCATION_HARD_NEGS:.0%})"
    )
    print(
        f"  - Other Fauna:    {other_budget} samples ({BUDGET_ALLOCATION_OTHER_FAUNA:.0%})"
    )

    # 5. Load Domain-Pretrained ResNet50 Feature Extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Try resolving ResNet model weights from Faster R-CNN run checkpoints
    weights_path = args.resnet_weights
    if not os.path.exists(weights_path):
        weights_path = FALLBACK_PRETRAINED_RESNET_WEIGHTS

    if not os.path.exists(weights_path):
        print(
            f"Warning: Pretrained weights not found at '{args.resnet_weights}' or fallback."
        )
        print("Falling back to standard ImageNet-pretrained ResNet50...")
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        modules = list(resnet.children())[:-1]
        feature_extractor = torch.nn.Sequential(*modules).to(device)
    else:
        print(f"Loading domain-pretrained model weights from: {weights_path}")
        # Faster R-CNN has 3 classes + 1 background = 4 classes
        frcnn_model = fasterrcnn_resnet50_fpn_v2()
        in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
        frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

        # Load states dict
        frcnn_model.load_state_dict(torch.load(weights_path, map_location=device))

        # Wrap ResNet backbone
        feature_extractor = DomainFeatureExtractor(frcnn_model.backbone.body).to(device)

    # Transform logic for crops
    transform = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 6. Apply diversity K-Means++ clustering independently inside each sub-pool
    wlt_curated = perform_diversity_sampling(
        wlt_df,
        wlt_budget,
        feature_extractor,
        transform,
        args.batch_size,
        device,
        "WLT Detections",
    )
    if not wlt_curated.empty:
        wlt_curated["curation_reason"] = "Target positive (WLT)"

    hard_negs_curated = perform_diversity_sampling(
        hard_negs_df,
        hard_negs_budget,
        feature_extractor,
        transform,
        args.batch_size,
        device,
        "Hard Negatives",
    )
    if not hard_negs_curated.empty:
        hard_negs_curated["curation_reason"] = "Hard Negative (Static Background)"

    other_curated = perform_diversity_sampling(
        other_df,
        other_budget,
        feature_extractor,
        transform,
        args.batch_size,
        device,
        "Other Fauna",
    )
    if not other_curated.empty:
        other_curated["curation_reason"] = "Fauna/Classifier support"

    # 7. Merge sub-pools and save outputs
    all_curated = pd.concat(
        [wlt_curated, hard_negs_curated, other_curated], ignore_index=True
    )

    if all_curated.empty:
        print("No predictions require curation. Exiting.")
        return

    # Write result
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    all_curated.to_csv(args.output_csv, index=False)

    rep_count = all_curated["is_representative"].sum()
    print("\n=======================================================")
    print("ACTIVE CURATION COMPLETED SUCCESSFULLY")
    print(f"  Total prioritized predictions: {len(all_curated)}")
    print(f"  Human curation budget selected: {rep_count}")
    print(f"  Curation Priority File saved:  {args.output_csv}")
    print("=======================================================")


if __name__ == "__main__":
    main()
