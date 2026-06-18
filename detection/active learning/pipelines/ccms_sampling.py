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
from PIL import Image
from tqdm import tqdm

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

from config import (
    CLASSES,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OCCURRENCE_THRESHOLD,
    DEFAULT_PRETRAINED_RESNET_WEIGHTS,
    FALLBACK_PRETRAINED_RESNET_WEIGHTS,
    DEFAULT_CURATION_CONF_THRESHOLD,
    BUDGET_ALLOCATION_TARGET,
    BUDGET_ALLOCATION_HARD_NEGS,
    BUDGET_ALLOCATION_OTHER_CLASSES,
    DEFAULT_CURATION_BUDGET,
    CURATION_TARGET_CLASS,
    DETECTION_THRESHOLDS,
)


class CropDataset(Dataset):
    """Dataset class that loads images and returns cropped patches based on bounding box coordinates."""

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
            xmin, ymin = max(0, int(row["xmin"])), max(0, int(row["ymin"]))
            xmax, ymax = (
                min(img.width, int(row["xmax"])),
                min(img.height, int(row["ymax"])),
            )

            if xmax <= xmin or ymax <= ymin:
                crop = img.resize((224, 224))
            else:
                crop = img.crop((xmin, ymin, xmax, ymax))

            if self.transform:
                crop = self.transform(crop)
            return crop, idx
        except Exception:
            if self.transform:
                return torch.zeros((3, 224, 224)), idx
            return None, idx


class DomainFeatureExtractor(torch.nn.Module):
    """Wraps a trained Faster R-CNN ResNet50 FPN backbone and pools final feature maps to 2048-dim embeddings."""

    def __init__(self, resnet_body):
        super().__init__()
        self.resnet_body = resnet_body
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features_dict = self.resnet_body(x)
        features = features_dict["3"]
        pooled = self.pool(features)
        return pooled.view(pooled.size(0), -1)


class ImageNetFeatureExtractor(torch.nn.Module):
    """Wraps standard ImageNet ResNet50 model and pools to 2048-dim embeddings."""

    def __init__(self, resnet):
        super().__init__()
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        out = self.features(x)
        return out.view(out.size(0), -1)


def extract_features(model, dataloader, device):
    """Extracts deep embeddings from cropped patches in the loader."""
    model.eval()
    features = []
    indices = []

    with torch.no_grad():
        for batch_imgs, batch_idx in tqdm(
            dataloader, desc="Extracting visual features"
        ):
            batch_imgs = batch_imgs.to(device)
            out = model(batch_imgs)
            features.append(out.cpu().numpy())
            indices.append(batch_idx.numpy())

    return np.concatenate(features, axis=0), np.concatenate(indices, axis=0)


def extract_camera_id(subfolder):
    """Extracts camera ID from the subfolder string."""
    if pd.isna(subfolder):
        return "unknown"
    parts = str(subfolder).split("/")
    return parts[0]


def flag_static_triggers(df, iou_threshold=0.7, occurrence_threshold=15):
    """Clusters bounding boxes spatially across fixed cameras to flag stationary repeat triggers."""
    if len(df) == 0:
        return df

    df = df.copy()
    df["camera_id"] = df["subfolder"].apply(extract_camera_id)
    df["is_static_trigger"] = False

    for cam_id, group in df.groupby("camera_id"):
        if cam_id in ["unknown", "Observed", "Seen"]:
            continue

        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values
        indices = group.index.values
        n = len(boxes)

        clusters = []
        for i in range(n):
            box = boxes[i]
            idx = indices[i]

            matched = False
            for cluster in clusters:
                rep = cluster["rep"]

                # Spatial IoU
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

        for cluster in clusters:
            count = len(cluster["indices"])
            if count > occurrence_threshold:
                df.loc[cluster["indices"], "is_static_trigger"] = True

    df = df.drop(columns=["camera_id"])
    return df


def compute_ccms_matrix(unique_images, df_boxes, embeddings):
    """Computes Category Conditioned Matching Similarity (CCMS) matrix between images."""
    # Normalize features for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    norm_embeddings = embeddings / norms

    # Map image paths to their detected objects
    img_to_objs = {img: [] for img in unique_images}
    df_boxes_reset = df_boxes.reset_index(drop=True)

    for idx, row in df_boxes_reset.iterrows():
        img = row["image_path"]
        if img in img_to_objs:
            img_to_objs[img].append(
                {
                    "feat": norm_embeddings[idx],
                    "conf": float(row["confidence"]),
                    "class": row["class_name"],
                }
            )

    N = len(unique_images)

    # Pre-group features by class for faster matrix multiplication
    img_class_feats = {}
    for img in unique_images:
        objs = img_to_objs[img]
        class_feats = {}
        for obj in objs:
            c = obj["class"]
            if c not in class_feats:
                class_feats[c] = []
            class_feats[c].append(obj["feat"])
        for c in class_feats:
            class_feats[c] = np.array(class_feats[c])
        img_class_feats[img] = class_feats

    # S_prime[i, j] = S'(O_i, O_j)
    S_prime = np.zeros((N, N))
    for i in range(N):
        img_i = unique_images[i]
        objs_i = img_to_objs[img_i]
        if not objs_i:
            continue
        sum_conf_i = sum(obj["conf"] for obj in objs_i)
        if sum_conf_i == 0:
            continue

        for j in range(N):
            if i == j:
                S_prime[i, j] = 1.0
                continue
            img_j = unique_images[j]
            class_feats_j = img_class_feats[img_j]

            weighted_sim = 0.0
            for obj in objs_i:
                c = obj["class"]
                conf = obj["conf"]
                if c in class_feats_j:
                    sims = np.dot(class_feats_j[c], obj["feat"])
                    max_sim = np.max(sims)
                else:
                    max_sim = 0.0
                weighted_sim += conf * max_sim
            S_prime[i, j] = weighted_sim / sum_conf_i

    # Make symmetric
    S = 0.5 * (S_prime + S_prime.T)
    return S


def run_ccms_clustering(unique_images, S, k, max_iter=15):
    """Performs k-Center Greedy initialization and modified k-Means++ refinement based on CCMS."""
    N = len(unique_images)
    if N <= k:
        return unique_images

    # Distance matrix D = 1.0 - S
    D = np.clip(1.0 - S, 0.0, 1.0)

    # 1. k-Center Greedy Initialization
    np.random.seed(42)
    centers = [np.random.randint(0, N)]
    min_dist = D[:, centers[0]].copy()

    for _ in range(1, k):
        next_center = np.argmax(min_dist)
        centers.append(next_center)
        min_dist = np.minimum(min_dist, D[:, next_center])

    # 2. Modified k-Means++ Refinement
    for iteration in range(max_iter):
        assignments = np.argmin(D[:, centers], axis=1)
        new_centers = []
        changed = False

        for cluster_idx in range(k):
            members = np.where(assignments == cluster_idx)[0]
            if len(members) == 0:
                new_centers.append(centers[cluster_idx])
                continue

            # Find image with max summed similarity to all other members in the cluster
            sub_S = S[members, :][:, members]
            summed_sims = np.sum(sub_S, axis=1)
            best_idx = members[np.argmax(summed_sims)]
            new_centers.append(best_idx)

            if best_idx != centers[cluster_idx]:
                changed = True

        centers = new_centers
        if not changed:
            print(f"CCMS Clustering: Converged after {iteration + 1} iterations.")
            break

    return [unique_images[idx] for idx in centers]


def perform_diversity_sampling(
    curation_df, budget, feature_extractor, transform, batch_size, device, category_name
):
    """Runs crop feature extraction, CCMS scoring, and clustering to select diverse priority queries."""
    if len(curation_df) == 0:
        return pd.DataFrame()

    curation_df = curation_df.copy()
    curation_df["is_representative"] = False

    unique_images = curation_df["image_path"].unique().tolist()
    n_samples = min(budget, len(unique_images))
    if n_samples <= 0:
        return curation_df

    # Extract features for all boxes in curation pool
    dataset = CropDataset(curation_df, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    embeddings, idxs = extract_features(feature_extractor, dataloader, device)

    # Reorder embeddings to align with curation_df rows
    order = np.argsort(idxs)
    embeddings = embeddings[order]

    # Compute CCMS Matrix between unique images
    print(
        f"CCMS: Computing similarity matrix for '{category_name}' with {len(unique_images)} images..."
    )
    S = compute_ccms_matrix(unique_images, curation_df, embeddings)

    # Cluster using k-Center Greedy and Refinement
    print(
        f"CCMS: Running two-stage clustering for '{category_name}' ({n_samples} clusters)..."
    )
    representative_images = run_ccms_clustering(unique_images, S, n_samples)

    # Select the highest uncertainty box per representative image
    rep_mask = curation_df["image_path"].isin(representative_images)
    curation_df.loc[rep_mask, "is_representative"] = True

    # We want each representative image to be marked, but only one box per image is kept
    # So we sort by uncertainty descending and drop duplicate image paths for representatives
    curation_df = curation_df.sort_values(
        by=["is_representative", "uncertainty"], ascending=[False, False]
    )

    # Mark duplicates as not representative
    representatives_only = curation_df[curation_df["is_representative"] == True].copy()
    representatives_only = representatives_only.drop_duplicates(
        subset=["image_path"], keep="first"
    )

    curation_df["is_representative"] = False
    curation_df.loc[representatives_only.index, "is_representative"] = True

    return curation_df


def main():
    parser = argparse.ArgumentParser(
        description="CCMS & Diversity Curation Sampling Script."
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions with uncertainty.",
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
        help="Path to ResNet50 weights.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=DEFAULT_CURATION_CONF_THRESHOLD,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_CURATION_BUDGET,
        help="Curation budget.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="Filter IoU threshold.",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help="Static trigger threshold.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Feature extraction batch size."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    if df.empty:
        print("CCMS: Empty predictions CSV. Graceful exit.")
        df["is_representative"] = []
        df["curation_reason"] = []
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        return

    # Ensure uncertainty is in columns
    if "uncertainty" not in df.columns:
        df["uncertainty"] = 1.0 - df["confidence"]

    # 1. Flag Static Triggers
    df = flag_static_triggers(df, args.iou_threshold, args.occurrence_threshold)

    # 2. Categorize candidates
    has_target = CURATION_TARGET_CLASS in CLASSES.values()
    has_other = any(name != CURATION_TARGET_CLASS for name in CLASSES.values())

    hard_negs_df = df[df["is_static_trigger"] == True].copy()

    target_mask = (df["class_name"] == CURATION_TARGET_CLASS) & (
        df["is_static_trigger"] == False
    )
    target_upper_limit = 0.70
    for idx, name in CLASSES.items():
        if name == CURATION_TARGET_CLASS:
            target_upper_limit = DETECTION_THRESHOLDS.get(idx, 0.70)
            break

    target_df = df[
        target_mask
        & (df["confidence"] < target_upper_limit)
        & (df["confidence"] >= 0.25)
    ].copy()

    other_mask = (df["class_name"] != CURATION_TARGET_CLASS) & (
        df["is_static_trigger"] == False
    )
    other_df = df[other_mask & (df["confidence"] < args.conf_threshold)].copy()

    # 3. Proportional budget splits
    alloc_target = BUDGET_ALLOCATION_TARGET if has_target else 0.0
    alloc_hard_negs = BUDGET_ALLOCATION_HARD_NEGS
    alloc_other = BUDGET_ALLOCATION_OTHER_CLASSES if has_other else 0.0

    total_alloc = alloc_target + alloc_hard_negs + alloc_other
    if total_alloc > 0:
        alloc_target /= total_alloc
        alloc_hard_negs /= total_alloc
        alloc_other /= total_alloc
    else:
        alloc_hard_negs = 1.0

    target_budget = int(args.n_clusters * alloc_target) if has_target else 0
    hard_negs_budget = int(args.n_clusters * alloc_hard_negs)
    other_budget = (
        args.n_clusters - target_budget - hard_negs_budget if has_other else 0
    )

    # Dynamic redistribution
    n_target_imgs = len(target_df["image_path"].unique())
    n_hard_negs_imgs = len(hard_negs_df["image_path"].unique())
    n_other_imgs = len(other_df["image_path"].unique())

    target_selected_count = min(target_budget, n_target_imgs)
    hard_negs_selected_count = min(hard_negs_budget, n_hard_negs_imgs)
    other_selected_count = min(other_budget, n_other_imgs)

    remaining_budget = args.n_clusters - (
        target_selected_count + hard_negs_selected_count + other_selected_count
    )

    if remaining_budget > 0:
        if n_target_imgs > target_selected_count:
            add = min(remaining_budget, n_target_imgs - target_selected_count)
            target_selected_count += add
            remaining_budget -= add
        if remaining_budget > 0 and n_other_imgs > other_selected_count:
            add = min(remaining_budget, n_other_imgs - other_selected_count)
            other_selected_count += add
            remaining_budget -= add
        if remaining_budget > 0 and n_hard_negs_imgs > hard_negs_selected_count:
            add = min(remaining_budget, n_hard_negs_imgs - hard_negs_selected_count)
            hard_negs_selected_count += add
            remaining_budget -= add

    target_budget = target_selected_count
    hard_negs_budget = hard_negs_selected_count
    other_budget = other_selected_count

    print(f"CCMS Sub-pool Candidate Images / Budgets:")
    print(f"  - Target Positive: {n_target_imgs} images / {target_budget} budget")
    print(f"  - Hard Negatives:  {n_hard_negs_imgs} images / {hard_negs_budget} budget")
    print(f"  - Other Classes:   {n_other_imgs} images / {other_budget} budget")

    # 4. Load Domain-Pretrained Feature Extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.resnet_weights
    if not os.path.exists(weights_path):
        weights_path = FALLBACK_PRETRAINED_RESNET_WEIGHTS

    if not os.path.exists(weights_path):
        print(
            f"CCMS: Pretrained weights not found. Falling back to ImageNet ResNet50..."
        )
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        feature_extractor = ImageNetFeatureExtractor(resnet).to(device)
    else:
        print(f"CCMS: Loading domain-pretrained weights from {weights_path}")
        frcnn_model = fasterrcnn_resnet50_fpn_v2()
        in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
        # 3 classes + 1 background = 4 classes
        frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
        frcnn_model.load_state_dict(torch.load(weights_path, map_location=device))
        feature_extractor = DomainFeatureExtractor(frcnn_model.backbone.body).to(device)

    # Image transform
    transform = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 5. Run diversity clustering
    target_curated = perform_diversity_sampling(
        target_df,
        target_budget,
        feature_extractor,
        transform,
        args.batch_size,
        device,
        f"Target Positives ({CURATION_TARGET_CLASS})",
    )
    if not target_curated.empty:
        target_curated["curation_reason"] = f"Target positive ({CURATION_TARGET_CLASS})"

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
        "Other Support Classes",
    )
    if not other_curated.empty:
        other_curated["curation_reason"] = "Other active support class"

    # Merge and save
    all_curated = pd.concat(
        [target_curated, hard_negs_curated, other_curated], ignore_index=True
    )
    if all_curated.empty:
        print("CCMS: No predictions require curation.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    all_curated.to_csv(args.output_csv, index=False)

    rep_count = all_curated["is_representative"].sum()
    print(
        f"CCMS: Saved {rep_count} representative priority candidates to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
