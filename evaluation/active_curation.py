import os
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


class CropDataset(Dataset):
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
            xmin, ymin = max(0, row["xmin"]), max(0, row["ymin"])
            xmax, ymax = min(img.width, row["xmax"]), min(img.height, row["ymax"])

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


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    indices = []

    with torch.no_grad():
        for batch_imgs, batch_idx in tqdm(dataloader, desc="Extracting Deep Features"):
            batch_imgs = batch_imgs.to(device)
            # Forward pass
            out = model(batch_imgs)
            # Flatten spatial dimensions
            out = out.view(out.size(0), -1)
            features.append(out.cpu().numpy())
            indices.append(batch_idx.numpy())

    return np.concatenate(features, axis=0), np.concatenate(indices, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Active Curation Pipeline")
    parser.add_argument(
        "--consensus_csv", type=str, required=True, help="Path to val_consensus.csv"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output curation_priority.csv",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.85,
        help="Confidence threshold to filter",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=100, help="Number of clusters for KMeans++"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction"
    )

    args = parser.parse_args()

    print(f"Loading consensus predictions from {args.consensus_csv}...")
    df = pd.read_csv(args.consensus_csv)

    # Filter for predictions requiring active curation (low/mid confidence)
    curation_df = df[df["confidence"] < args.conf_threshold].copy()
    print(
        f"Found {len(curation_df)} predictions below {args.conf_threshold} threshold."
    )

    if len(curation_df) == 0:
        print("No predictions require curation. Exiting.")
        return

    # Normalize entropy and bbox_variance to compute uncertainty score
    scaler = MinMaxScaler()
    curation_df[["norm_entropy", "norm_bbox_var"]] = scaler.fit_transform(
        curation_df[["entropy", "bbox_variance"]]
    )

    # Difficulty Calibrated Uncertainty Score (linear combination for this binary use-case)
    # Give equal weight to classification confusion and localization variance
    curation_df["uncertainty"] = (
        0.5 * curation_df["norm_entropy"] + 0.5 * curation_df["norm_bbox_var"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Pre-trained ResNet50
    print("Loading pre-trained ResNet50...")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet50(weights=weights)
    # Remove the final classification layer (fc) to get 2048-d features
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules).to(device)

    # Setup Dataset and DataLoader
    transform = v2.Compose(
        [
            v2.Resize((224, 224), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CropDataset(curation_df, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Extract Features
    embeddings, idxs = extract_features(feature_extractor, dataloader, device)

    # Ensure order matches df
    order = np.argsort(idxs)
    embeddings = embeddings[order]

    # K-Means Clustering
    n_clusters = min(args.n_clusters, len(curation_df))
    print(f"Running K-Means++ clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=5, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    curation_df["cluster_id"] = cluster_labels

    # Identify representatives (highest uncertainty per cluster)
    curation_df = curation_df.sort_values(
        by=["cluster_id", "uncertainty"], ascending=[True, False]
    )
    curation_df["is_representative"] = False
    representative_idx = curation_df.groupby("cluster_id").head(1).index
    curation_df.loc[representative_idx, "is_representative"] = True

    # Save Output
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Clean up intermediate norm columns before saving
    curation_df = curation_df.drop(columns=["norm_entropy", "norm_bbox_var"])

    curation_df.to_csv(args.output_csv, index=False)
    print(f"Successfully saved curation priorities to {args.output_csv}")
    print(
        f"Top {n_clusters} representative boundary cases are flagged 'is_representative' = True."
    )


if __name__ == "__main__":
    main()
