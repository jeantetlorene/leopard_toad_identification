import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
from collections import defaultdict

# Import local modules
from config import Config
from model import SimCLRBackbone
from augmentations import ResizeAndPad


def load_model():
    model = SimCLRBackbone()
    weights_path = os.path.join(Config.WEIGHTS_DIR, "resnet50_backbone_final.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=Config.DEVICE)
    model.backbone.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    return model


def get_transform():
    return T.Compose(
        [
            ResizeAndPad(Config.IMG_SIZE, fill=0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main():
    transform = get_transform()
    model = load_model()

    print("Collecting dataset metadata...")
    image_paths = []
    toad_ids = []
    id_counts = defaultdict(int)

    for root, dirs, files in os.walk(Config.DATA_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                full_path = os.path.join(root, file)
                toad_id = os.path.basename(root)
                image_paths.append(full_path)
                toad_ids.append(toad_id)
                id_counts[toad_id] += 1

    print(f"Total images: {len(image_paths)}")
    print(f"Total unique IDs: {len(id_counts)}")

    # Filter for queries (IDs with > 1 image)
    query_indices = [i for i, tid in enumerate(toad_ids) if id_counts[tid] > 1]
    print(f"Number of possible queries (IDs with > 1 image): {len(query_indices)}")

    if not query_indices:
        print("No IDs with multiple images found. Cannot perform retrieval evaluation.")
        return

    print("Extracting embeddings...")
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_images.append(transform(img))

            batch_tensor = torch.stack(batch_images).to(Config.DEVICE)
            feat = model(batch_tensor)
            embeddings.append(feat.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("Evaluating Top-K Retrieval...")
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    mrr = 0.0

    # Use NearestNeighbors for speed
    # We fit on ALL embeddings, but we will exclude the query itself from results
    nn = NearestNeighbors(
        n_neighbors=11, metric="cosine"
    )  # 11 because 1 is the query itself
    nn.fit(embeddings)

    for idx in tqdm(query_indices):
        query_id = toad_ids[idx]
        query_feat = embeddings[idx].reshape(1, -1)

        # Get neighbors
        distances, indices = nn.kneighbors(query_feat)

        # indices[0] contains the indices of neighbors, first one is likely the query itself
        neighbor_indices = indices[0]
        # Remove the query index if it's there
        neighbor_indices = [i for i in neighbor_indices if i != idx]
        # Take Top 10
        neighbor_indices = neighbor_indices[:10]

        # Check matches
        match_ranks = [
            r + 1 for r, i in enumerate(neighbor_indices) if toad_ids[i] == query_id
        ]

        if match_ranks:
            top1_hits += 1 if match_ranks[0] <= 1 else 0
            top5_hits += 1 if match_ranks[0] <= 5 else 0
            top10_hits += 1 if match_ranks[0] <= 10 else 0
            mrr += 1.0 / match_ranks[0]

    num_queries = len(query_indices)
    print("\n--- Re-Identification Evaluation Results ---")
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Top-1 Accuracy:  {top1_hits / num_queries:.4f}")
    print(f"Top-5 Accuracy:  {top5_hits / num_queries:.4f}")
    print(f"Top-10 Accuracy: {top10_hits / num_queries:.4f}")
    print(f"MRR:             {mrr / num_queries:.4f}")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
