import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from sklearn.neighbors import NearestNeighbors

import argparse
from config import Config
from augmentations import ResizeAndPad, get_inference_transform
from model import SimCLRBackbone


def load_model(weights_path=None):
    model = SimCLRBackbone()
    if weights_path is None:
        weights_path = os.path.join(Config.WEIGHTS_DIR, "resnet50_backbone_final.pth")

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=Config.DEVICE)
        model.backbone.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Weights not found at {weights_path}. Using untrained backbone."
        )

    model.to(Config.DEVICE)
    model.eval()
    return model


def get_transform():
    return get_inference_transform(Config.IMG_SIZE)


# Global variables for the app
model = None
transform = None
db_embeddings = None
db_image_paths = None
db_toad_ids = None


def index_database(data_dir=None):
    global db_embeddings, db_image_paths, db_toad_ids
    if data_dir is None:
        data_dir = Config.DATA_DIR

    print(f"Indexing database from {data_dir}...")

    image_paths = []
    toad_ids = []

    # Walk through the data directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                full_path = os.path.join(root, file)
                toad_id = os.path.basename(root)
                image_paths.append(full_path)
                toad_ids.append(toad_id)

    if not image_paths:
        print("No images found in database directory.")
        return

    embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_images.append(transform(img))

            batch_tensor = torch.stack(batch_images).to(Config.DEVICE)
            feat = model(batch_tensor)
            embeddings.append(feat.cpu().numpy())

    db_embeddings = np.concatenate(embeddings, axis=0)
    # Normalize for cosine similarity
    db_embeddings = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    db_image_paths = image_paths
    db_toad_ids = toad_ids
    print(f"Indexed {len(db_image_paths)} images.")


def identify_toad(input_img, top_k=5):
    if input_img is None:
        return None, "Please upload an image."

    # Preprocess input image
    img = Image.fromarray(input_img).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        query_embedding = model(img_t).cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn.fit(db_embeddings)
    distances, indices = nn.kneighbors(query_embedding)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        sim = 1.0 - dist

        path = db_image_paths[idx]
        toad_id = db_toad_ids[idx]

        match_img = Image.open(path)
        results.append((match_img, f"ID: {toad_id}\nSimilarity: {sim:.3f}"))

    return results


# Create Gradio interface
with gr.Blocks(title="Leopard Toad Identification") as demo:
    gr.Markdown("# 🐸 Leopard Toad Identification (SimCLR)")
    gr.Markdown(
        "Upload a cropped toad image to find the most similar individuals in the database."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Toad Image")
            identify_btn = gr.Button("Identify", variant="primary")

        with gr.Column(scale=2):
            output_gallery = gr.Gallery(label="Top Matches", columns=3, height="auto")

    identify_btn.click(fn=identify_toad, inputs=input_image, outputs=output_gallery)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Toad Identification App.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=Config.DATA_DIR,
        help="Path to the directory containing indexed toad chips.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=os.path.join(Config.WEIGHTS_DIR, "resnet50_backbone_final.pth"),
        help="Path to model weights file.",
    )
    args = parser.parse_args()

    # Initialize app with arguments
    model = load_model(args.weights_path)
    transform = get_transform()
    index_database(args.data_dir)

    demo.launch(share=True)
