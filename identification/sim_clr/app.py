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

# Import local modules
from config import Config
from augmentations import ResizeAndPad


class SimCLRBackbone(nn.Module):
    def __init__(self, base_model=models.resnet50):
        super(SimCLRBackbone, self).__init__()
        self.backbone = base_model(weights=None)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


def load_model():
    model = SimCLRBackbone()
    weights_path = os.path.join(Config.WEIGHTS_DIR, "resnet50_backbone_final.pth")

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=Config.DEVICE)
        model.backbone.load_state_dict(state_dict)
    else:
        print("Warning: Trained weights not found. Using untrained backbone.")

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


# Global variables for the app
model = None
transform = None
db_embeddings = None
db_image_paths = None
db_toad_ids = None


def index_database():
    global db_embeddings, db_image_paths, db_toad_ids
    print("Indexing database...")

    image_paths = []
    toad_ids = []

    # Walk through the data directory
    for root, dirs, files in os.walk(Config.DATA_DIR):
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


# Initialize app
model = load_model()
transform = get_transform()
index_database()

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
    demo.launch(share=True)
