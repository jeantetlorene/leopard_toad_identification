# Leopard Toad Identification

This directory contains the pipeline for individual leopard toad re-identification (re-ID) using contrastive learning (SimCLR).

## Directory Structure

- `sim_clr/`: Core SimCLR implementation, including training scripts and visualization apps.
- `toads_by_id/`: Raw images of leopard toads organized into subfolders by individual ID.
- `toads_by_id_crop/`: Cropped images of toads extracted using a fine-tuned RT-DETR detection model. This is the primary dataset for training the identification model.
- `hotspotter/`: Legacy or alternative re-identification tools.

## SimCLR Identification Pipeline

The identification system uses a SimCLR-based approach (Self-Supervised Contrastive Learning) to learn unique pattern representations of leopard toads without requiring explicit labels during the pre-training phase, followed by ID-aware splitting for robust evaluation.

### Components in `sim_clr/`:

1. **`config.py`**: Central configuration for hyperparameters, paths, and training settings.
2. **`augmentations.py`**: Specialized image augmentations (padding-aware resizing, Gaussian blur, color jitter) designed to preserve toad patterns.
3. **`dataset.py`**: Handles ID-aware data splitting to ensure validation is performed on individuals the model has never seen before.
4. **`model.py`**: ResNet50-based encoder with a non-linear projection head.
5. **`loss.py`**: NT-Xent (Normalized Temperature-scaled Cross Entropy) loss implementation.
6. **`train.py`**: Main training script with early stopping and learning rate scheduling (Linear Warmup + Cosine Annealing).
7. **`app.py`**: A Gradio-based web application for real-time toad identification and similarity retrieval.

## Getting Started

### 1. Training the Model
To start training the SimCLR model on the cropped toad dataset:
```bash
python sim_clr/train.py
```

### 2. Running Identification App
To launch the Gradio interface for identifying toads:
```bash
python sim_clr/app.py
```

## Setup
Ensure you are using the project's virtual environment:
```bash
source ../.venv/bin/activate
```
