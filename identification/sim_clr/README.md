# SimCLR Modular Identification Pipeline

This directory contains a modular implementation of the SimCLR (Contrastive Learning) pipeline for leopard toad identification.

## Key Features

- **Modular Training**: Specify data directories and output locations via command-line arguments.
- **ID-Aware Splitting**: Automatically splits data by individual (Toad ID) to ensure robust validation.
- **Interactive App**: Gradio-based application for similarity search and toad identification.

## Usage

### 1. Data Preparation
Organize your images into subfolders where each subfolder name is the individual's ID (e.g., `toad_0001/`, `toad_0002/`).

You can use the provided `identification/sort_chips.py` script to organize flat datasets if you have a mapping CSV.

### 2. Training
Train a new model on a specific dataset:

```bash
python identification/sim_clr/train.py \
    --data_dir /path/to/your/sorted_chips \
    --weights_dir /path/to/save/weights \
    --logs_dir /path/to/save/logs \
    --epochs 100 \
    --batch_size 32
```

### 3. Identification App
Launch the identification interface with specific weights and database:

```bash
python identification/sim_clr/app.py \
    --data_dir /path/to/your/sorted_chips \
    --weights_path /path/to/your/resnet50_backbone_final.pth
```

## Configuration
Default hyperparameters and device settings are located in `config.py`.
