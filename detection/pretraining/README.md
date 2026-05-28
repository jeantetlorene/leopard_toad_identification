# Pretraining Module

This directory contains the data downloading, ingestion, model training, and evaluation scripts used during the **intermediate domain-specific pretraining phase** of the Leopard Toad detection project.

To keep the repository clean and maintainable, the pretraining folder has been modularized into three functional subdirectories. All scripts have been modernized to resolve paths dynamically relative to their script location, making the codebase fully transportable.

---

## Directory Structure

```text
pretraining/
├── data/              # Data downloading, filtering, and dataset preparation
│   ├── count_animals_by_dataset.py
│   ├── create_yolo_dataset.py
│   ├── download_california_small_animals.py
│   ├── download_ohio_small_animals.py
│   └── megadetect_toad.py
├── pipelines/         # Model pretraining and quantitative evaluation pipelines
│   ├── evaluate_faster_rcnn.py
│   ├── train_faster_rcnn.py
│   ├── train_rtdetr.py
│   ├── train_vit.py
│   └── train_yolo.py
├── utils/             # Dataset visualization and auditing utilities
│   ├── visualize_detections.py
│   └── visualize_yolo.py
└── runs/              # Pretraining run outputs, logs, and weights (checkpoint assets)
```

---

## Subdirectories Overview

### 1. Data Ingestion & Preparation (`/data`)
Scripts responsible for acquiring pretraining datasets and formatting them into standard object detection annotation layouts:
- **`download_california_small_animals.py`**: Downloads and filters small mammals and amphibians from the California region.
- **`download_ohio_small_animals.py`**: Downloads and prepares corresponding data from the Ohio region.
- **`create_yolo_dataset.py`**: Converts raw ingested data into standard YOLO-ready dataset structures.
- **`megadetect_toad.py`**: Utilizes MegaDetector for automated bounding box pre-labeling.
- **`count_animals_by_dataset.py`**: A helper analytical script to audit total animal detections across datasets.

### 2. Pretraining Pipelines (`/pipelines`)
Unified model pretraining scripts for different model architectures, enabling domain-specific pretraining prior to target active learning fine-tuning:
- **`train_faster_rcnn.py`**: Core pretraining pipeline for PyTorch Faster R-CNN using intermediate ResNet50 layers.
- **`train_yolo.py`**: Pretraining script for Ultralytics YOLOv8 models.
- **`train_rtdetr.py`**: Pretraining script for RT-DETR models.
- **`train_vit.py`**: Pretraining pipeline for Vision Transformer classifiers.
- **`evaluate_faster_rcnn.py`**: Evaluates pretrained Faster R-CNN weights against validation/test sets, generating Precision-Recall envelopes and confusion matrices.

### 3. Auditing & Visualization Utilities (`/utils`)
Visual inspection tools for auditing model prediction outputs and dataset labels:
- **`visualize_detections.py`**: Interactive Gradio block viewer that allows browsing MegaDetector-derived json bounding boxes.
- **`visualize_yolo.py`**: Renders annotation boxes directly from YOLO text format onto images for manual inspection.
