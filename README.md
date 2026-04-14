# Leopard Toad Identification & Re-identification

This repository contains the codebase and methodology for detecting, tracking, and identifying individual leopard toads from multi-year camera trap data (2023-2025). The project combines state-of-the-art object detection models with feature-matching algorithms to automate ecological monitoring and population analysis.

## Repository Structure

The project is divided into two primary modules: **Detection** and **Identification**.

### 1. Detection (`/detection`)
This module handles the automatic localization of leopard toads in raw camera trap images using various object detection architectures (YOLOv8, Faster R-CNN, and RT-DETR). 

Key files and features:
- **`batch_inference.py`**: Automates detection on large datasets. Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing in parallel using `ThreadPoolExecutor`, performs batch inference, and outputs bounding box coordinates and confidence scores to CSV files. Setup to support multiple model types (YOLO, Faster R-CNN).
- **`threshold_sweeping.py`**: Contains the threshold-sweeping evaluation pipeline. Generates PR curves, recall-threshold trade-off analyses, and confidence distributions to determine optimal model thresholds that maximize recall (targeting 95–98%) while minimizing manual review overhead. 
- **`visualize_gradio.py` & `gradio_app.py`**: Interactive Gradio interfaces used for visually auditing model predictions, with options to quickly jump to specific image indices and review outputs in real time.

### 2. Identification (`/identification`)
Once toads are detected and cropped, this module is responsible for the re-identification of individual toads to determine if unique individuals are reoccurring within the habitat.

Key files and features:
- **`batch_hotspotter.py`**: A batch processing script utilizing a Hotspotter-inspired algorithm (SIFT feature extraction + FLANN-based matching) to compare cropped toad images against a database, generating potential matches for manual validation.
- **`hotspotter.ipynb`**: Interactive notebook version of the Hotspotter matching process for fine-grained testing and parameter tuning.
- **`train_simclr.ipynb`**: Notebook detailing a deep-learning approach utilizing SimCLR (contrastive learning) to pull visual embeddings from toad patterns for advanced re-identification.
- **`visualize.ipynb`**: General visualization notebook for viewing matched pairs and feature keypoints.

## Tech Stack
- **Deep Learning**: PyTorch, Torchvision, Ultralytics YOLOv8
- **Computer Vision**: OpenCV (CLAHE, SIFT, FLANN)
- **UI / Visualization**: Gradio, Matplotlib
- **Data Engineering**: Pandas, concurrent.futures

## Getting Started
Ensure you have the required dependencies listed in your virtual environment (`.venv`).
Most scripts are designed to be run as standalone modules or via interactive Jupyter notebooks depending on if you are doing inference, training, or manual review.