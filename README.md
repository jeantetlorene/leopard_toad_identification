# Leopard Toad Identification & Re-identification

This repository contains the codebase and methodology for detecting, tracking, and identifying individual leopard toads from multi-year camera trap data (2023-2025). The project combines state-of-the-art object detection models with feature-matching algorithms to automate ecological monitoring and population analysis.

## Repository Structure

The project is divided into two primary modules: **Detection** and **Identification**.

### 1. Detection (`/detection`)
This module handles the automatic localization of leopard toads in raw camera trap images using various object detection architectures (YOLOv8, Faster R-CNN, and RT-DETR). 

Key files and features:
- **`active learning/pipelines/run_inference_pipeline.py`**: A fully modular, configurable batch inference pipeline to run deep learning models (e.g. RT-DETR) on multi-year year folder structures. Features CLAHE contrast enhancement, batch prediction, and an integrated post-processing switch (`--filter_static`) to automatically remove background triggers.
- **`active learning/pipelines/filter_static_false_positives.py`**: Standalone post-processing script to eliminate stationary camera trap false positives (triggers on leaves, rocks, ripples, etc.) by clustering bounding boxes spatially within camera sequences and suppressing those that trigger more than `--occurrence_threshold` times.
- **`batch_inference.py`**: Automates detection on large datasets using YOLO or Faster R-CNN models.
- **`threshold_sweeping.py`**: Contains the threshold-sweeping evaluation pipeline. Generates PR curves, recall-threshold trade-off analyses, and confidence distributions to determine optimal model thresholds that maximize recall (targeting 95–98%) while minimizing manual review overhead. 
- **`active learning/pipelines/active_curation.py`**: Redesigned active learning curation script. Loads domain-pretrained ResNet50 weights, clusters crops using K-Means++ independently across balanced categories (40% WLT, 30% spatial hard negatives, 30% other fauna), and outputs human curation priority lists targeting the highest-uncertainty instances.
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

### Running the Inference & Filtering Pipeline
You can run the batch inference pipeline with automated static background trigger filtering using the following commands:

```bash
# Run batch inference and automatically clean stationary background false positives
.venv/bin/python "detection/active learning/pipelines/run_inference_pipeline.py" \
  --model_path "detection/active learning/rtdetr_clahe/runs/cycle_2_pretrained_phase2/weights/best.pt" \
  --output_dir "detection/results/detect_rtdetr_cycle2_clahe_pretrained" \
  --img_size 640 \
  --batch_size 128 \
  --filter_static \
  --iou_threshold 0.7 \
  --occurrence_threshold 15

# Run the static bounding box filter independently on an existing prediction CSV
.venv/bin/python "detection/active learning/pipelines/filter_static_false_positives.py" \
  --input_csv "detection/results/detect_rtdetr_cycle2_clahe_pretrained/all_unlabeled_predictions.csv" \
  --iou_threshold 0.7 \
  --occurrence_threshold 15

# Run the active curation priority selector pipeline using domain feature extraction and category splits
.venv/bin/python "detection/active learning/pipelines/active_curation.py" \
  --consensus_csv "detection/results/detect_rtdetr_cycle2_clahe_pretrained/all_unlabeled_predictions.csv" \
  --output_csv "detection/results/detect_rtdetr_cycle2_clahe_pretrained/curation_priority.csv" \
  --n_clusters 100
```