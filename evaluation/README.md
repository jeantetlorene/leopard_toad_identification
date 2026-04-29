# Evaluation & Active Curation Pipeline

This directory contains the scripts and outputs for evaluating the trained object detection models (YOLO, RT-DETR, Faster R-CNN) and strategically curating their predictions to build a high-quality ground truth dataset. The workflow is designed to handle large volumes of predictions, filter out noise through model consensus, and utilize advanced active learning techniques to efficiently review boundary cases.

## Workflow Overview

The pipeline consists of three sequential steps:

1. **Prediction Generation (`generate_predictions.py`)**
   Runs batch inference on the hold-out validation and test sets across all trained model checkpoints. It outputs raw bounding box predictions and confidence scores into separate CSV files within model-specific subdirectories (e.g., `yolo_cycle_4_pretrained_phase1/test.csv`).

2. **Model Consensus Cross-Referencing (`cross_reference.py`)**
   Aggregates the raw predictions from all models and clusters them spatially using an Intersection over Union (IoU $\ge$ 0.5) threshold. A prediction is only retained if it achieves a minimum consensus—specifically, if at least 3 unique models agree on the detection.
   - **Outputs:** Saves the filtered, highly robust predictions to the `consensus_predictions/` directory.
   - **Key Metrics Added:** `agreed_models_count`, `min_confidence`, `bbox_variance` (localization uncertainty), and `entropy` (classification confusion).

3. **Active Learning Curation (`active_curation.py`)**
   Addresses the diminishing returns of manual review by mathematically isolating redundant false positives (e.g., pebbles) and prioritizing high-uncertainty boundary cases.
   - Filters predictions below a specified confidence threshold (e.g., `< 0.90`).
   - Uses a pre-trained ResNet50 to extract 2048-dimensional deep feature embeddings from the cropped bounding boxes.
   - Applies **K-Means++ Clustering** to group visually similar artifacts.
   - Applies **Difficulty Calibrated Uncertainty Sampling (DCUS)** using classification entropy and bounding box variance to rank the samples for review.
   - **Outputs:** A prioritized CSV (`curation_priority.csv`) that allows reviewers to inspect a small number of representative samples from each cluster and immediately discard large groups of noise.

4. **Performance Evaluation (`plot_pr_curve.py`)**
   A standalone utility that calculates and plots the Precision-Recall curve of the detector based on your manual evaluations, helping to determine the optimal deployment threshold.
## Usage

### 1. Generate Predictions
```bash
python generate_predictions.py
```
*Ensure checkpoints are correctly placed and the target model architectures match before running.*

### 2. Cross-Reference
```bash
python cross_reference.py
```
*This will generate `val_consensus.csv` and `test_consensus.csv` in `consensus_predictions/`.*

### 3. Active Curation
```bash
python active_curation.py \
    --consensus_csv consensus_predictions/val_consensus.csv \
    --output_csv consensus_predictions/val_curation_priority.csv \
    --conf_threshold 0.85 \
    --n_clusters 100 \
    --batch_size 32
```
*Adjust `--conf_threshold` and `--n_clusters` based on your annotation budget and dataset size.*

### 4. Manual Review
Load the output `val_curation_priority.csv` into the Gradio UI (`detection/visualize_gradio.py`). Enable "Show Representative Boundary Cases Only" and flag the images. This automatically creates a `val_curation_priority_evaluations.csv`.

### 5. Plot Precision-Recall Curve
```bash
python plot_pr_curve.py consensus_predictions/val_curation_priority.csv
```
*Outputs a `pr_curve.png` plot showing the performance based on your manual review.*

## Directory Structure

- `generate_predictions.py`: Inference pipeline.
- `cross_reference.py`: IoU-based model consensus pipeline.
- `active_curation.py`: Feature-based diversity and uncertainty sampling script.
- `plot_pr_curve.py`: Utility to plot Precision-Recall curve after manual review.
- `consensus_predictions/`: Target directory for cross-referenced and actively curated output CSVs.
- `[model_name]_[cycle]_[phase]/`: Automatically generated directories storing raw prediction CSVs for each model variation.
