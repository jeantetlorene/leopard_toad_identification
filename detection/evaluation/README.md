# Model Evaluation Framework

This directory contains the unified evaluation framework for assessing the performance of object detection models (YOLO, RT-DETR, Faster R-CNN) on continuous camera trap data from the "4R" (test) and "5Z" (val) locations.

## Workflow: Two-Tiered Evaluation

The evaluation is designed to address the challenges of camera trap data (sparse detections, vast background noise) using a custom methodology:

1.  **Image-Level (Binary Classification)**: Evaluates the model's fundamental ability to flag animal-containing frames across the entire continuous sequence.
    - **Metrics**: **Recall** (Sensitivity) and **Specificity** (True Negative Rate).
2.  **Detection-Level (Instance Localization)**: Evaluates bounding box accuracy and classification strictly within the subset of images containing manual annotations.
    - **Metrics**: **Average Precision (AP)** and **Average Recall (AR)** calculated via confidence threshold sweeping (0.01 to 0.95).

---

## Unified Evaluation Pipeline

The framework is modularized for efficiency and standardized comparison across all model architectures and preprocessing variants (Plain vs. CLAHE).

### Modular Components
- **[`config.py`](config.py)**: Centralized configuration for paths, camera-to-dataset mappings, and hyperparameters.
- **[`data_utils.py`](data_utils.py)**: Utilities for parallel image loading, on-the-fly CLAHE enhancement, and robust ground-truth label matching using original file basenames.
- **[`metrics.py`](metrics.py)**: Mathematical implementation of binary image-level metrics and IoU-based detection matching.
- **[`models/`](models/)**: Unified wrappers for YOLO, RT-DETR, and Faster R-CNN ensuring a consistent inference interface.
- **[`evaluate_single.py`](evaluate_single.py)**: Core logic for evaluating a single model variant on a specific dataset.
- **[`run_all_evaluations.py`](run_all_evaluations.py)**: Master orchestration script for bulk evaluation across all active learning cycles and variants.

### Usage

To perform the full evaluation across all model variations (3 architectures × 2 preprocessing variants × 5 cycles):

```bash
# Must be run within the project's .venv
cd detection/evaluation
../../.venv/bin/python3 run_all_evaluations.py --batch_size 64
```

### Outputs
Results are automatically organized in the `results/` directory:
- **`all_models_summary.csv`**: A macro-level comparison of all evaluated models at a baseline (0.1) threshold.
- **`[model]_[processing]/`**: Detailed per-cycle results:
    - `{cycle}_{variant}_{dataset}_metrics.csv`: Full threshold sweeping data for Recall/Specificity curves.
    - `{cycle}_{variant}_{dataset}_raw.json`: Raw prediction and ground truth data for every processed image.

---

## Directory Organization
- `consensus_predictions/`: Ground truth mapping CSVs generated from manual audits.
- `data/`: Manually verified YOLO-format labels for the test and val sets.
- `results/`: Target directory for all generated evaluation metrics and raw prediction JSONs.
