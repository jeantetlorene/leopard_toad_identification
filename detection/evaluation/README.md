# Model Evaluation Framework

This directory contains the unified evaluation framework for assessing the performance of object detection models (YOLO, RT-DETR, Faster R-CNN) on continuous camera trap data from the "4R" (test) and "5Z" (val) locations.

## Workflow: Two-Tiered Evaluation

The evaluation is designed to address the challenges of camera trap data (sparse detections, vast background noise) using a custom methodology:

1.  **Image-Level (Binary Classification)**: Evaluates the model's fundamental ability to flag animal-containing frames.
    - **Methodology**: To characterize real-world filtering performance, models can be evaluated on the **Full Unlabelled Sequence** (e.g., all 147k images from a single camera).
    - **Metrics**: **Recall** (Sensitivity) and **Specificity** (True Negative Rate).
2.  **Detection-Level (Instance Localization)**: Evaluates bounding box accuracy and classification strictly within the subset of images containing manual annotations.
    - **Metrics**: **Average Precision (AP)** and **Average Recall (AR)** calculated via confidence threshold sweeping (0.01 to 0.95).

---

## Preprocessing Analysis (Effect of CLAHE)

We provide a specialized report for evaluating the impact of **Contrast Limited Adaptive Histogram Equalisation (CLAHE)** on initial model performance (Cycle 0).

- **[`preprocessing_results.md`](preprocessing_results.md)**: Final report containing comparative tables and visualizations.
- **[`generate_preprocessing_report.py`](generate_preprocessing_report.py)**: Automation script that aggregates metrics and generates "Ultralytics-style" Precision-Recall curves.

### Visualizations
The PR curves are generated using standard VOC/COCO interpolation (monotonic decreasing) and are bounded from 0 to 1 recall. They include shaded areas and macro-averaged metrics (mAP, AR) in the legend.

---

## Unified Evaluation Pipeline

The evaluation pipeline is highly optimized to **decouple heavy inference from metrics calculation**. If raw prediction JSON files (`_raw.json`) are already generated and present in the `results/` directory, the master script will automatically skip inference and only calculate/update the performance metrics. 

### Modular Components
- **[`config.py`](config.py)**: Centralized configuration for paths, camera-to-dataset mappings, and hyperparameters.
- **[`data_utils.py`](data_utils.py)**: Utilities for parallel image loading, on-the-fly CLAHE enhancement, and robust ground-truth label matching.
- **[`metrics.py`](metrics.py)**: Mathematical implementation of binary image-level metrics and IoU-based detection matching.
- **[`inference.py`](inference.py)**: Core logic for loading a single model variant and generating spatial predictions. Supports a `--full_sequence` mode for large-scale filtering analysis.
- **[`evaluation_suite.py`](evaluation_suite.py)**: Comprehensive metric calculator. Aggregates cached prediction JSONs to output threshold sweeps, ROC-AUCs, and unified comparison CSVs.
- **[`run_all_evaluations.py`](run_all_evaluations.py)**: Master orchestration script for bulk evaluation. It manages inference caching and triggers the full evaluation suite automatically.

### Usage

**1. Run Bulk Evaluation (Detection-Level & Subsets):**
This executes evaluation on both `test` and `val` datasets for all available model iterations. Models with cached `_raw.json` files will gracefully bypass the inference phase.
```bash
python3 run_all_evaluations.py --batch_size 64
```

**2. Run Full-Sequence Evaluation (Filtering Analysis):**
This executes predictions across the entire unbroken sequence of unlabelled images to explicitly verify binary filtering capability. 
```bash
python3 run_all_evaluations.py --cycles 0 --models yolo --full_sequence --batch_size 256
```

**3. Generate Preprocessing Report:**
```bash
python3 generate_preprocessing_report.py
```

---

## Directory Organization
- `consensus_predictions/`: Ground truth mapping CSVs generated from manual audits.
- `data/`: Manually verified YOLO-format labels for the test and val sets.
- `results/files/`: Target directory for all generated evaluation metrics, unified sweep CSVs, and `preprocessing_results.md`.
- `results/plots/`: Target directory for all generated ROC plots and PR curve visualizations.
- `results/<model_type>_<processing>/`: Contains the raw prediction JSON files (`_raw.json`).
