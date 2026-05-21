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

We provide a specialized report for evaluating the impact of **Contrast Limited Adaptive Histogram Equalisation (CLAHE)** on initial model performance (Cycle 0).

- **`results/files/preprocessing_results.md`**: Final report containing comparative tables and visualizations.
- **`reporting/generate_preprocessing_report.py`**: Automation script that aggregates metrics and generates PR curves.

## Architecture Benchmarking (Cycle 0)

We provide a report for benchmarking the fundamental baseline architectures (YOLO, RT-DETR, Faster R-CNN, and MegaDetector) at Cycle 0 before transfer learning.

- **`results/files/architecture_results.md`**: Final report containing computational benchmarking (Params, GFLOPs, Inference ms) and mAP50.
- **`reporting/generate_architecture_report.py`**: Automation script that computes confusion matrices and dynamic speed metrics.

### Visualizations
The PR curves are generated using standard VOC/COCO interpolation (monotonic decreasing) and are bounded from 0 to 1 recall. They include shaded areas and macro-averaged metrics (mAP, AR) in the legend.

---

## Unified Evaluation Pipeline

The evaluation pipeline is highly optimized to **decouple heavy inference from metrics calculation**. If raw prediction JSON files (`_raw.json`) are already generated and present in the `results/` directory, the master script will automatically skip inference and only calculate/update the performance metrics. 

### Modular Components
- **[`config.py`](eval_utils/config.py)**: Centralized configuration for paths, camera-to-dataset mappings, and hyperparameters.
- **[`data_utils.py`](eval_utils/data_utils.py)**: Utilities for parallel image loading, on-the-fly CLAHE enhancement, and robust ground-truth label matching.
- **[`metrics.py`](eval_utils/metrics.py)**: Mathematical implementation of binary image-level metrics and IoU-based detection matching.
- **[`inference.py`](eval_utils/inference.py)**: Core logic for loading a single model variant and generating spatial predictions. Supports a `--full_sequence` mode for large-scale filtering analysis.
- **[`evaluation_suite.py`](eval_utils/evaluation_suite.py)**: Comprehensive metric calculator. Aggregates cached prediction JSONs to output threshold sweeps, ROC-AUCs, and unified comparison CSVs.
- **[`binary_eval_test_pool.py`](pipelines/binary_eval_test_pool.py)**: Dedicated script for computing binary image-level metrics strictly on the test unlabeled pool (`test_full_seq`), generating ROC-AUC and threshold sweeps.
- **[`plot_binary_roc_baseline.py`](reporting/plot_binary_roc_baseline.py)**: Specialized script to plot bounded (0 to 1) ROC curves for the baseline architectures (YOLO, Faster R-CNN, RT-DETR) directly from raw predictions.
- **[`run_all_evaluations.py`](pipelines/run_all_evaluations.py)**: Master orchestration script for bulk evaluation.

### Usage

**1. Run Bulk Evaluation (Detection-Level & Subsets):**
This executes evaluation on both `test` and `val` datasets for all available model iterations dynamically discovered from the runs directory. 
- **Smart Resuming:** Models with partially generated `_raw.json` files will gracefully bypass completed images and append new predictions to save progress. Use `--overwrite` if you explicitly want to clear existing predictions and start fresh.
```bash
python3 pipelines/run_all_evaluations.py --batch_size 64
```

**2. Run Targeted Evaluation:**
Evaluate specific subsets of cycles, models, or variants. The script will dynamically find all available options that match your filters.
```bash
python3 pipelines/run_all_evaluations.py --cycles 4 --models yolo faster_rcnn --variants pretrained
```

**3. Run Full-Sequence Evaluation (Filtering Analysis):**
This executes predictions across the entire unbroken sequence of unlabelled images to explicitly verify binary filtering capability. 
```bash
python3 pipelines/run_all_evaluations.py --cycles 4 --models yolo --full_sequence --batch_size 256
```

**4. Run Evaluation Suite Independently:**
If predictions are already generated, you can run the evaluation suite independently with granular filtering.
```bash
python3 eval_utils/evaluation_suite.py --models yolo rtdetr --processing clahe plain --cycles 4 --variants pretrained scratch
```

**3. Generate Preprocessing Report:**
Gathers data for preprocessing effects.
```bash
python3 reporting/generate_preprocessing_report.py
```
Plot the PR Curves for the baselines:
```bash
python3 reporting/plot_preprocessing.py
```

**4. Generate Architecture Benchmark Report:**
Gathers data for architecture effects (requires inference benchmarking).
```bash
python3 reporting/generate_architecture_report.py
```
Plot the Confusion Matrices for the baselines:
```bash
python3 reporting/plot_architecture.py
```

**5. Generate Transfer Learning Report:**
Gathers data across cycles to track mAP progression.
```bash
python3 reporting/generate_transfer_learning_report.py
```

**6. Evaluate Image-Level Binary Filtering:**
Calculate sweep metrics, generate the image-level markdown report, and plot bounded ROC curves for the test unlabeled pool. Make sure to generate the MegaDetector baseline first.
```bash
python3 pipelines/run_megadetector.py
python3 pipelines/binary_eval_test_pool.py
python3 reporting/generate_image_level_report.py
python3 reporting/plot_binary_roc_baseline.py
```

---

## Directory Organization
- `pipelines/`: Core execution scripts for running evaluations.
- `reporting/`: Scripts for generating reports and plotting results.
- `eval_utils/`: Shared logic, configuration, metrics, and model wrappers.
- `docs/`: Supplemental documentation and instructions.
- `data/`: The authoritative ground truth. Contains `test/` and `val/` directories with `images/`, `labels/`, and `image_mapping.csv`.
- `results/files/`: Target directory for all generated evaluation metrics, unified sweep CSVs, and `preprocessing_results.md`.
- `results/plots/`: Target directory for all generated ROC plots and PR curve visualizations.
- `results/<model_type>_<processing>/`: Contains the raw prediction JSON files (`_raw.json`).
