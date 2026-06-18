# Leopard Toad Detection Pipeline

This directory contains the pipeline for automatically localizing and class-identifying Western Leopard Toads (WLT) in camera trap imagery using deep object detection models (YOLO, Faster R-CNN, and RT-DETR).

## Framework Overview

The overall object detection setup is divided into four main stages, as mapped out in the detection workflow diagram ([WLT.pdf](WLT.pdf)):

1. **Data Preparation**: Preprocessing raw sensor images and partitioning datasets by camera location to prevent evaluation bias.
2. **Transfer Learning**: Domain pre-training followed by phase-frozen finetuning on the target local dataset.
3. **Active Learning Loop**: Iterative query selection combining consensus models, classification/localization uncertainty metrics, and diverse feature clustering.
4. **Model Evaluation**: Precision, recall, threshold sweeping, and bounding-box mAP evaluation on ground-truth subsets.

---

## 1. Data Preparation

Data preparation extracts labeled target sets and formats the raw camera trap feed for model training.

* **CLAHE Preprocessing**: Night-time infrared camera trap imagery is preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) to amplify low-contrast patterns and borders.
  * Code: [preprocess_clahe.py](evaluation/preprocess_clahe.py)
* **Location-Based Splitting**: Datasets are partitioned into Train, Validation, and Test pools strictly based on physical camera locations. This ensures independent validation on unseen sensor backdrops.
  * Configuration and mappings: [dataset_utils.py](active%20learning/pipelines/dataset_utils.py)

---

## 2. Transfer Learning

Transfer learning is used to adapt general-domain object detectors to domain-specific wildlife monitoring.

* **Domain Adaptation & Pre-Training**: Foundational architectures (YOLO, Faster R-CNN, RT-DETR) are first pretrained on supplementary datasets (e.g., iNaturalist and Open Images).
  * Scripts and configurations: [pretraining/](pretraining)
* **Finetuning (Phase 1 & Phase 2)**: Finetuning models on the local target dataset occurs in two phases:
  * *Phase 1 (Freeze Backbone)*: Freezes the backbone layers to adapt only the detection head.
  * *Phase 2 (Global Finetuning)*: Unfreezes all layers for global optimization across all parameters.
  * Code: [train_model.py](active%20learning/pipelines/train_model.py)

---

## 3. Active Learning Loop

The active learning loop iteratively updates the models by selecting the most informative, hard, or ambiguous examples for manual annotation.

* **Loop Orchestrator**: Manages state-based execution, training phases, predictions, curation, and query exports cycle-by-cycle.
  * Code: [run_active_learning_loop.py](active%20learning/pipelines/run_active_learning_loop.py)
* **Inference on Unlabeled Pool**: Batch inference pipeline that generates predictions on the unlabeled dataset.
  * Code: [run_inference_pipeline.py](active%20learning/pipelines/run_inference_pipeline.py)
* **Difficulty-Calibrated Uncertainty Sampling (DCUS)**: Estimates classification entropy and calculates running class-wise difficulty coefficients (based on validation set box-matching or AP50) to weight and sum object-level uncertainties.
  * Code: [dcus_sampling.py](active%20learning/pipelines/dcus_sampling.py)
* **Category-Conditioned Matching Similarity (CCMS) & Diversity Clustering**: Extracts visual feature crops and calculates pairwise similarities, then queries diverse representatives using k-Center Greedy and modified k-Means++ clustering centroids.
  * Code: [ccms_sampling.py](active%20learning/pipelines/ccms_sampling.py)
* **Human Annotation (Oracle Export)**: Saves prioritized candidates to a priority query list for manual curation.
  * Code: [run_active_learning_loop.py (Oracle Export)](active%20learning/pipelines/run_active_learning_loop.py#L48-L76)

---

## 4. Model Evaluation

Evaluation assesses both image-level presence filtering and detection-level coordinate accuracy.

* **Detection-Level Evaluation**: Computes taxonomic class precision, recall, confusion matrices, and mean Average Precision (mAP50 and mAP50-95).
  * Code: [run_active_learning_eval.py](evaluation/pipelines/run_active_learning_eval.py)
* **Image-Level Filtering (Binary Evaluation)**: Evaluates the model's capacity to filter out empty frames (Recall vs. Specificity trade-offs).
  * Code: [binary_eval_test_pool_wlt.py](evaluation/pipelines/binary_eval_test_pool_wlt.py)
* **Threshold Sweeping Calibration**: Sweep precision-recall values across thresholds to optimize class-specific confidence cutoffs.
  * Code: [optimize_ultralytics_thresholds.py](evaluation/pipelines/optimize_ultralytics_thresholds.py)
* **Cross-Model Consensus Generation**: Merges predictions from multiple models to cross-reference agreement, estimating spatial and classification variance.
  * Code: [cross_reference.py](evaluation/curation/cross_reference.py)

---

## Interactive Visualization

* **Gradio Auditing Tool**: Interactive dashboard to review detection results, filter triggers, and inspect model outputs.
  * Code: [gradio_app.py](gradio_app.py) and [visualize_gradio.py](visualize_gradio.py)
