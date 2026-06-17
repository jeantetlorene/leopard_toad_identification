# Western Leopard Toad (WLT) Detection & Identification

This repository contains the codebase and tools for the automatic detection and individual re-identification (re-ID) of Western Leopard Toads (WLT) from camera trap data, primarily to evaluate migratory tunnel usage.

## Repository Structure

The project is structured into three main folders:

### 1. dataset/
Contains the raw and processed datasets used throughout the project.

### 2. detection/
Handles toad localization and bounding box detection within camera trap images.
* **pretraining/**: Model pre-training scripts using supplementary datasets (e.g., iNaturalist, Open Images).
* **active learning/**: The active learning loop orchestrator, pipelines, and selectors to iteratively curate high-yield training samples.
* **results/**, **runs/**, **training/**: Output folders containing detection predictions, trained weights, and training run progress logs.

### 3. identification/
Responsible for re-identifying individual toads across tunnel ends to track recurrence and evaluate tunnel usage.
* **data/**: Image crops of detected toads.
* **hotspotter/**: The re-identification engine implementing a Hotspotter-inspired algorithm (SIFT descriptor extraction, FLANN matching, and RANSAC spatial verification) to match toad spot patterns between the opposite ends of migratory road-underpass tunnels (Z vs. R cameras) within the same year.
* **results/**: Subfolder storing all matching CSV outputs and compiled premium visual match PDF reports.
* **simclr/** (Experimental): Directory containing early experimental contrastive learning (SimCLR) implementations (not used in the final production pipeline).

---

## Getting Started

To get started with running inference, training, or re-identification:
1. Ensure your virtual environment (`.venv`) is activated.
2. For object detection and model training, refer to the tools in the [detection/](detection) folder.
3. For re-identification and match generation, refer to the [identification/hotspotter/README.md](identification/hotspotter/README.md).