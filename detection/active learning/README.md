# Western Leopard Toad - Active Learning Suite

This directory contains the multi-architecture active learning (AL) suite designed to efficiently sample a 1.4TB camera trap dataset for Western Leopard Toad (WLT) detection. 

It supports **YOLO**, **RT-DETR**, and **Faster R-CNN** with fully isolated trajectories for comparing **Pretrained** vs. **From-Scratch** initializations.

---

## Repository Structure

- **`config.py`**: Central single-source-of-truth configuration file. Contains defaults for model paths, prediction directories, class maps, optimal validation thresholds, active learning budget ratios, and pretrained ResNet weights.
- **`__init__.py`**: Packages the folder and resolves workspace pathing using robust environment setups.
- **`pipelines/`**: The core modular active learning pipeline engine:
  - **`run_active_learning_loop.py`**: The unified orchestrator. Coordinates training (Phase 1), batch inference (Phase 2), active curation (Phase 3), oracle candidate CSV exporting (Phase 4), and loop state progression (Phase 5).
  - **`run_inference_pipeline.py`**: High-performance batch inference engine that processes years of camera trap images in parallel (using `ThreadPoolExecutor` and FP16 precision) with integrated CLAHE contrast preprocessing.
  - **`filter_static_false_positives.py`**: Bounding box spatial clustering script that identifies and suppresses stationary background false positives (triggers on leaves, rocks, or waves) across fixed cameras.
  - **`active_curation.py`**: Priority curation engine. Uses a domain-pretrained ResNet50 backbone to extract deep visual embeddings of cropped detections, applies split-budget K-Means++ clustering to avoid class imbalance, and selects representative queries.
  - **`config.py`**: Imports the central configuration parameters.
- **`yolo/`**, **`rtdetr/`**, **`faster_rcnn/`**: Sibling folders housing individual training and architecture definition scripts.
- **`data/`**: Cycle-specific datasets containing active learning labels and split yaml files.

---

## Curation Strategy (Solving Imbalance & False Positives)

Standard global uncertainty/diversity sampling fails due to two main problems:
1. **Generic False Positives:** Leaf ripples, waving grass, or rocks trigger thousands of highly confident bounding boxes in the same spot.
2. **Extreme Class Imbalance:** WLT are extremely rare compared to small mammals, other amphibians, or empty frames, meaning K-Means++ globally will drown out WLT samples.

### The Balanced Category Split
Our active curation pipeline solves these issues by:
1. Running the **spatial bounding box filter** to cluster boxes per camera and flag stationary repeat triggers.
2. Dividing all detections into three sub-pools:
   - **Western Leopard Toad (40% Budget)**: Confident but non-static toad candidates to enrich the positive train set.
   - **Stationary Background Triggers (30% Budget)**: Confident stationary triggers to be verified as background empty images, forcing the model to ignore them.
   - **Other Fauna (30% Budget)**: Small mammals and other amphibians to support classifier boundary definitions.
3. Performing **independent K-Means++ clustering** inside each sub-pool to select the most diverse, high-uncertainty representatives.

---

## How to Run the Active Learning Loop Step-by-Step

The active learning loop consists of a 5-phase cycle. Follow these instructions to run the loop.

### Step 1: Initialize Cycle 0
Ensure that the cycle folder `detection/active learning/data/<model>/<mode>/cycle_0/` contains only the initial seed dataset (`dataset.yaml` and standard images).
* To start a fresh loop or reset any existing cycle states, run the loop orchestrator with the `--reset` flag:
  ```bash
  .venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
    --model_type rtdetr \
    --clahe \
    --mode pretrained \
    --reset
  ```

### Step 2: Run the Loop Orchestrator
Run the unified active learning loop orchestrator:
```bash
.venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
  --model_type [yolo/rtdetr/faster_rcnn] \
  --mode [pretrained/scratch] \
  --budget 100 \
  --iou_threshold 0.7 \
  --occurrence_threshold 15
```

#### What happens during the execution:
* **Phase 1 (Model Training)**: Triggered automatically. The orchestrator calls the unified `pipelines/train_model.py` module to train the target model. If `--mode pretrained` is used, it does Phased Unfreezing (Phase 1 head only, Phase 2 entire model). If `--mode scratch` is used, it trains from scratch.
* **Phase 2 (Automated Batch Inference)**: The newly trained model is run on the massive unlabeled years' pools using `pipelines/run_inference_pipeline.py`.
* **Phase 3 (Active Curation)**: Overlapping camera trap detections are spatially clustered to suppress recurring background triggers. Detections are categorized and clustered using K-Means++ to select diverse representatives.
* **Phase 4 (Oracle Export)**: Generates a prioritized oracle candidate CSV listing the image paths to annotate:
  `detection/active learning/<model_folder>/cycles/<mode>/cycle_{X}/al_query_candidates_<mode>_cycle_{X}.csv`
* **Phase 5 (Loop State Update)**: Automatically increments the cycle count inside the model's state JSON and pauses.

### Step 3: Human Annotation & Cycle Progression
When the orchestrator pauses:
1. Review and annotate the prioritized candidate images listed in the cycle CSV in **Label Studio**.
2. Combine all previous training images with these newly annotated images.
3. Save the new combined training dataset in the next cycle folder:
   `detection/active learning/data/<model>/<mode>/cycle_{X + 1}/`
4. Re-run the active learning orchestrator command! The script will automatically load the updated state JSON, detect Cycle `X+1`, and begin the next cycle.
