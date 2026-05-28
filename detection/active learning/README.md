# Western Leopard Toad - Active Learning Suite

This directory contains the multi-architecture active learning (AL) suite designed to efficiently sample a 1.4TB camera trap dataset for Western Leopard Toad (WLT) detection. 

It supports **YOLO**, **RT-DETR**, and **Faster R-CNN** with fully isolated trajectories for comparing **Pretrained** vs. **From-Scratch** initializations.

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

## Running Isolated Experiments (`--experiment_name`)

If you want to run new data or start a new active learning loop without replacing or overwriting your previous results, you can use the optional `--experiment_name` parameter. The pipeline dynamically isolates all output files:

* **State Tracking JSON**: Saved as `al_state_<model>_<processing>_<mode>_<experiment_name>.json`.
* **Dataset Folders**: Isolates cycle datasets under `data/<model>/<mode>/<experiment_name>/cycle_{X}/`.
* **Model Checkpoints**: Saves intermediate runs under `<model>_clahe/runs/<experiment_name>/`.
* **Inference Detections**: Saves prediction CSVs under `results/<experiment_name>/`.
* **Curation Candidates**: Saves query candidate list CSVs under `<model>_clahe/cycles/<mode>/<experiment_name>/cycle_{X}/`.

*Note: Leaving `--experiment_name` empty falls back gracefully to standard default folders, keeping the suite fully backwards-compatible.*

---

## How to Run the Active Learning Loop Step-by-Step

The active learning loop consists of a 5-phase cycle. Follow these instructions to run the loop.

### Step 1: Initialize Cycle 0
Ensure that the cycle folder contains only the initial seed dataset (`dataset.yaml` and standard images).
* **For Default Experiments**: Place them in `detection/active learning/data/<model>/<mode>/cycle_0/`.
* **For Custom Experiments**: Place them in `detection/active learning/data/<model>/<mode>/<experiment_name>/cycle_0/`.

To start/reset a fresh loop at Cycle 0, run with the `--reset` flag:
```bash
.venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
  --model_type rtdetr \
  --clahe \
  --mode pretrained \
  --experiment_name exp2 \
  --reset
```

### Step 2: Run the Loop Orchestrator
To launch the automated pipeline cycle:
```bash
.venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
  --model_type [yolo/rtdetr/faster_rcnn] \
  --mode [pretrained/scratch] \
  --experiment_name exp2 \
  --budget 100 \
  --iou_threshold 0.7 \
  --occurrence_threshold 15
```

#### What happens during execution:
* **Phase 1 (Model Training)**: Triggered automatically. The orchestrator calls the unified `pipelines/train_model.py` module to train the target model. If `--mode pretrained` is used, it does Phased Unfreezing (Phase 1 head only, Phase 2 entire model) with automatic CLAHE LAB-space monkey-patching. If `--mode scratch` is used, it trains from scratch.
* **Phase 2 (Automated Batch Inference)**: The newly trained model is run on the massive unlabeled years' pools using `pipelines/run_inference_pipeline.py`.
* **Phase 3 (Active Curation)**: Overlapping camera trap detections are spatially clustered to suppress recurring background triggers. Detections are categorized and clustered using K-Means++ to select diverse representatives.
* **Phase 4 (Oracle Export)**: Generates a prioritized oracle candidate CSV listing the image paths to annotate:
  `detection/active learning/<model_folder>/cycles/<mode>/<experiment_name>/cycle_{X}/al_query_candidates_<mode>_cycle_{X}.csv`
* **Phase 5 (Loop State Update)**: Automatically increments the cycle count inside the model's state JSON and pauses.

### Step 3: Human Annotation & Cycle Progression
When the orchestrator pauses:
1. Review and annotate the prioritized candidate images listed in the cycle CSV in **Label Studio**.
2. Combine all previous training images with these newly annotated images.
3. Save the new combined training dataset in the next cycle folder:
   `detection/active learning/data/<model>/<mode>/[experiment_name]/cycle_{X + 1}/`
4. Re-run the active learning orchestrator command! The script will automatically load the updated state JSON, detect Cycle `X+1`, and begin the next cycle.
