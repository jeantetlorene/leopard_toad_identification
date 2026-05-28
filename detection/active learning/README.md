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

## Dynamic & Task-Agnostic Class Configurations

The active learning pipeline is completely generic and customizable. It reads the dataset's taxonomies from the original `classes.txt` files inside the cycle's folders dynamically, and supports arbitrary target mappings.

In `detection/active learning/central_config.py`, you can customize:
1. **`TARGET_CLASSES`**: List of class names to output from the trained models. If set to `None`, the model is trained on all categories from the dataset's `classes.txt`.
2. **`CLASS_MAPPING`**: A dictionary that maps dataset class names to model target class names. Any category not included in this mapping (or mapped to `None`) is ignored and treated as **background** (suppressed in annotations).
3. **`CURATION_TARGET_CLASS`**: The name of the primary class of interest for category-biased curation sampling.
4. **Generalized Budget Allocation**:
   - `BUDGET_ALLOCATION_TARGET`: Budget ratio for the target class of interest.
   - `BUDGET_ALLOCATION_HARD_NEGS`: Budget ratio for stationary background/hard negative clustering.
   - `BUDGET_ALLOCATION_OTHER_CLASSES`: Budget ratio for other active support classes predicted by the model.

### Configuration Examples:
* **Scenario 1: Train ONLY on WLT (others as background)**
  ```python
  TARGET_CLASSES = ["Western_Leopard_Toad"]
  CLASS_MAPPING = {"Western_Leopard_Toad": "Western_Leopard_Toad"}
  CURATION_TARGET_CLASS = "Western_Leopard_Toad"
  ```
* **Scenario 2: Merge non-target categories into "Other"**
  ```python
  TARGET_CLASSES = ["Other", "Western_Leopard_Toad"]
  CLASS_MAPPING = {
      "Other_Amphibian": "Other",
      "Small_Mammal": "Other",
      "Western_Leopard_Toad": "Western_Leopard_Toad"
  }
  CURATION_TARGET_CLASS = "Western_Leopard_Toad"
  ```
* **Scenario 3: Standard Training (default)**
  ```python
  TARGET_CLASSES = ["Other_Amphibian", "Small_Mammal", "Western_Leopard_Toad"]
  CLASS_MAPPING = None
  CURATION_TARGET_CLASS = "Western_Leopard_Toad"
  ```

During training, a lightweight `mapped` subfolder is created under the active learning data split, mapping label classes instantly using symlinks to save disk space.

---

## How to Run the Active Learning Loop Step-by-Step

The active learning orchestrator has been designed to be **highly parallelizable, robust, and smart**. It supports running multiple models, training modes, and pre-processing configurations in a single command, performs pre-flight sanity checks before starting, and skips steps that are already completed.

### Step 1: Initialize Cycle 0
Ensure that the cycle folder contains only the initial seed dataset (`train/` and `val/` splits containing `images/` and `labels/`).
* **For Default Experiments**: Place them in `detection/active learning/data/<model>/<mode>/cycle_0/`.
* **For Custom Experiments**: Place them in `detection/active learning/data/<model>/<mode>/<experiment_name>/cycle_0/`.

To start/reset a fresh loop for multiple configurations back to Cycle 0, run with the `--reset` flag:
```bash
.venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
  --model_type yolo rtdetr faster_rcnn \
  --mode pretrained \
  --clahe \
  --experiment_name detect_2 \
  --reset
```

### Step 2: Run the Loop Orchestrator (Multi-Model & Caching Support)
You can launch individual runs or schedule **multiple model architectures, modes, and pre-processing techniques simultaneously**:
```bash
.venv/bin/python "detection/active learning/pipelines/run_active_learning_loop.py" \
  --model_type yolo rtdetr faster_rcnn \
  --mode pretrained scratch \
  --clahe \
  --plain \
  --experiment_name detect_2 \
  --budget 100
```

#### Key Orchestrator Features:
1. **Pre-flight Sanity Checks**: 
   Before running any computation, the script validates **all planned configurations** to verify:
   - Cycle dataset directories exist and contain valid training subdirectories.
   - Initial domain-pretrained model weights are present on disk if `--mode pretrained` is requested.
   If any configuration fails checks, it halts immediately with clear directions to prevent wasting hours of training.

2. **Fine-grained Skipping (Step Caching)**:
   If a run fails halfway through a configuration (e.g. out of memory, power loss) or if you want to rerun a batch, the script **automatically detects and skips already-completed phases**:
   - **Phase 1 Skip**: Skips model training if `best.pt` is already found.
   - **Phase 2 Skip**: Skips time-consuming unlabeled inference if the output predictions CSV is found.
   - **Phase 3 Skip**: Skips active curation if the diversity-prioritized CSV exists.
   - **Phase 4 Skip**: Skips candidate export if the final `al_query_candidates_<mode>_cycle_{X}.csv` exists.
   - **Full Cycle Skip**: Skips the entire loop iteration for that configuration if the final CSV is already exported.
   *Note: Use the `--force` flag to override caching and force re-execution of all phases.*

### Step 3: Human Annotation & Cycle Progression
When the orchestrator pauses:
1. Review and annotate the prioritized candidate images listed in each configuration's cycle CSV in **Label Studio**.
2. Combine all previous training images with these newly annotated images.
3. Save the new combined training dataset in the next cycle folder for each model configuration:
   `detection/active learning/data/<model>/<mode>/[experiment_name]/cycle_{X + 1}/`
4. Re-run the active learning orchestrator command! The script will automatically load the updated state JSON, detect Cycle `X+1`, and begin the next cycle.
