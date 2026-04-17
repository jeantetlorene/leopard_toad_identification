# Western Leopard Toad - Active Learning Pipeline

This folder contains a multi-architecture active learning (AL) suite designed to efficiently sample a 1.4TB dataset for toad detection. It supports **YOLO**, **RT-DETR**, and **Faster R-CNN** with fully isolated trajectories for comparing **Pretrained** vs. **From-Scratch** initializations.

## Directory Structure
- `/yolo`: YOLO specific scripts and configuration.
- `/rtdetr`: RT-DETR specific scripts and configuration.
- `/faster_rcnn`: PyTorch-native Faster R-CNN scripts.
- `/data`: Contains separate dataset folders:
  - `detect_pretrained/`: Dataset used for the Pretrained model loop.
  - `detect_scratch/`: Dataset used for the From-Scratch model loop.

## How to Run
Every pipeline requires a `--mode` argument to determine whether it uses the pretrained or from-scratch state.

### YOLO
```bash
cd yolo
python3 main_al_loop.py --mode pretrained
python3 main_al_loop.py --mode scratch
```

### RT-DETR
```bash
cd rtdetr
python3 main_al_loop.py --mode pretrained
python3 main_al_loop.py --mode scratch
```

### Faster R-CNN
```bash
cd faster_rcnn
python3 main_al_loop.py --mode pretrained
python3 main_al_loop.py --mode scratch
```

## State Management
- **States:** The script tracks progress via `al_state_pretrained.json` and `al_state_scratch.json` within each model folder.
- **Candidates:** Query results are exported to `al_query_candidates_pretrained.csv` and `al_query_candidates_scratch.csv`.

## Performance Optimizations
- **Threadpool Loading:** Uses 64 CPU workers for fast image I/O from network/disk storage.
- **Batched Inference:** Optimized for A6000 with huge batch sizes (512) and FP16 precision.
- **Data Leakage Protection:** Automatically blacklists any image already present in the training directory from appearing in the query candidates.

## To Restart a Cycle
If you wish to re-run from Cycle 0:
1. Delete the `al_state_*.json` files in the model folder.
2. (Optional) Clear the `runs/` folder to save disk space.
3. Ensure the respective `detect_*` folder in `/data` contains only the initial seed dataset.
