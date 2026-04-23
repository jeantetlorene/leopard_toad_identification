import json
import csv
import os
import numpy as np
import argparse
from tqdm import tqdm
from ultralytics import RTDETR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["pretrained", "scratch"],
    required=True,
    help="Run AL loop using either pretrained base or purely from scratch.",
)
args = parser.parse_args()

os.environ["AL_MODE"] = args.mode

# Enforce working directory to be this folder so RTDETR strictly downloads weights and creates any cache/runs here
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRETRAINED_WEIGHTS,
    SCRATCH_WEIGHTS,
    BUDGET_PER_CYCLE,
    AL_STATE_JSON,
    INFER_BATCH_SIZE,
)
from trainer import train_phase_1, train_phase_2, train_scratch
from inference import get_unlabeled_pool, extract_features_and_boxes_batch
from sampler import DCUS, diversity_sampling


def save_state(state):
    with open(AL_STATE_JSON, "w") as f:
        json.dump(state, f)


def load_state():
    if os.path.exists(AL_STATE_JSON):
        with open(AL_STATE_JSON, "r") as f:
            return json.load(f)
    return {"cycle": 0, "model_paths": {}}


def write_candidates_csv(selected_paths, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "status"])
        for path in selected_paths:
            writer.writerow([path, "To annotate"])
    print(f"\n[ORACLE PAUSE] Exported {len(selected_paths)} queries to {csv_path}")
    print(
        "Please annotate these images in Label Studio, move the YOLO-formatted labels back to your cycle-specific 'train' folder, and run this script again."
    )


def main():
    state = load_state()
    cycle = state["cycle"]

    print(
        f"\n{'=' * 50}\nStarting Active Learning Cycle {cycle} [{args.mode.upper()}]\n{'=' * 50}"
    )

    # Always start from baseline weights to prevent catastrophic forgetting
    pretrained_base = PRETRAINED_WEIGHTS
    scratch_base = SCRATCH_WEIGHTS

    from config import BASE_DIR, RTDETR_DIR

    model_name = "rtdetr"
    cycle_dir = os.path.join(RTDETR_DIR, "cycles", args.mode, f"cycle_{cycle}")
    os.makedirs(cycle_dir, exist_ok=True)

    dataset_dir = os.path.join(
        BASE_DIR, "active learning", "data", model_name, args.mode, f"cycle_{cycle}"
    )
    dataset_yaml = os.path.join(cycle_dir, f"dataset_{args.mode}_cycle_{cycle}.yaml")

    yaml_content = f"""path: {dataset_dir}
train: train/images
val: val/images
test: test/images

names:
  0: Other_Amphibian
  1: Small_Mammal
  2: Western_Leopard_Toad
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)

    print(f">> Cycle {cycle}: Training Model.")
    if args.mode == "pretrained":
        expected_p2_model = os.path.join(
            RTDETR_DIR, "runs", f"cycle_{cycle}_pretrained_phase2", "weights", "best.pt"
        )
        if os.path.exists(expected_p2_model):
            print(
                f"  Found existing trained model for Cycle {cycle}. Skipping training."
            )
            pretrained_model = expected_p2_model
        else:
            print("\n--- Pretrained Model (Phased Unfreezing) ---")
            p1 = train_phase_1(
                pretrained_base,
                f"cycle_{cycle}_pretrained",
                dataset_yaml,
                freeze=15,
                epochs=100,
                patience=25,
            )
            p2 = train_phase_2(p1, f"cycle_{cycle}_pretrained", dataset_yaml, epochs=30)
            pretrained_model = p2
        state["model_paths"]["pretrained"] = pretrained_model
    else:
        expected_scratch_model = os.path.join(
            RTDETR_DIR, "runs", f"cycle_{cycle}_scratch_scratch", "weights", "best.pt"
        )
        if os.path.exists(expected_scratch_model):
            print(
                f"  Found existing trained model for Cycle {cycle}. Skipping training."
            )
            scratch_model = expected_scratch_model
        else:
            print("\n--- From-Scratch Model ---")
            scratch_model = train_scratch(
                scratch_base, f"cycle_{cycle}_scratch", dataset_yaml, epochs=60
            )
        state["model_paths"]["scratch"] = scratch_model

    print(
        "\n>> Scanning for unlabelled images from /srv ... (Excluding 4R, 5Z val/test cameras)"
    )
    pool = get_unlabeled_pool(args.mode, cycle)
    print(f"Found {len(pool)} unlabelled images in the massive pool.")

    if len(pool) == 0:
        print("No unlabeled images found. Exiting.")
        return

    print("\n>> Starting Inference & Feature Extraction")
    if args.mode == "pretrained":
        if not os.path.exists(pretrained_model):
            print(f"Error: {pretrained_model} not found.")
            return
        active_model = RTDETR(pretrained_model)
    else:
        if not os.path.exists(scratch_model):
            print(f"Error: {scratch_model} not found.")
            return
        active_model = RTDETR(scratch_model)

    dcus = DCUS()

    valid_paths = []
    image_uncertainties = []
    embeddings_list = []

    for i in tqdm(range(0, len(pool), INFER_BATCH_SIZE), desc="Inferring Pool"):
        chunk = pool[i : i + INFER_BATCH_SIZE]
        batch_paths, batch_boxes, batch_features = extract_features_and_boxes_batch(
            active_model, chunk
        )

        for k in range(len(batch_paths)):
            boxes = batch_boxes[k]
            u_img = dcus.image_uncertainty(boxes)

            valid_paths.append(batch_paths[k])
            image_uncertainties.append(u_img)
            embeddings_list.append(batch_features[k])

    print("\n>> Applying Stage 3: Difficulty Calibrated Uncertainty Sampling (DCUS)..")
    image_uncertainties = np.array(image_uncertainties)
    sorted_indices = np.argsort(image_uncertainties)[::-1]

    candidate_pool_size = min(2000, len(valid_paths))
    candidate_indices = sorted_indices[:candidate_pool_size]
    print(f"Filtered mass pool down to Top {candidate_pool_size} candidate images.")

    print(
        f"\n>> Applying Stage 4: Diversity Final Selection on remaining {candidate_pool_size} images using KMeans++..."
    )
    candidate_embeddings = np.array([embeddings_list[i] for i in candidate_indices])

    selected_subset = diversity_sampling(
        candidate_embeddings, candidate_indices, n_samples=BUDGET_PER_CYCLE
    )
    selected_paths = [valid_paths[i] for i in selected_subset]

    candidate_csv_path = os.path.join(
        cycle_dir, f"al_query_candidates_{args.mode}_cycle_{cycle}.csv"
    )
    write_candidates_csv(selected_paths, candidate_csv_path)

    state["cycle"] += 1
    save_state(state)


if __name__ == "__main__":
    main()
