import json
import csv
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Enforce working directory to be this folder so YOLO strictly downloads weights and creates any cache/runs here
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRETRAINED_WEIGHTS,
    SCRATCH_WEIGHTS,
    BUDGET_PER_CYCLE,
    CANDIDATE_OUTPUT_CSV,
    AL_STATE_JSON,
    BATCH_SIZE,
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


def write_candidates_csv(selected_paths):
    with open(CANDIDATE_OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "status"])
        for path in selected_paths:
            writer.writerow([path, "To annotate"])
    print(
        f"\n[ORACLE PAUSE] Exported {len(selected_paths)} queries to {CANDIDATE_OUTPUT_CSV}"
    )
    print(
        "Please annotate these images in Label Studio, move the YOLO-formatted labels back to your 'train' folder, and run this script again."
    )


def main():
    state = load_state()
    cycle = state["cycle"]

    print(f"\n{'=' * 50}\nStarting Active Learning Cycle {cycle}\n{'=' * 50}")

    pretrained_model = state["model_paths"].get("pretrained", PRETRAINED_WEIGHTS)
    scratch_model = state["model_paths"].get("scratch", SCRATCH_WEIGHTS)

    if cycle == 0:
        print(">> Cycle 0: Initial Training on Seed Dataset.")
        print("\n--- Scenario 1: Pretrained Model (Phased Unfreezing) ---")
        p1 = train_phase_1(
            pretrained_model,
            f"cycle_{cycle}_pretrained",
            freeze=15,
            epochs=100,
            patience=15,
        )
        # Few more epochs for Phase 2 unfrozen
        p2 = train_phase_2(p1, f"cycle_{cycle}_pretrained", epochs=15)
        pretrained_model = p2

        print("\n--- Scenario 2: From-Scratch Model ---")
        scratch_model = train_scratch(
            scratch_model, f"cycle_{cycle}_scratch", epochs=60
        )

        state["model_paths"]["pretrained"] = pretrained_model
        state["model_paths"]["scratch"] = scratch_model
        state["cycle"] += 1
        save_state(state)
        print(">> Initial Models Trained. Advancing to Cycle 1 for pool inference.")
        cycle = 1

    print(
        "\n>> Scanning for unlabelled images from /srv ... (Excluding 4R, 5Z val/test cameras)"
    )
    pool = get_unlabeled_pool()
    print(f"Found {len(pool)} unlabelled images in the massive pool.")

    if len(pool) == 0:
        print("No unlabeled images found. Exiting.")
        return

    # To honor performance and the entire pool, we process in chunks
    print(
        "\n>> Starting Inference & Feature Extraction (This will take considerable time on 1.4TB)"
    )
    active_model = YOLO(pretrained_model)
    dcus = DCUS()

    valid_paths = []
    image_uncertainties = []
    embeddings_list = []

    # Process in batches
    for i in tqdm(range(0, len(pool), BATCH_SIZE), desc="Inferring Pool"):
        chunk = pool[i : i + BATCH_SIZE]
        batch_paths, batch_boxes, batch_features = extract_features_and_boxes_batch(
            active_model, chunk
        )

        for k in range(len(batch_paths)):
            boxes = batch_boxes[k]
            u_img = dcus.image_uncertainty(boxes)

            # Store data
            valid_paths.append(batch_paths[k])
            image_uncertainties.append(u_img)
            embeddings_list.append(batch_features[k])

    print("\n>> Applying Stage 3: Difficulty Calibrated Uncertainty Sampling (DCUS)..")
    image_uncertainties = np.array(image_uncertainties)
    # Sort descending by uncertainty
    sorted_indices = np.argsort(image_uncertainties)[::-1]

    # Take the top N most uncertain candidates as our candidate pool
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

    write_candidates_csv(selected_paths)

    state["cycle"] += 1
    save_state(state)


if __name__ == "__main__":
    main()
