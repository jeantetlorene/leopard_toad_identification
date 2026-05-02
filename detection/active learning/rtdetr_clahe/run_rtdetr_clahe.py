import os
import argparse
import sys

# Change working dir to ensure ultralytics operates properly relative to this dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR, PRETRAINED_WEIGHTS, SCRATCH_WEIGHTS, RTDETR_DIR
from trainer_clahe import train_phase_1, train_phase_2, train_scratch


def run_training_for_mode(mode):
    print(f"\n{'=' * 50}\nStarting RT-DETR CLAHE Training [{mode.upper()}]\n{'=' * 50}")

    data_dir = os.path.join(BASE_DIR, "active learning", "data", "rtdetr", mode)
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Skipping.")
        return

    cycles = sorted([d for d in os.listdir(data_dir) if d.startswith("cycle_")])
    if not cycles:
        print(f"No cycles found in {data_dir}. Skipping.")
        return

    for cycle_folder in cycles:
        cycle_num = cycle_folder.split("_")[1]
        print(f"\n>> Cycle {cycle_num}: Training RT-DETR Model with CLAHE.")

        cycle_dir = os.path.join(RTDETR_DIR, "cycles", mode, f"cycle_{cycle_num}")
        os.makedirs(cycle_dir, exist_ok=True)

        dataset_dir = os.path.join(data_dir, cycle_folder)
        dataset_yaml = os.path.join(cycle_dir, f"dataset_{mode}_cycle_{cycle_num}.yaml")

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

        if mode == "pretrained":
            expected_p2_model = os.path.join(
                RTDETR_DIR,
                "runs",
                f"cycle_{cycle_num}_pretrained_phase2",
                "weights",
                "best.pt",
            )
            if os.path.exists(expected_p2_model):
                print(
                    f"  Found existing Phase 2 trained model for Cycle {cycle_num}. Skipping."
                )
            else:
                print("\n--- Pretrained Model (Phased Unfreezing) ---")
                expected_p1_model = os.path.join(
                    RTDETR_DIR,
                    "runs",
                    f"cycle_{cycle_num}_pretrained_phase1",
                    "weights",
                    "best.pt",
                )
                if os.path.exists(expected_p1_model):
                    print(f"  Found existing Phase 1 trained model for Cycle {cycle_num}. Skipping Phase 1.")
                    p1 = expected_p1_model
                else:
                    p1 = train_phase_1(
                        PRETRAINED_WEIGHTS,
                        f"cycle_{cycle_num}_pretrained",
                        dataset_yaml,
                        freeze=15,
                        epochs=100,
                        patience=25,
                    )
                p2 = train_phase_2(
                    p1, f"cycle_{cycle_num}_pretrained", dataset_yaml, epochs=30
                )
        else:
            expected_scratch_model = os.path.join(
                RTDETR_DIR,
                "runs",
                f"cycle_{cycle_num}_scratch_scratch",
                "weights",
                "best.pt",
            )
            if os.path.exists(expected_scratch_model):
                print(
                    f"  Found existing trained model for Cycle {cycle_num}. Skipping."
                )
            else:
                print("\n--- From-Scratch Model ---")
                train_scratch(
                    SCRATCH_WEIGHTS,
                    f"cycle_{cycle_num}_scratch",
                    dataset_yaml,
                    epochs=60,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["pretrained", "scratch", "both"], default="both"
    )
    args = parser.parse_args()

    if args.mode in ["pretrained", "both"]:
        run_training_for_mode("pretrained")
    if args.mode in ["scratch", "both"]:
        run_training_for_mode("scratch")
