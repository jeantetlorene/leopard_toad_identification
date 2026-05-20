import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from eval_utils.config import RESULTS_DIR, DEVICE
from eval_utils.inference import generate_predictions
from eval_utils.models.megadetector_wrapper import MegaDetectorWrapper


def main():
    import json

    model_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation/eval_utils/models/weights/md_v5a.0.0.pt"

    if not os.path.exists(model_path):
        print(f"Error: MegaDetector weights not found at {model_path}")
        return

    wrapper = MegaDetectorWrapper(model_path, device=DEVICE)

    res_dir = os.path.join(RESULTS_DIR, "megadetector_plain")
    os.makedirs(res_dir, exist_ok=True)

    # We use "test_full_seq" dataset name for the inference logic in generate_predictions
    raw_file = os.path.join(res_dir, "cycle_0_pretrained_test_full_seq_raw.json")

    existing_results = []
    processed_paths = set()

    if os.path.exists(raw_file):
        print(f"Found existing predictions for {raw_file}, attempting to resume...")
        try:
            with open(raw_file, "r") as f:
                existing_results = json.load(f)
                processed_paths = {res["path"] for res in existing_results}
        except (json.JSONDecodeError, KeyError):
            print("Failed to read existing JSON, starting fresh.")
            existing_results = []
            processed_paths = set()

    generate_predictions(
        wrapper,
        "test",  # The dataset split
        use_clahe=False,
        full_sequence=True,  # Entire test pool
        batch_size=16,  # Reduced batch size to prevent System RAM OOM from dataloader prefetching
        processed_paths=processed_paths,
        output_file=raw_file,
        existing_results=existing_results,
    )


if __name__ == "__main__":
    main()
