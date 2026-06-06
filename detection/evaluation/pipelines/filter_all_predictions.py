import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, FILTERED_FILE_SUFFIX
import eval_utils.config as config
from eval_utils.data_utils import load_predictions_from_json


def process_file(raw_file):
    # Force load_predictions_from_json to load raw and run the filter
    config.USE_FILTERED_PREDICTIONS = False

    filtered_file = raw_file.replace("_raw.json", FILTERED_FILE_SUFFIX)
    is_full_seq = "full_seq" in raw_file

    try:
        filtered_results = load_predictions_from_json(raw_file, is_full_seq=is_full_seq)
        with open(filtered_file, "w") as f:
            json.dump(filtered_results, f)
        return True, raw_file, None
    except Exception as e:
        return False, raw_file, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter prediction files to suppress static background triggers."
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=["yolo", "rtdetr", "faster_rcnn"],
        help="Specific model family to filter (yolo, rtdetr, faster_rcnn)",
    )
    parser.add_argument(
        "--model-name",
        choices=["yolo", "rtdetr", "faster_rcnn"],
        help="Specific model family to filter (yolo, rtdetr, faster_rcnn)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered files instead of skipping them",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent worker processes (default: 2, set to 1 for sequential)",
    )
    args = parser.parse_args()

    model_filter = args.model or args.model_name

    print("Scanning results directory for prediction files...")
    raw_files = []
    for root, _, files in os.walk(RESULTS_DIR):
        for file in files:
            if file.endswith("_raw.json"):
                full_path = os.path.join(root, file)

                # Apply model filter if provided
                if model_filter:
                    normalized_path = full_path.lower()
                    if model_filter == "faster_rcnn":
                        if "faster_rcnn" not in normalized_path:
                            continue
                    elif model_filter == "rtdetr":
                        if "rtdetr" not in normalized_path:
                            continue
                    elif model_filter == "yolo":
                        if "yolo" not in normalized_path:
                            continue

                # Skip already completed files if overwrite is not set
                if not args.overwrite:
                    filtered_file = full_path.replace("_raw.json", FILTERED_FILE_SUFFIX)
                    if os.path.exists(filtered_file):
                        continue

                raw_files.append(full_path)

    print(f"Found {len(raw_files)} raw prediction files to process.")
    if not raw_files:
        print("No files to process. Exiting.")
        return

    max_workers = args.workers
    print(f"Starting processing using {max_workers} worker processes...")

    success_count = 0
    error_count = 0

    if max_workers <= 1:
        # Run sequentially (avoid multiprocessing completely, safe from memory limits)
        for raw_file in tqdm(raw_files, desc="Pre-filtering files"):
            success, raw_file, err_msg = process_file(raw_file)
            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"\nError processing {raw_file}: {err_msg}")
    else:
        # Run in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, raw_file): raw_file
                for raw_file in raw_files
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Pre-filtering files"
            ):
                success, raw_file, err_msg = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\nError processing {raw_file}: {err_msg}")

    print(f"\nPre-filtering completed. Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
