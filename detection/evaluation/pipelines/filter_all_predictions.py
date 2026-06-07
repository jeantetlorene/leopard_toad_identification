import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, FILTERED_FILE_SUFFIX
import eval_utils.config as config
from eval_utils.data_utils import load_predictions_from_json
from eval_utils.spatial_filter import apply_spatial_filter


import ijson
import orjson

import ijson
import orjson


def process_file(raw_file):
    # Force load_predictions_from_json to load raw and run the filter
    config.USE_FILTERED_PREDICTIONS = False

    filtered_file = raw_file.replace("_raw.json", FILTERED_FILE_SUFFIX)
    is_full_seq = "full_seq" in raw_file

    try:
        t_start = time.time()

        # Step 1: Stream extract lightweight predictions
        t_pass1_start = time.time()
        from eval_utils.data_utils import get_clean_ground_truth

        lightweight_results = []

        with open(raw_file, "rb") as f:
            for img_idx, item in enumerate(ijson.items(f, "item", use_float=True)):
                preds = item.get("predictions", [])
                if not preds:
                    continue  # Skip empty images for spatial filter building

                # Keep lightweight dict for spatial filtering
                lightweight_results.append(
                    {
                        "img_idx": img_idx,
                        "path": item.get("path", ""),
                        "predictions": preds,
                    }
                )

        t_pass1 = time.time() - t_pass1_start

        # Step 2: Spatial Filtering
        t_filter_start = time.time()
        # Compute indices to remove using the new return_indices_only flag
        indices_to_remove, total_removed = apply_spatial_filter(
            lightweight_results, return_indices_only=True
        )
        t_filter = time.time() - t_filter_start

        # Step 3: Stream write the filtered file and apply GT refresh
        t_pass2_start = time.time()
        with open(raw_file, "rb") as f_in, open(filtered_file, "wb") as f_out:
            f_out.write(b"[")
            first = True
            for img_idx, item in enumerate(ijson.items(f_in, "item", use_float=True)):
                # Refresh GT purely per-image
                path = item.get("path", "")
                is_positive, gt_boxes, split = get_clean_ground_truth(path)
                if split is not None:
                    item["is_positive"] = is_positive
                    item["gt_boxes"] = gt_boxes
                elif is_full_seq:
                    item["is_positive"] = False
                    item["gt_boxes"] = []

                # Filter predictions if this image has suppression triggers
                remove_set = indices_to_remove.get(img_idx)
                if remove_set and "predictions" in item:
                    item["predictions"] = [
                        p
                        for p_idx, p in enumerate(item["predictions"])
                        if p_idx not in remove_set
                    ]

                if not first:
                    f_out.write(b",")
                f_out.write(orjson.dumps(item))
                first = False
            f_out.write(b"]")

        t_pass2 = time.time() - t_pass2_start

        total_time = time.time() - t_start
        print(
            f"\n[{os.path.basename(raw_file)}] "
            f"Pass1: {t_pass1:.2f}s | "
            f"Filter: {t_filter:.2f}s | "
            f"Pass2/Save: {t_pass2:.2f}s | "
            f"Total: {total_time:.2f}s"
        )

        return True, raw_file, None
    except Exception as e:
        import traceback

        traceback.print_exc()
        return False, raw_file, str(e)


def get_total_ram_gb():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    parts = line.split()
                    return int(parts[1]) / (1024 * 1024)
    except Exception:
        pass
    return 16.0  # fallback to 16GB


def main():
    try:
        cpu_count = os.cpu_count()
        default_workers = max(1, cpu_count - 2) if cpu_count else 2
    except AttributeError:
        default_workers = 2

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
        default=default_workers,
        help="Number of concurrent worker processes (default: cpu_count - 2, set to 1 for sequential)",
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
        # Run in parallel with dynamic memory-aware throttling to prevent OOM
        file_sizes = {f: os.path.getsize(f) for f in raw_files}
        system_memory = get_total_ram_gb() * 1024 * 1024 * 1024
        # Keep total active file sizes under ~4% of system memory to prevent RAM exhaustion.
        # Even with O(1) streaming, ijson buffers and lightweight prediction dicts can stack up across 18 workers.
        max_active_file_size = int(system_memory * 0.04)

        print(
            f"System memory: {system_memory / (1024**3):.1f} GB. Max parallel file size limit: {max_active_file_size / (1024**2):.1f} MB."
        )

        # Sort files by size ascending so pending_files.pop() yields the largest files first
        pending_files = sorted(raw_files, key=lambda f: file_sizes[f])

        active_futures = {}
        current_active_size = 0

        with ProcessPoolExecutor(
            max_workers=max_workers, max_tasks_per_child=1
        ) as executor:
            pbar = tqdm(total=len(raw_files), desc="Pre-filtering files")
            while pending_files or active_futures:
                # Submit jobs up to limit
                while (
                    pending_files
                    and len(active_futures) < max_workers
                    and (
                        current_active_size == 0
                        or current_active_size + file_sizes[pending_files[-1]]
                        <= max_active_file_size
                    )
                ):
                    next_file = pending_files.pop()
                    file_size = file_sizes[next_file]

                    future = executor.submit(process_file, next_file)
                    active_futures[future] = (next_file, file_size)
                    current_active_size += file_size

                if active_futures:
                    done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        filename, file_size = active_futures.pop(future)
                        current_active_size -= file_size
                        pbar.update(1)

                        try:
                            success, raw_file, err_msg = future.result()
                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                                print(f"\nError processing {raw_file}: {err_msg}")
                        except Exception as e:
                            error_count += 1
                            print(f"\nWorker crashed processing {filename}: {str(e)}")
            pbar.close()

    print(f"\nPre-filtering completed. Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
