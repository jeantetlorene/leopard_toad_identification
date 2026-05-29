#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import cv2
from tqdm import tqdm
import concurrent.futures

# Set sys.path so we can import eval_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from eval_utils.config import CLAHE_PREPROCESSED_DIR, DATASETS
from eval_utils.data_utils import get_camera_images, apply_clahe

# Base source path
SRC_ROOT = "/srv/shared_leopard_toad"


def process_single_image(args):
    """
    Worker function to process a single image.
    args: (src_path, dst_path, overwrite)
    """
    src_path, dst_path, overwrite = args
    try:
        # Check if already processed
        if not overwrite and os.path.exists(dst_path):
            return True, "skipped"

        # Read original image
        im = cv2.imread(src_path)
        if im is None:
            return False, f"Failed to read image: {src_path}"

        # Apply CLAHE
        im_clahe = apply_clahe(im)

        # Ensure destination folder exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Write CLAHE-processed image
        cv2.imwrite(dst_path, im_clahe)
        return True, "success"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Offline CLAHE Preprocessing for Full Camera Datasets"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Target cameras to preprocess (e.g. 4R 5Z). If provided, overrides --split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "both"],
        default="both",
        help="Target dataset split cameras to preprocess: 'test' (5Z), 'val' (4R), or 'both' (4R and 5Z).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images processed per camera (useful for testing/dry-runs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Defaults to all CPU cores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite already preprocessed images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl and list images without performing preprocessing.",
    )

    args = parser.parse_args()

    # Determine cameras to process based on split / cameras options
    if args.cameras:
        target_cameras = args.cameras
    else:
        if args.split == "test":
            target_cameras = [DATASETS["test"]["camera"]]
        elif args.split == "val":
            target_cameras = [DATASETS["val"]["camera"]]
        else:  # both
            target_cameras = [DATASETS["val"]["camera"], DATASETS["test"]["camera"]]

    print("=================================================================")
    print("      OFFLINE CLAHE PREPROCESSING PIPELINE FOR CAMERA SETS      ")
    print("=================================================================")
    print(f"Source Root Path: {SRC_ROOT}")
    print(f"Target CLAHE Path: {CLAHE_PREPROCESSED_DIR}")
    print(
        f"Selected Split: {args.split if not args.cameras else 'N/A (overridden by --cameras)'}"
    )
    print(f"Target Cameras: {target_cameras}")
    print(f"Overwrite Existing: {args.overwrite}")
    print(f"Dry Run: {args.dry_run}")
    print("=================================================================\n")

    # Ensure output base directory exists
    os.makedirs(CLAHE_PREPROCESSED_DIR, exist_ok=True)

    # 1. Discover all images to process
    jobs = []
    total_images_found = 0
    already_done_count = 0

    for camera in target_cameras:
        print(f"Crawling images for camera {camera}...")
        camera_images = get_camera_images(camera, root_path=SRC_ROOT)
        print(f"Found {len(camera_images)} images for camera {camera}.")

        if args.limit:
            print(f"Limiting to first {args.limit} images for camera {camera}.")
            camera_images = camera_images[: args.limit]

        total_images_found += len(camera_images)

        for src_path in camera_images:
            # Map source path under SRC_ROOT to parallel path under CLAHE_PREPROCESSED_DIR
            norm_src = os.path.normpath(src_path)
            # Find relative path from SRC_ROOT
            rel_path = os.path.relpath(norm_src, SRC_ROOT)
            dst_path = os.path.join(CLAHE_PREPROCESSED_DIR, rel_path)

            if not args.overwrite and os.path.exists(dst_path):
                already_done_count += 1
                continue

            jobs.append((norm_src, dst_path, args.overwrite))

    print(f"\nTotal images discovered: {total_images_found}")
    if already_done_count > 0:
        print(f"Already preprocessed (skipping): {already_done_count}")
    print(f"Remaining images to process: {len(jobs)}")

    if args.dry_run:
        print("Dry run completed. No files were written.")
        return

    if not jobs:
        print("No images found to process. Exiting.")
        return

    # 2. Run parallel execution using ProcessPoolExecutor
    num_workers = args.workers or os.cpu_count() or 1
    print(f"Launching pool with {num_workers} parallel workers...")

    success_count = 0
    skipped_count = already_done_count
    failed_count = 0
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Wrap execution with tqdm for a beautiful progress bar
        results = list(
            tqdm(
                executor.map(process_single_image, jobs),
                total=len(jobs),
                desc="Preprocessing CLAHE",
            )
        )

        for (success, msg), (src_path, _, _) in zip(results, jobs):
            if success:
                if msg == "skipped":
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_count += 1
                failures.append((src_path, msg))

    print("\n=================================================================")
    print("                      PREPROCESSING COMPLETE                     ")
    print("=================================================================")
    print(f"Successfully Preprocessed: {success_count}")
    print(f"Skipped (Already Exists):   {skipped_count}")
    print(f"Failed to Process:         {failed_count}")
    print("=================================================================")

    if failures:
        print("\nFailures encountered:")
        for path, err in failures[:10]:
            print(f" - {path}: {err}")
        if len(failures) > 10:
            print(f" ... and {len(failures) - 10} more failures.")


if __name__ == "__main__":
    main()
