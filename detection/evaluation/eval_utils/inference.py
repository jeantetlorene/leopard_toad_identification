import os
import cv2

cv2.setNumThreads(0)
import concurrent.futures
from tqdm import tqdm
import threading
import json
import torch
from torch.utils.data import Dataset, DataLoader

from eval_utils.config import (
    DATASETS,
    DEFAULT_BATCH_SIZE,
    FASTER_RCNN_SUB_BATCH_SIZE,
)
from eval_utils.data_utils import (
    apply_clahe,
    get_camera_images,
    get_ground_truth_positives,
    get_dataset_images,
    get_clean_ground_truth,
)


class ImageDataset(Dataset):
    def __init__(self, paths, use_clahe=False):
        self.paths = paths
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        if im is None:
            return None, path
        if self.use_clahe:
            im = apply_clahe(im)
        return im, path


def custom_collate(batch):
    imgs = [item[0] for item in batch if item[0] is not None]
    paths = [item[1] for item in batch if item[0] is not None]
    return imgs, paths


def generate_predictions(
    model_wrapper,
    dataset_name,
    use_clahe=False,
    limit=None,
    batch_size=DEFAULT_BATCH_SIZE,
    full_sequence=False,
    processed_paths=None,
    output_file=None,
    existing_results=None,
):
    """
    Run inference on a dataset and return the raw prediction results.
    Does NOT calculate evaluation metrics.
    """
    ds_info = DATASETS[dataset_name]
    positives = get_ground_truth_positives(dataset_name)

    if full_sequence:
        all_images = get_camera_images(ds_info["camera"])
    else:
        all_images = get_dataset_images(dataset_name)

    if limit:
        all_images = all_images[:limit]

    if processed_paths:
        all_images = [p for p in all_images if p not in processed_paths]

    results = existing_results if existing_results else []

    if not all_images:
        print(f"All images already processed for {dataset_name}.")
        return results

    print(
        f"Generating predictions on {dataset_name} ({len(all_images)} images, CLAHE={use_clahe})"
    )

    save_lock = threading.Lock()

    def async_save(data, path):
        if not save_lock.acquire(blocking=False):
            return  # Skip saving if another thread is already saving
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        finally:
            save_lock.release()

    # Dynamically scale workers to prevent over-prefetching overhead on small datasets
    max_workers = 16 if full_sequence else 4
    num_workers = min(max_workers, os.cpu_count() or 4)

    # Cap workers if we have fewer batches than workers
    num_batches = (len(all_images) + batch_size - 1) // batch_size
    num_workers = min(num_workers, max(1, num_batches))

    dataset = ImageDataset(all_images, use_clahe)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    for i, (imgs, valid_paths) in enumerate(tqdm(dataloader)):
        if not imgs:
            continue

        batch_preds = model_wrapper.predict_batch(
            imgs, sub_batch_size=FASTER_RCNN_SUB_BATCH_SIZE
        )

        for path, preds in zip(valid_paths, batch_preds):
            is_positive, gt_boxes, _ = get_clean_ground_truth(path)
            results.append(
                {
                    "path": path,
                    "is_positive": is_positive,
                    "gt_boxes": gt_boxes,
                    "predictions": preds,
                }
            )

        # Incrementally save every 50 batches if output_file is provided
        if output_file and i > 0 and i % 50 == 0:
            # Make a shallow copy of the list to avoid RuntimeError during iteration
            res_copy = list(results)
            threading.Thread(target=async_save, args=(res_copy, output_file)).start()

    # Final save
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f)

    return results
