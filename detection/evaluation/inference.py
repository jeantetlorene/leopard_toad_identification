import os
import cv2
import concurrent.futures
from tqdm import tqdm

from config import (
    DATASETS,
    DEFAULT_BATCH_SIZE,
    FASTER_RCNN_SUB_BATCH_SIZE,
)
from data_utils import (
    apply_clahe,
    load_labels,
    get_best_label_match,
    get_camera_images,
    get_ground_truth_positives,
    get_dataset_images,
)


def generate_predictions(
    model_wrapper,
    dataset_name,
    use_clahe=False,
    limit=None,
    batch_size=DEFAULT_BATCH_SIZE,
    full_sequence=False,
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

    label_map = load_labels(os.path.join(ds_info["gt_dir"], "labels"))
    results = []

    print(
        f"Generating predictions on {dataset_name} ({len(all_images)} images, CLAHE={use_clahe})"
    )

    for i in tqdm(range(0, len(all_images), batch_size)):
        batch_paths = all_images[i : i + batch_size]

        # Load and preprocess in parallel
        imgs = []
        valid_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

            def load_and_prep(p):
                im = cv2.imread(p)
                if im is None:
                    return None
                if use_clahe:
                    im = apply_clahe(im)
                return im

            futures = {executor.submit(load_and_prep, p): p for p in batch_paths}
            for future in concurrent.futures.as_completed(futures):
                p = futures[future]
                im = future.result()
                if im is not None:
                    imgs.append(im)
                    valid_paths.append(p)

        if not imgs:
            continue

        batch_preds = model_wrapper.predict_batch(
            imgs, sub_batch_size=FASTER_RCNN_SUB_BATCH_SIZE
        )

        for path, preds in zip(valid_paths, batch_preds):
            is_positive = path in positives
            results.append(
                {
                    "path": path,
                    "is_positive": is_positive,
                    "gt_boxes": get_best_label_match(path, label_map)
                    if is_positive
                    else [],
                    "predictions": preds,
                }
            )

    return results
