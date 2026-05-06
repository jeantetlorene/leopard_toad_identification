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

    label_map = load_labels(os.path.join(ds_info["gt_dir"], "labels"))
    results = existing_results if existing_results else []

    if not all_images:
        print(f"All images already processed for {dataset_name}.")
        return results

    print(
        f"Generating predictions on {dataset_name} ({len(all_images)} images, CLAHE={use_clahe})"
    )

    def load_and_prep(p):
        im = cv2.imread(p)
        if im is None:
            return None, p
        if use_clahe:
            im = apply_clahe(im)
        return im, p

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for i in tqdm(range(0, len(all_images), batch_size)):
            batch_paths = all_images[i : i + batch_size]

            # Load and preprocess in parallel
            imgs = []
            valid_paths = []

            for im, p in executor.map(load_and_prep, batch_paths):
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

            # Incrementally save every 50 batches if output_file is provided
            if output_file and (i // batch_size) % 50 == 0:
                import json

                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)

    # Final save
    if output_file:
        import json

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results
