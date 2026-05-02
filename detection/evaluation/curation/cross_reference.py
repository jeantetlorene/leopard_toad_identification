import os
import csv
from pathlib import Path
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import math


def cluster_boxes(df_group, iou_threshold=0.5):
    """
    Cluster bounding boxes for a specific image and class based on IoU threshold.
    Returns a list of consensus prediction dictionaries.
    """
    # Sort by confidence descending to use highest confidence as seed
    df_group = df_group.sort_values(by="confidence", ascending=False).reset_index(
        drop=True
    )

    clusters = []

    while not df_group.empty:
        # Seed is the first row
        seed_row = df_group.iloc[0]
        seed_box = torch.tensor(
            [[seed_row["xmin"], seed_row["ymin"], seed_row["xmax"], seed_row["ymax"]]],
            dtype=torch.float32,
        )

        # All boxes in current pool
        all_boxes = torch.tensor(
            df_group[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
        )

        # Calculate IoU
        ious = torchvision.ops.box_iou(seed_box, all_boxes)[0]  # Shape: (N,)

        # Find matching boxes
        match_idx = (ious >= iou_threshold).numpy()

        cluster_df = df_group[match_idx]

        # Unique models
        unique_models = cluster_df["model_name"].unique().tolist()

        if len(unique_models) >= 3:
            # Valid cluster found
            mean_conf = float(cluster_df["confidence"].mean())
            # Binary entropy estimation
            entropy = -mean_conf * math.log(mean_conf + 1e-9) - (
                1 - mean_conf
            ) * math.log((1 - mean_conf) + 1e-9)

            # Calculate bounding box area variance
            areas = (cluster_df["xmax"] - cluster_df["xmin"]) * (
                cluster_df["ymax"] - cluster_df["ymin"]
            )
            bbox_var = float(areas.var(ddof=0)) if len(cluster_df) > 1 else 0.0

            clusters.append(
                {
                    "image_path": seed_row["image_path"],
                    "image_name": seed_row["image_name"],
                    "subfolder": seed_row["subfolder"],
                    "class_id": seed_row["class_id"],
                    "class_name": seed_row["class_name"],
                    "confidence": float(cluster_df["confidence"].max()),
                    "min_confidence": float(cluster_df["confidence"].min()),
                    "mean_confidence": mean_conf,
                    "entropy": entropy,
                    "bbox_variance": bbox_var,
                    "xmin": round(float(cluster_df["xmin"].mean()), 1),
                    "ymin": round(float(cluster_df["ymin"].mean()), 1),
                    "xmax": round(float(cluster_df["xmax"].mean()), 1),
                    "ymax": round(float(cluster_df["ymax"].mean()), 1),
                    "agreed_models_count": len(unique_models),
                    "agreed_models": ", ".join(sorted(unique_models)),
                }
            )

        # Remove matched boxes from pool
        df_group = df_group[~match_idx].reset_index(drop=True)

    return clusters


def process_group(group_data):
    """
    Worker function to process a single group to allow for multiprocessing.
    """
    group_df = group_data[1]
    return cluster_boxes(group_df, iou_threshold=0.5)


def cross_reference_split(split_name, evaluation_dir):
    """
    Cross reference predictions across models for a given split (e.g. 'val', 'test').
    """
    print(f"\nProcessing {split_name} split...")
    eval_dir_path = Path(evaluation_dir)

    # 1. Collect all csv files
    dfs = []
    model_dirs = [
        d
        for d in eval_dir_path.iterdir()
        if d.is_dir()
        and d.name != "consensus_predictions"
        and not d.name.startswith(".")
    ]

    for m_dir in model_dirs:
        csv_path = m_dir / f"{split_name}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df["model_name"] = m_dir.name
                    dfs.append(df)
            except pd.errors.EmptyDataError:
                pass

    if not dfs:
        print(f"No {split_name}.csv files found.")
        return

    all_preds = pd.concat(dfs, ignore_index=True)
    print(f"Total predictions loaded: {len(all_preds)} from {len(dfs)} models.")

    # Group by image path and class ID
    groups = list(all_preds.groupby(["image_path", "class_id"]))
    print(f"Unique (image, class) groups to process: {len(groups)}")

    consensus_results = []

    # Process groups using multiprocessing to speed things up
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(
            tqdm(
                executor.map(process_group, groups),
                total=len(groups),
                desc="Cross-referencing",
            )
        )

    for res in results:
        consensus_results.extend(res)

    print(f"Found {len(consensus_results)} consensus predictions.")

    if consensus_results:
        out_dir = eval_dir_path / "consensus_predictions"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{split_name}_consensus.csv"

        out_df = pd.DataFrame(consensus_results)
        # Sort for better readability
        out_df = out_df.sort_values(
            by=["agreed_models_count", "confidence"], ascending=[False, False]
        )
        out_df.to_csv(out_file, index=False)
        print(f"Saved consensus results to {out_file}")


def main():
    evaluation_dir = "/home/Joshua/Downloads/leopard_toad_identification/evaluation"

    cross_reference_split("val", evaluation_dir)
    cross_reference_split("test", evaluation_dir)


if __name__ == "__main__":
    main()
