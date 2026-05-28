#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np

PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPELINES_DIR not in sys.path:
    sys.path.append(PIPELINES_DIR)

# Import central configurations
from config import DEFAULT_IOU_THRESHOLD, DEFAULT_OCCURRENCE_THRESHOLD


def extract_camera_id(subfolder):
    """
    Extracts the camera ID (top-level directory name) from a subfolder path string.
    Example: '6R/108MEDIA' -> '6R'
    """
    if pd.isna(subfolder):
        return "unknown"
    parts = str(subfolder).split("/")
    return parts[0]


def filter_static_detections(df, iou_threshold=0.5, occurrence_threshold=5):
    """
    Identifies and removes bounding boxes that trigger repeatedly in the same
    spatial location across fixed camera stations (static triggers).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns: 'subfolder', 'xmin', 'ymin', 'xmax', 'ymax'
    iou_threshold : float
        Intersection-over-Union threshold above which overlapping boxes are clustered.
    occurrence_threshold : int
        Maximum triggers allowed for a spatial cluster before it is suppressed.

    Returns:
    --------
    filtered_df : pd.DataFrame
        DataFrame with static triggers removed.
    removed_count : int
        Number of bounding boxes suppressed.
    """
    if len(df) == 0:
        return df, 0

    df = df.copy()
    df["camera_id"] = df["subfolder"].apply(extract_camera_id)

    indices_to_remove = set()

    # Group by camera station (since camera locations are fixed)
    for cam_id, group in df.groupby("camera_id"):
        if cam_id in ["unknown", "Observed", "Seen"]:
            continue

        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values
        indices = group.index.values
        n = len(boxes)

        clusters = []  # list of dicts: {'rep': [xmin, ymin, xmax, ymax], 'indices': []}

        for i in range(n):
            box = boxes[i]
            idx = indices[i]

            matched = False
            for cluster in clusters:
                rep = cluster["rep"]

                # Spatial IoU Calculation
                inter_x1 = max(box[0], rep[0])
                inter_y1 = max(box[1], rep[1])
                inter_x2 = min(box[2], rep[2])
                inter_y2 = min(box[3], rep[3])

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (rep[2] - rep[0]) * (rep[3] - rep[1])
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou >= iou_threshold:
                    cluster["indices"].append(idx)
                    m = len(cluster["indices"])
                    cluster["rep"] = [
                        (rep[0] * (m - 1) + box[0]) / m,
                        (rep[1] * (m - 1) + box[1]) / m,
                        (rep[2] * (m - 1) + box[2]) / m,
                        (rep[3] * (m - 1) + box[3]) / m,
                    ]
                    matched = True
                    break
            if not matched:
                clusters.append({"rep": list(box), "indices": [idx]})

        # Suppress static background triggers
        for cluster in clusters:
            count = len(cluster["indices"])
            if count > occurrence_threshold:
                print(
                    f"  [Filtered] Camera '{cam_id}': Box {np.round(cluster['rep'], 1)} triggered {count} times."
                )
                indices_to_remove.update(cluster["indices"])

    filtered_df = df.drop(index=list(indices_to_remove))
    filtered_df = filtered_df.drop(columns=["camera_id"])
    return filtered_df, len(indices_to_remove)


def main():
    parser = argparse.ArgumentParser(
        description="Filter out static background false positive detections (static triggers) from camera traps."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input prediction CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save the filtered CSV (defaults to input_name_filtered.csv).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Intersection-over-Union threshold for spatial clustering of boxes (default: {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--occurrence_threshold",
        type=int,
        default=DEFAULT_OCCURRENCE_THRESHOLD,
        help=f"Maximum triggers allowed per bounding box cluster before suppression (default: {DEFAULT_OCCURRENCE_THRESHOLD}).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist.")
        return

    print("==================================================================")
    print(f"RUNNING STATIC BOUNDING BOX FILTER")
    print(f"  Input File:   {args.input_csv}")
    print(f"  IoU Thresh:   {args.iou_threshold}")
    print(f"  Max Triggers: {args.occurrence_threshold}")
    print("==================================================================")

    df = pd.read_csv(args.input_csv)
    original_count = len(df)

    filtered_df, removed_count = filter_static_detections(
        df, args.iou_threshold, args.occurrence_threshold
    )

    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        output_path = f"{base}_filtered{ext}"
    else:
        output_path = args.output_csv

    filtered_df.to_csv(output_path, index=False)

    print("\n------------------------------------------------------------------")
    print(f"Filtering complete:")
    print(f"  Original predictions:      {original_count}")
    print(
        f"  Suppressed static boxes:   {removed_count} ({removed_count / original_count:.1% if original_count > 0 else 0})"
    )
    print(f"  Cleaned predictions saved: {len(filtered_df)}")
    print(f"  Destination file:          {output_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
