import os
import re
from collections import defaultdict
from eval_utils.config import (
    POST_PROCESS_IOU_THRESHOLD,
    POST_PROCESS_OCCURRENCE_THRESHOLD,
    MIN_CONF_THRESHOLD,
)


CAMERA_PATTERN = re.compile(r"^\d[A-Za-z]$")


def apply_spatial_filter(results):
    """
    Suppresses bounding boxes that trigger repeatedly in the same spatial location
    across fixed camera stations in the same year (static triggers).
    Uses a spatial grid index (spatial hashing) to achieve O(N) complexity.
    """
    if not results:
        return results

    for res in results:
        res["predictions"] = [
            p for p in res["predictions"] if p["conf"] >= MIN_CONF_THRESHOLD
        ]

    # Group predictions by (camera_id, year, class)
    # A prediction item is: (image_index, prediction_index, bbox)
    groups = defaultdict(list)
    path_cache = {}  # dirname -> (camera, year)

    for img_idx, res in enumerate(results):
        path = res["path"]
        dirname = os.path.dirname(path)

        if dirname in path_cache:
            camera, year = path_cache[dirname]
        else:
            parts = os.path.normpath(path).split(os.sep)

            # Extract year
            year = "unknown_year"
            for part in parts:
                if len(part) == 4 and part.isdigit():
                    year = part
                    break

            # Extract camera ID
            camera = "unknown_camera"
            for part in parts:
                if CAMERA_PATTERN.match(part):
                    camera = part.upper()
                    break

            path_cache[dirname] = (camera, year)

        if camera in ["unknown_camera", "OBSERVED", "SEEN"]:
            continue

        for pred_idx, pred in enumerate(res["predictions"]):
            cls_id = pred["cls"]
            group_key = (camera, year, cls_id)
            groups[group_key].append(
                {"img_idx": img_idx, "pred_idx": pred_idx, "bbox": pred["bbox"]}
            )

    indices_to_remove = defaultdict(set)  # img_idx -> set of pred_idx to remove
    cell_size = 0.05  # Grid cell size for spatial hashing

    def get_overlapping_cells(box):
        cx, cy, w, h = box
        x_start = int(max(0.0, cx - w / 2) / cell_size)
        x_end = int(min(1.0, cx + w / 2) / cell_size)
        y_start = int(max(0.0, cy - h / 2) / cell_size)
        y_end = int(min(1.0, cy + h / 2) / cell_size)

        # Limit grid cell registration for large bounding boxes (e.g. background/faulty triggers)
        # to prevent memory explosion and slow loops.
        num_cells = (x_end - x_start + 1) * (y_end - y_start + 1)
        if num_cells > 16:
            cx_cell = int(max(0.0, min(1.0, cx)) / cell_size)
            cy_cell = int(max(0.0, min(1.0, cy)) / cell_size)
            max_grid_idx = int(1.0 / cell_size)
            return {
                (gx, gy)
                for gx in range(cx_cell - 1, cx_cell + 2)
                for gy in range(cy_cell - 1, cy_cell + 2)
                if 0 <= gx <= max_grid_idx and 0 <= gy <= max_grid_idx
            }

        return {
            (gx, gy)
            for gx in range(x_start, x_end + 1)
            for gy in range(y_start, y_end + 1)
        }

    # Run spatial IoU clustering per group
    for group_key, group_preds in groups.items():
        if not group_preds:
            continue

        # If a group has fewer predictions than the threshold, no static triggers can be established
        if len(group_preds) <= POST_PROCESS_OCCURRENCE_THRESHOLD:
            continue

        # Grid index: maps (gx, gy) -> list of cluster dictionaries
        grid = defaultdict(list)
        cluster_id_counter = 0

        for pred in group_preds:
            box = pred["bbox"]
            cx1, cy1, w1, h1 = box
            area1 = w1 * h1
            pred_cells = get_overlapping_cells(box)

            # Find candidate clusters from overlapping grid cells
            candidate_clusters = {}
            for cell in pred_cells:
                for cluster in grid[cell]:
                    candidate_clusters[cluster["id"]] = cluster

            matched = False
            for cluster in candidate_clusters.values():
                # 1. Quick area-based ratio check:
                # IoU <= min(A1, A2) / max(A1, A2). If area ratio is less than threshold, skip.
                area2 = cluster["rep_area"]
                if (
                    area1 < POST_PROCESS_IOU_THRESHOLD * area2
                    or area2 < POST_PROCESS_IOU_THRESHOLD * area1
                ):
                    continue

                # 2. Quick center/width/height shift bounds check:
                # For IoU >= T, the center shift in x must be <= (w1 + w2)/2 - T * max(w1, w2)
                rep = cluster["rep"]
                cx2, cy2, w2, h2 = rep
                max_w = w1 if w1 > w2 else w2
                limit_x = (w1 + w2) * 0.5 - POST_PROCESS_IOU_THRESHOLD * max_w
                if abs(cx1 - cx2) > limit_x:
                    continue

                max_h = h1 if h1 > h2 else h2
                limit_y = (h1 + h2) * 0.5 - POST_PROCESS_IOU_THRESHOLD * max_h
                if abs(cy1 - cy2) > limit_y:
                    continue

                # Spatial IoU Calculation (normalized cx, cy, w, h format)
                b1_x1, b1_y1 = cx1 - w1 * 0.5, cy1 - h1 * 0.5
                b1_x2, b1_y2 = cx1 + w1 * 0.5, cy1 + h1 * 0.5
                b2_x1, b2_y1 = cx2 - w2 * 0.5, cy2 - h2 * 0.5
                b2_x2, b2_y2 = cx2 + w2 * 0.5, cy2 + h2 * 0.5

                inter_x1 = max(b1_x1, b2_x1)
                inter_y1 = max(b1_y1, b2_y1)
                inter_x2 = min(b1_x2, b2_x2)
                inter_y2 = min(b1_y2, b2_y2)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0

                if iou >= POST_PROCESS_IOU_THRESHOLD:
                    m = len(cluster["items"])
                    if m <= POST_PROCESS_OCCURRENCE_THRESHOLD + 5:
                        cluster["items"].append(pred)
                        m += 1
                        # Update running average representative
                        new_rep = [(rep[k] * (m - 1) + box[k]) / m for k in range(4)]
                        cluster["rep"] = new_rep
                        cluster["rep_area"] = new_rep[2] * new_rep[3]

                        # Update grid registration if cells changed
                        new_cells = get_overlapping_cells(new_rep)
                        old_cells = cluster["cells"]
                        if new_cells != old_cells:
                            # Remove from cells no longer overlapped
                            for cell in old_cells - new_cells:
                                if cluster in grid[cell]:
                                    grid[cell].remove(cluster)
                            # Add to new cells overlapped
                            for cell in new_cells - old_cells:
                                grid[cell].append(cluster)
                            cluster["cells"] = new_cells
                    else:
                        # Already established static trigger; suppress immediately
                        indices_to_remove[pred["img_idx"]].add(pred["pred_idx"])

                    matched = True
                    break
            if not matched:
                new_cluster = {
                    "id": cluster_id_counter,
                    "rep": list(box),
                    "rep_area": area1,
                    "items": [pred],
                    "cells": pred_cells,
                }
                cluster_id_counter += 1
                for cell in pred_cells:
                    grid[cell].append(new_cluster)

        # Gather all unique clusters from the grid to identify static triggers
        all_clusters = {}
        for cell, clusters_list in grid.items():
            for cluster in clusters_list:
                all_clusters[cluster["id"]] = cluster

        # Suppress static triggers
        for cluster in all_clusters.values():
            count = len(cluster["items"])
            if count > POST_PROCESS_OCCURRENCE_THRESHOLD:
                # Add all prediction indices in this cluster to removal set
                for item in cluster["items"]:
                    indices_to_remove[item["img_idx"]].add(item["pred_idx"])

    # Reconstruct results list without suppressed predictions in-place
    total_removed = 0
    total_preds_before = 0

    for img_idx, res in enumerate(results):
        total_preds_before += len(res["predictions"])
        remove_set = indices_to_remove.get(img_idx)

        if remove_set:
            new_preds = []
            for pred_idx, pred in enumerate(res["predictions"]):
                if pred_idx not in remove_set:
                    new_preds.append(pred)
                else:
                    total_removed += 1
            res["predictions"] = new_preds

    if total_removed > 0:
        print(
            f"  [Post-Processing] Suppressed {total_removed} static background false positive boxes out of {total_preds_before} total predictions."
        )

    return results
