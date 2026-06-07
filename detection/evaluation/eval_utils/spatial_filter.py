import os
import re
from collections import defaultdict
import numpy as np
from scipy.spatial import KDTree
from eval_utils.config import (
    POST_PROCESS_IOU_THRESHOLD,
    POST_PROCESS_OCCURRENCE_THRESHOLD,
    MIN_CONF_THRESHOLD,
)

CAMERA_PATTERN = re.compile(r"^\d[A-Za-z]$")


def apply_spatial_filter(
    results, return_indices_only=False, min_conf_threshold=MIN_CONF_THRESHOLD
):
    """
    Suppresses bounding boxes that trigger repeatedly in the same spatial location
    across fixed camera stations in the same year (static triggers).
    Uses a pure dynamic grid index to achieve O(1) clustering complexity per box.
    """
    if not results:
        if return_indices_only:
            return defaultdict(set), 0
        return results

    # Group predictions by (camera_id, year, class)
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

        for pred_idx, pred in enumerate(res.get("predictions", [])):
            if pred.get("conf", 1.0) < min_conf_threshold:
                continue

            cls_id = pred["cls"]
            group_key = (camera, year, cls_id)
            groups[group_key].append(
                {"img_idx": img_idx, "pred_idx": pred_idx, "bbox": pred["bbox"]}
            )

    indices_to_remove = defaultdict(set)  # img_idx -> set of pred_idx to remove
    cell_size = 0.05

    # Run spatial IoU clustering per group using pure Dynamic Grid (Spatial Hashing)
    for group_key, group_preds in groups.items():
        if not group_preds:
            continue

        if len(group_preds) <= POST_PROCESS_OCCURRENCE_THRESHOLD:
            continue

        clusters = []
        grid = defaultdict(list)

        for idx, pred in enumerate(group_preds):
            box = pred["bbox"]
            cx1, cy1, w1, h1 = box
            area1 = w1 * h1

            candidates = []

            # Query pure dynamic grid
            gx_cell = int(max(0.0, min(1.0, cx1)) / cell_size)
            gy_cell = int(max(0.0, min(1.0, cy1)) / cell_size)

            # Check a 3x3 neighborhood
            for gx in range(max(0, gx_cell - 1), min(20, gx_cell + 2)):
                for gy in range(max(0, gy_cell - 1), min(20, gy_cell + 2)):
                    if (gx, gy) in grid:
                        candidates.extend(grid[(gx, gy)])

            # Sort by ID to ensure deterministic behavior
            candidates.sort(key=lambda c: c["id"])

            matched = False
            for cluster in candidates:
                area2 = cluster["rep_area"]
                if area1 < 0.8 * area2 or area2 < 0.8 * area1:
                    continue

                rep = cluster["rep"]
                cx2, cy2, w2, h2 = rep

                # Center shift bounds checks
                max_w = w1 if w1 > w2 else w2
                limit_x = (w1 + w2) * 0.5 - POST_PROCESS_IOU_THRESHOLD * max_w
                if abs(cx1 - cx2) > limit_x:
                    continue

                max_h = h1 if h1 > h2 else h2
                limit_y = (h1 + h2) * 0.5 - POST_PROCESS_IOU_THRESHOLD * max_h
                if abs(cy1 - cy2) > limit_y:
                    continue

                # Spatial IoU Calculation
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

                        old_rep = rep
                        new_rep = [(rep[k] * (m - 1) + box[k]) / m for k in range(4)]
                        cluster["rep"] = new_rep
                        cluster["rep_area"] = new_rep[2] * new_rep[3]

                        # Update grid registration if rep shifted cells
                        old_gx = int(max(0.0, min(1.0, old_rep[0])) / cell_size)
                        old_gy = int(max(0.0, min(1.0, old_rep[1])) / cell_size)
                        new_gx = int(max(0.0, min(1.0, new_rep[0])) / cell_size)
                        new_gy = int(max(0.0, min(1.0, new_rep[1])) / cell_size)
                        if (old_gx, old_gy) != (new_gx, new_gy):
                            if cluster in grid[(old_gx, old_gy)]:
                                grid[(old_gx, old_gy)].remove(cluster)
                            grid[(new_gx, new_gy)].append(cluster)
                    else:
                        indices_to_remove[pred["img_idx"]].add(pred["pred_idx"])

                    matched = True
                    break

            if not matched:
                new_cluster = {
                    "id": len(clusters),
                    "rep": list(box),
                    "rep_area": area1,
                    "items": [pred],
                }
                clusters.append(new_cluster)
                gx = int(max(0.0, min(1.0, cx1)) / cell_size)
                gy = int(max(0.0, min(1.0, cy1)) / cell_size)
                grid[(gx, gy)].append(new_cluster)

        # Gather final indices to remove from all clusters
        for cluster in clusters:
            if len(cluster["items"]) > POST_PROCESS_OCCURRENCE_THRESHOLD:
                for item in cluster["items"]:
                    indices_to_remove[item["img_idx"]].add(item["pred_idx"])

    total_preds_before = sum(len(res.get("predictions", [])) for res in results)
    total_removed = sum(len(s) for s in indices_to_remove.values())

    if total_removed > 0:
        print(
            f"  [Post-Processing] Suppressed {total_removed} static background false positive boxes out of {total_preds_before} total predictions."
        )

    if return_indices_only:
        return indices_to_remove, total_removed

    # Reconstruct results list without suppressed predictions in-place
    for img_idx, res in enumerate(results):
        remove_set = indices_to_remove.get(img_idx)

        if remove_set and "predictions" in res:
            new_preds = []
            for pred_idx, pred in enumerate(res["predictions"]):
                if pred_idx not in remove_set:
                    new_preds.append(pred)
            res["predictions"] = new_preds

    return results
