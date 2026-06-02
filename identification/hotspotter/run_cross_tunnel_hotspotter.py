"""
Run cross-tunnel toad re-identification (matching Z vs R cameras within the same year).
"""

import os
import re
import csv
import pandas as pd
from tqdm import tqdm

from matcher import BatchHotSpotter
import config


def classify_camera(path):
    """
    Classify whether the image belongs to a 'Z' or 'R' end camera of a tunnel.
    """
    path_lower = path.lower()

    # Custom rule for Observed/Seen DSCF08xx.JPG files which belong to 4Z
    if "observed/seen" in path_lower and "dscf08" in path_lower:
        return "Z"

    parts = path_lower.split("/")
    for part in parts:
        if re.search(r"\b\d+z", part):
            return "Z"
        if re.search(r"\b\d+r", part):
            return "R"

    # Try search anywhere in the path
    if re.search(r"\d+z", path_lower):
        return "Z"
    if re.search(r"\d+r", path_lower):
        return "R"

    return "Unknown"


def extract_year(path):
    """
    Extract the year (2023, 2024, or 2025) from the image path or filename.
    """
    match = re.search(r"\b(2023|2024|2025)\b", path)
    if match:
        return match.group(1)
    return "Unknown"


def main():
    print("Loading predictions and building trackable name mappings...")
    df_wlt = pd.read_csv(config.PREDICTIONS_CSV)
    trackable_to_path = {}
    for _, row in df_wlt.iterrows():
        img_path = row["image_path"]
        ref_prefix = "/srv/shared_leopard_toad/"
        if ref_prefix in img_path:
            rel_path = img_path.split(ref_prefix, 1)[1]
        else:
            rel_path = img_path.lstrip("/")
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        trackable_name = rel_path_no_ext.replace("/", "__")
        trackable_to_path[trackable_name] = img_path

    # List all crop files in the predictions crops folder
    crops_dir = config.CROPS_DIR
    crop_files = [
        f
        for f in os.listdir(crops_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Found {len(crop_files)} crops in the folder: {crops_dir}")

    # Classify crops into Z and R camera groups by Year
    z_crops_by_year = {"2023": [], "2024": [], "2025": []}
    r_crops_by_year = {"2023": [], "2024": [], "2025": []}

    for fname in crop_files:
        # Reconstruct the trackable name by removing '_cropX.ext'
        trackable_name = fname.rsplit("_crop", 1)[0]
        original_path = trackable_to_path.get(trackable_name, "")

        if not original_path:
            camera = classify_camera(fname)
            year = extract_year(fname)
        else:
            camera = classify_camera(original_path)
            year = extract_year(original_path)

        if year not in ["2023", "2024", "2025"]:
            print(
                f"Warning: Could not classify year for: {fname} (path: {original_path})"
            )
            continue

        if camera == "Z":
            z_crops_by_year[year].append((fname, original_path))
        elif camera == "R":
            r_crops_by_year[year].append((fname, original_path))
        else:
            print(
                f"Warning: Could not classify camera for: {fname} (path: {original_path})"
            )

    for y in ["2023", "2024", "2025"]:
        print(
            f"Year {y}: Z camera crops = {len(z_crops_by_year[y])}, "
            f"R camera crops = {len(r_crops_by_year[y])}"
        )

    # Initialize Hotspotter matcher
    hotspotter = BatchHotSpotter()

    # Extract SIFT features grouped by year
    print("\nExtracting SIFT features for Z crops...")
    z_features_by_year = {"2023": {}, "2024": {}, "2025": {}}
    for y in ["2023", "2024", "2025"]:
        for fname, path in tqdm(z_crops_by_year[y], desc=f"SIFT Z-crops {y}"):
            img_path = os.path.join(crops_dir, fname)
            kps, des = hotspotter.get_features(img_path)
            if kps is not None and des is not None:
                z_features_by_year[y][fname] = (kps, des)

    print("\nExtracting SIFT features for R crops...")
    r_features_by_year = {"2023": {}, "2024": {}, "2025": {}}
    for y in ["2023", "2024", "2025"]:
        for fname, path in tqdm(r_crops_by_year[y], desc=f"SIFT R-crops {y}"):
            img_path = os.path.join(crops_dir, fname)
            kps, des = hotspotter.get_features(img_path)
            if kps is not None and des is not None:
                r_features_by_year[y][fname] = (kps, des)

    # Perform cross-tunnel matching (Z crops vs R crops) restricted to same-year
    print("\nPerforming cross-tunnel matching within same years...")
    possible_matches = []

    for y in ["2023", "2024", "2025"]:
        z_keys = list(z_features_by_year[y].keys())
        r_keys = list(r_features_by_year[y].keys())

        print(
            f"Matching Year {y}: {len(z_keys)} Z vs {len(r_keys)} R "
            f"({len(z_keys) * len(r_keys)} pairs)"
        )

        z_fname_to_path = {fname: path for fname, path in z_crops_by_year[y]}
        r_fname_to_path = {fname: path for fname, path in r_crops_by_year[y]}

        for z_name in tqdm(z_keys, desc=f"Matching Z vs R in {y}"):
            kp1, des1 = z_features_by_year[y][z_name]
            for r_name in r_keys:
                kp2, des2 = r_features_by_year[y][r_name]

                score, matches = hotspotter.match_features(kp1, des1, kp2, des2)

                if score >= config.SCORE_THRESHOLD:
                    possible_matches.append(
                        {
                            "crop_Z": z_name,
                            "crop_R": r_name,
                            "score": int(score),
                            "year": y,
                            "original_path_Z": z_fname_to_path[z_name],
                            "original_path_R": r_fname_to_path[r_name],
                        }
                    )

    # Sort matches from highest score to lowest
    possible_matches.sort(key=lambda x: x["score"], reverse=True)
    print(
        f"\nFound {len(possible_matches)} possible cross-tunnel matches "
        f"with score >= {config.SCORE_THRESHOLD}."
    )

    # Save results to a CSV file
    print(f"Saving matches to: {config.CROSS_TUNNEL_MATCHES_CSV}")
    with open(config.CROSS_TUNNEL_MATCHES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_Z",
                "crop_R",
                "score",
                "year",
                "original_path_Z",
                "original_path_R",
            ],
        )
        writer.writeheader()
        writer.writerows(possible_matches)

    print("Re-identification matching completed successfully!")


if __name__ == "__main__":
    main()
