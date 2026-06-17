"""
Visualize cross-tunnel toad matches and export publication-ready reports to a premium styled PDF.
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import argparse
import re

# Ensure hotspotter folder is in the system path for configuration imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from matcher import BatchHotSpotter


def extract_camera(path_or_filename):
    match = re.search(r"\b\d+[ZR]\b", path_or_filename, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    match = re.search(r"\d+[ZR]", path_or_filename, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return "Unknown"


def extract_year(path_or_filename):
    # Try matching at the start of filename (e.g. 2024__18...)
    match = re.search(r"^(2023|2024|2025|2026)", path_or_filename)
    if match:
        return match.group(0)
    # Try matching directory separator (e.g. /2024/ or \2024\)
    match = re.search(r"[/\\](2023|2024|2025|2026)[/\\]", path_or_filename)
    if match:
        return match.group(1)
    # Fallback to general search anywhere
    match = re.search(r"(2023|2024|2025|2026)", path_or_filename)
    if match:
        return match.group(0)
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-tunnel toad matches and export reports to a PDF."
    )
    parser.add_argument(
        "-n",
        "--num-matches",
        type=int,
        default=None,
        help="Number of top matches to visualize (default: all matches found)",
    )
    parser.add_argument(
        "-p",
        "--prep-mode",
        type=str,
        choices=["none", "original", "improved"],
        default="original",
        help="Preprocessing pipeline to use (default: 'original')",
    )
    args = parser.parse_args()

    num_matches = args.num_matches
    prep_mode = args.prep_mode

    # 1. Load matches from the generated CSV
    base, ext = os.path.splitext(config.CROSS_TUNNEL_MATCHES_CSV)
    csv_path = f"{base}_{prep_mode}{ext}"
    if not os.path.exists(csv_path):
        print(
            f"Error: {csv_path} not found.\n"
            f"Please run 'python run_cross_tunnel_hotspotter.py --prep-mode {prep_mode}' first to generate matches."
        )
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("No matches found in the CSV to visualize.")
        return

    if num_matches is None:
        num_to_visualize = len(df)
    else:
        if num_matches <= 0:
            print("Error: The number of matches to visualize must be at least 1.")
            return
        num_to_visualize = min(num_matches, len(df))

    # Take the top matches based on RANSAC inlier scores
    top_matches = df.head(num_to_visualize)

    # Initialize the re-identification matcher
    hotspotter = BatchHotSpotter(prep_mode=prep_mode)
    pdf_path = os.path.join(
        config.IDENTIFICATION_DIR,
        "results",
        f"top_{num_to_visualize}_cross_tunnel_matches_{prep_mode}.pdf",
    )

    print(
        f"Generating report containing top {num_to_visualize} toad matches [Prep Mode: {prep_mode}]..."
    )

    # Create multi-page PDF using PdfPages backend
    with PdfPages(pdf_path) as pdf:
        for idx, (_, row) in enumerate(top_matches.iterrows()):
            crop_z_name = row["crop_Z"]
            crop_r_name = row["crop_R"]
            score = int(row["score"])
            year = row["year"]

            path_z = os.path.join(config.CROPS_DIR, crop_z_name)
            path_r = os.path.join(config.CROPS_DIR, crop_r_name)

            # Check if crops exist on disk
            if not os.path.exists(path_z) or not os.path.exists(path_r):
                print(
                    f"Warning: Missing crop files for {crop_z_name} or {crop_r_name}. Skipping."
                )
                continue

            img_z = cv2.imread(path_z)
            img_r = cv2.imread(path_r)

            if img_z is None or img_r is None:
                print(
                    f"Warning: Failed to load crop images: {crop_z_name} / {crop_r_name}. Skipping."
                )
                continue

            # Run re-identification to compute matching SIFT inliers
            kp1, des1 = hotspotter.get_features(path_z)
            kp2, des2 = hotspotter.get_features(path_r)
            _, inliers = hotspotter.match_features(kp1, des1, kp2, des2)

            # Preprocess the visual stages for side-by-side mapping
            prep_z = hotspotter.preprocess_image_by_mode(img_z)
            prep_r = hotspotter.preprocess_image_by_mode(img_r)

            # Convert preprocessed outputs to color spaces to permit vivid matching lines
            color_prep_z = cv2.cvtColor(prep_z, cv2.COLOR_GRAY2BGR)
            color_prep_r = cv2.cvtColor(prep_r, cv2.COLOR_GRAY2BGR)

            # Draw green keypoint connection vectors
            match_img = cv2.drawMatches(
                color_prep_z,
                kp1,
                color_prep_r,
                kp2,
                inliers,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(46, 204, 113),  # Premium emerald green (RGB: 46, 204, 113)
            )

            # Initialize Matplotlib figure with styled ratios
            fig, ax = plt.subplots(figsize=(16, 9))

            # Set clean white background
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            ax.axis("off")

            # Parse crop details
            camera_z = extract_camera(crop_z_name)
            camera_r = extract_camera(crop_r_name)

            year_z = extract_year(crop_z_name)
            year_r = extract_year(crop_r_name)

            id_z = os.path.splitext(crop_z_name)[0]
            id_r = os.path.splitext(crop_r_name)[0]

            full_path_z = row["original_path_Z"]
            full_path_r = row["original_path_R"]

            text_z = (
                f"ID: {id_z}\n"
                f"Camera: {camera_z}  |  Year: {year_z}\n"
                f"File: {full_path_z}"
            )
            text_r = (
                f"ID: {id_r}\n"
                f"Camera: {camera_r}  |  Year: {year_r}\n"
                f"File: {full_path_r}"
            )

            # Add text under left image (axes coordinates)
            ax.text(
                0.02,
                -0.08,
                text_z,
                transform=ax.transAxes,
                fontsize=9,
                fontfamily="monospace",
                verticalalignment="top",
                color="black",
            )

            # Add text under right image (axes coordinates)
            ax.text(
                0.52,
                -0.08,
                text_r,
                transform=ax.transAxes,
                fontsize=9,
                fontfamily="monospace",
                verticalalignment="top",
                color="black",
            )

            # Clean styled header labels for white background
            title_text = f"Leopard Toad Cross-Tunnel Match #{idx + 1} (SIFT Inliers Score: {score})"
            fig.suptitle(
                title_text,
                color="black",
                fontsize=16,
                fontweight="bold",
                y=0.96,
                family="sans-serif",
            )

            # Adjust spacing and save to PDF page
            plt.subplots_adjust(top=0.90, bottom=0.20, left=0.04, right=0.96)
            pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
            plt.close(fig)

    print(f"\nMatch report compilation finished! Saved PDF to:\n-> {pdf_path}")


if __name__ == "__main__":
    main()
