"""
Visualize cross-tunnel toad matches and export publication-ready reports to a premium styled PDF.
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

# Ensure hotspotter folder is in the system path for configuration imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from matcher import BatchHotSpotter


def main():
    # 1. Load matches from the generated CSV
    if not os.path.exists(config.CROSS_TUNNEL_MATCHES_CSV):
        print(
            f"Error: {config.CROSS_TUNNEL_MATCHES_CSV} not found.\n"
            f"Please run 'python run_cross_tunnel_hotspotter.py' first to generate matches."
        )
        return

    df = pd.read_csv(config.CROSS_TUNNEL_MATCHES_CSV)
    if len(df) == 0:
        print("No matches found in the CSV to visualize.")
        return

    # Take the top 3 matches based on RANSAC inlier scores
    top_matches = df.head(3)

    # Initialize the re-identification matcher
    hotspotter = BatchHotSpotter()
    pdf_path = os.path.join(config.IDENTIFICATION_DIR, "top_3_cross_tunnel_matches.pdf")

    print(f"Generating report containing top 3 toad matches...")

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

            # Manually reconstruct the preprocessed visual stages for side-by-side mapping
            padded_z = hotspotter.pad_to_square(img_z)
            gray_z = (
                cv2.cvtColor(padded_z, cv2.COLOR_BGR2GRAY)
                if len(padded_z.shape) == 3
                else padded_z
            )
            resized_z = cv2.resize(
                gray_z,
                (config.TARGET_SIZE, config.TARGET_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )

            padded_r = hotspotter.pad_to_square(img_r)
            gray_r = (
                cv2.cvtColor(padded_r, cv2.COLOR_BGR2GRAY)
                if len(padded_r.shape) == 3
                else padded_r
            )
            resized_r = cv2.resize(
                gray_r,
                (config.TARGET_SIZE, config.TARGET_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )

            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=config.CLAHE_TILE_GRID_SIZE,
            )
            eq_z = clahe.apply(resized_z)
            eq_r = clahe.apply(resized_r)

            # Bilateral filter denoising
            filt_z = cv2.bilateralFilter(
                eq_z,
                d=config.BILATERAL_D,
                sigmaColor=config.BILATERAL_SIGMA_COLOR,
                sigmaSpace=config.BILATERAL_SIGMA_SPACE,
            )
            filt_r = cv2.bilateralFilter(
                eq_r,
                d=config.BILATERAL_D,
                sigmaColor=config.BILATERAL_SIGMA_COLOR,
                sigmaSpace=config.BILATERAL_SIGMA_SPACE,
            )

            # Unsharp masking detail amplification
            blur_z = cv2.GaussianBlur(
                filt_z, config.SHARPEN_KERNEL, config.SHARPEN_SIGMA
            )
            prep_z = cv2.addWeighted(
                filt_z,
                config.SHARPEN_WEIGHT1,
                blur_z,
                config.SHARPEN_WEIGHT2,
                0,
            )

            blur_r = cv2.GaussianBlur(
                filt_r, config.SHARPEN_KERNEL, config.SHARPEN_SIGMA
            )
            prep_r = cv2.addWeighted(
                filt_r,
                config.SHARPEN_WEIGHT1,
                blur_r,
                config.SHARPEN_WEIGHT2,
                0,
            )

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

            # # Clean styled header labels for white background
            # title_text = (
            #     f"Leopard Toad Cross-Tunnel Match #{idx + 1}\n"
            #     f"Cohort Year: {year}  |  RANSAC Verification Inliers: {score}"
            # )
            # fig.suptitle(
            #     title_text,
            #     color="black",
            #     fontsize=18,
            #     fontweight="bold",
            #     y=0.95,
            #     family="sans-serif",
            # )

            # Adjust spacing and save to PDF page
            plt.subplots_adjust(top=0.88, bottom=0.06, left=0.04, right=0.96)
            pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300)
            plt.close(fig)

    print(f"\nMatch report compilation finished! Saved PDF to:\n-> {pdf_path}")


if __name__ == "__main__":
    main()
