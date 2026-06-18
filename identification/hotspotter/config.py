"""
Configuration settings for the HotSpotter re-identification module.
"""

import os

# Base paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification"
IDENTIFICATION_DIR = os.path.join(BASE_DIR, "identification")
DETECTION_DIR = os.path.join(BASE_DIR, "detection")

# CSV paths
PREDICTIONS_CSV = os.path.join(
    DETECTION_DIR,
    "results",
    "detect_rtdetr_cycle2_clahe_pretrained",
    "wlt_predictions.csv",
)
CROSS_TUNNEL_MATCHES_CSV = os.path.join(
    IDENTIFICATION_DIR, "results", "cross_tunnel_matches_filtered.csv"
)
ALL_MATCHES_CSV = os.path.join(
    IDENTIFICATION_DIR, "results", "possible_matches_filtered.csv"
)

# Crop directory paths
CROPS_DIR = os.path.join(IDENTIFICATION_DIR, "data", "wlt_predictions_crops_filtered")

# SIFT Parameters optimized for toad patterns
SIFT_CONTRAST_THRESHOLD = 0.02  # Captures low-contrast spots (default 0.04)
SIFT_EDGE_THRESHOLD = 10  # Retains keypoints along spot edges (default 10)

# Preprocessing Pipeline Settings
TARGET_SIZE = 500  # Standard bicubic upscaling width/height
CLAHE_CLIP_LIMIT = 2.0  # CLAHE clip limit
CLAHE_TILE_GRID_SIZE = (5, 5)  # CLAHE tile grid size
BILATERAL_D = 9  # Bilateral filter pixel diameter
BILATERAL_SIGMA_COLOR = 75  # Bilateral filter color space sigma
BILATERAL_SIGMA_SPACE = 75  # Bilateral filter coordinate space sigma
SHARPEN_KERNEL = (5, 5)  # Unsharp mask Gaussian kernel size
SHARPEN_SIGMA = 10.0  # Unsharp mask Gaussian sigma
SHARPEN_WEIGHT1 = 1.5  # Base image weighting for unsharp mask
SHARPEN_WEIGHT2 = -0.5  # Blurred image weighting for unsharp mask

# Matcher settings
SCORE_THRESHOLD = 15  # Inlier score threshold for a valid match (>= 15 is strong)
RATIO_THRESHOLD = 0.75  # Lowe's ratio test threshold

# FLANN Matcher parameters
FLANN_INDEX_KDTREE = 1
TREES = 5
CHECKS = 50

# RANSAC parameters
RANSAC_REPROJ_THRESHOLD = 5.0
