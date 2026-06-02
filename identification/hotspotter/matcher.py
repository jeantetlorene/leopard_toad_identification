"""
HotSpotter feature extraction and matching classes.
"""

import cv2
import numpy as np
import config


class BatchHotSpotter:
    """
    SIFT and RANSAC-based keypoint matcher for re-identifying individuals.
    """

    def __init__(self):
        # 1. Initialize SIFT Detector
        # self.sift = cv2.SIFT_create()

        # Optimized for toad patterns:
        self.sift = cv2.SIFT_create(
            contrastThreshold=0.02,  # Captures low-contrast spot details
            edgeThreshold=15,  # Retains keypoints along spot edges
        )

        # 2. Initialize Matcher (FLANN-based is faster/standard for SIFT)
        index_params = dict(algorithm=config.FLANN_INDEX_KDTREE, trees=config.TREES)
        search_params = dict(checks=config.CHECKS)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def pad_to_square(self, img):
        """
        Pads a rectangular image to square shape using symmetric constant black borders.
        This preserves the aspect ratio and prevents geometric distortion during resizing.
        """
        h, w = img.shape[:2]
        if h == w:
            return img
        pad = abs(h - w) // 2
        if h > w:
            return cv2.copyMakeBorder(
                img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0
            )
        return cv2.copyMakeBorder(
            img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0
        )

    def get_features(self, image_path):
        """
        Loads an image, applies a robust computer vision preprocessing pipeline,
        and extracts SIFT keypoints and descriptors from the enhanced image.

        The Preprocessing Pipeline consists of:
        1. Aspect-Ratio-Preserving Padding to Square
        2. Grayscale Conversion
        3. Bicubic Upscaling to TARGET_SIZE (e.g. 500x500)
        4. CLAHE Contrast Enhancement
        5. Bilateral Filtering (removes grain noise, preserves edges)
        6. Unsharp Masking (sharpens transition borders)
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        # 1. Aspect-Ratio-Preserving Padding
        padded = self.pad_to_square(image)

        # 2. Grayscale Conversion
        if len(padded.shape) == 3:
            gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        else:
            gray = padded

        # 3. Bicubic Upscaling to standard square dimensions
        resized = cv2.resize(
            gray,
            (config.TARGET_SIZE, config.TARGET_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )
        equalized = clahe.apply(resized)

        # 5. Bilateral Filtering to remove grain noise while preserving sharp boundaries
        filtered = cv2.bilateralFilter(
            equalized,
            d=config.BILATERAL_D,
            sigmaColor=config.BILATERAL_SIGMA_COLOR,
            sigmaSpace=config.BILATERAL_SIGMA_SPACE,
        )

        # 6. Unsharp Masking (high-frequency detail amplification)
        blurred = cv2.GaussianBlur(
            filtered,
            config.SHARPEN_KERNEL,
            config.SHARPEN_SIGMA,
        )
        preprocessed = cv2.addWeighted(
            filtered,
            config.SHARPEN_WEIGHT1,
            blurred,
            config.SHARPEN_WEIGHT2,
            0,
        )

        # Detect and Compute SIFT Keypoints & Descriptors
        keypoints, descriptors = self.sift.detectAndCompute(preprocessed, None)
        return keypoints, descriptors

    def match_features(self, kp1, des1, kp2, des2, ratio_thresh=None):
        """
        Matches pre-computed features using FLANN, Lowe's ratio test, and RANSAC.
        """
        if ratio_thresh is None:
            ratio_thresh = config.RATIO_THRESHOLD

        # Check if features were actually found
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0, []

        # KNN Matching (k=2)
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)

        # Lowe's Ratio Test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        # Spatial Verification (RANSAC)
        score = 0
        final_matches = []

        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD
            )

            if mask is not None:
                matches_mask = mask.ravel().tolist()
                # Count "Inliers"
                score = np.sum(matches_mask)
                final_matches = [
                    good_matches[i] for i in range(len(good_matches)) if matches_mask[i]
                ]

        return score, final_matches
