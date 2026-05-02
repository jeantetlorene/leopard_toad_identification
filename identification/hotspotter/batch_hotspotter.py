# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
from itertools import combinations


# %%
# =============================================================================
# CLASS: BATCH HOTSPOTTER MATCHER
# =============================================================================
class BatchHotSpotter:
    def __init__(self):
        # 1. Initialize SIFT Detector
        self.sift = cv2.SIFT_create()

        # 2. Initialize Matcher (FLANN-based is faster/standard for SIFT)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Higher checks = more precision, slower
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def get_features(self, image_path):
        """
        Loads an image from path and extracts SIFT keypoints and descriptors.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and Compute
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, kp1, des1, kp2, des2, ratio_thresh=0.75):
        """
        Matches pre-computed features using FLANN, Lowe's ratio test, and RANSAC.
        """
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

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                matches_mask = mask.ravel().tolist()
                # Count "Inliers"
                score = np.sum(matches_mask)
                final_matches = [
                    good_matches[i] for i in range(len(good_matches)) if matches_mask[i]
                ]

        return score, final_matches


# %%
# Configuration
DATA_DIR = (
    "/home/Joshua/Downloads/leopard_toad_identification/dataset/dataset_reid_crops"
)
SCORE_THRESHOLD = 20  # Adjust this if needed (>10 is usually a strong match)
OUTPUT_CSV = "/home/Joshua/Downloads/leopard_toad_identification/identification/possible_matches.csv"

# 1. Pre-compute features for all images to save time.
# This prevents computing the SIFT features repeatedly for the same image.
hotspotter = BatchHotSpotter()
all_images = [
    f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Found {len(all_images)} images. Extracting features...")

image_features = {}
for filename in tqdm(all_images, desc="Extracting SIFT features"):
    img_path = os.path.join(DATA_DIR, filename)
    kps, des = hotspotter.get_features(img_path)
    if kps is not None and des is not None:
        image_features[filename] = (kps, des)

# %%
# 2. Perform NxN matching comparisons
print("Performing combination matching limit overlaps...")
possible_matches = []

# Generate all unique pairs
# combinations will generate pairs like (A,B), without checking (B,A) and (A,A)
image_names = list(image_features.keys())
pairs = list(combinations(image_names, 2))

for img1_name, img2_name in tqdm(pairs, desc="Matching image pairs"):
    kp1, des1 = image_features[img1_name]
    kp2, des2 = image_features[img2_name]

    score, matches = hotspotter.match_features(kp1, des1, kp2, des2)

    # Store match if the score exceeds manual validation threshold
    if score >= SCORE_THRESHOLD:
        possible_matches.append(
            {"image1": img1_name, "image2": img2_name, "score": int(score)}
        )

# Sort the matches from highest score to lowest
possible_matches.sort(key=lambda x: x["score"], reverse=True)
print(
    f"Found {len(possible_matches)} possible matches with a score >= {SCORE_THRESHOLD}."
)

# %%
# 3. Save results to a CSV for easier manual processing offline
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image1", "image2", "score"])
    writer.writeheader()
    writer.writerows(possible_matches)

print(f"Results saved to {OUTPUT_CSV}")


# %%
# 4. Optional: Visualization of top matches for manual validation
def visualize_match(img1_name, img2_name, score):
    img1_path = os.path.join(DATA_DIR, img1_name)
    img2_path = os.path.join(DATA_DIR, img2_name)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    kp1, des1 = image_features[img1_name]
    kp2, des2 = image_features[img2_name]

    _, matches = hotspotter.match_features(kp1, des1, kp2, des2)

    result_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
    )

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{img1_name} vs {img2_name} | Score: {score}", fontsize=14)
    plt.axis("off")
    plt.show()


# Show the top 5 matches right here in the notebook view
for match in possible_matches[:5]:
    visualize_match(match["image1"], match["image2"], match["score"])

# %%
