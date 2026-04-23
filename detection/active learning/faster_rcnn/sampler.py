import numpy as np
import math
from sklearn.cluster import KMeans


class DCUS:
    """
    Difficulty Calibrated Uncertainty Sampling (DCUS).
    Scores bounding boxes weighting by class difficulty.
    """

    def __init__(self, num_classes=3, alpha=0.3, beta=0.2, xi=0.6, m0=0.99):
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = math.exp(1 / alpha) - 1
        self.xi = xi

        # Difficulties initialized to 0. In a real dynamic setup this updates per epoch.
        # For simplicity in testing offline, we treat classes 0 (Amphibian), 1 (Small Mammal),
        # and 2 (Leopard Toad). Naturally, 2 is the most difficult.
        self.d = np.array([0.1, 0.2, 0.9])

    def compute_difficulty_coefficient(self):
        """Compute difficulty weight w_i for each class"""
        w = 1 + self.alpha * self.beta * np.log(1 + self.gamma * self.d)
        return w

    def image_uncertainty(self, boxes):
        """
        Compute uncertainty for an image given its detected objects.
        boxes: list of dicts with 'conf' and 'cls'.
        """
        w = self.compute_difficulty_coefficient()
        if not boxes:
            return 0.0

        max_u = 0.0
        for box in boxes:
            c = int(box["cls"])
            if c < self.num_classes:
                p = box["conf"]
                # Binary cross-entropy estimation of prediction uncertainty
                entropy = -p * math.log(p + 1e-9) - (1 - p) * math.log((1 - p) + 1e-9)
                u = w[c] * entropy
                if u > max_u:
                    max_u = u
        return max_u


def diversity_sampling(embeddings, candidate_indices, n_samples):
    """
    Applies k-means++ clustering on deep feature embeddings to ensure Diversity.
    embeddings: nxD numpy array
    candidate_indices: array mapping local index back to global candidate pool
    n_samples: Annotation budget limit
    """
    if len(embeddings) <= n_samples:
        return candidate_indices

    kmeans = KMeans(n_clusters=n_samples, init="k-means++", n_init=1, random_state=42)
    kmeans.fit(embeddings)

    selected_indices = []
    # Identify the actual image nearest to each cluster centroid
    # as the representative diverse sample
    for center_idx in range(n_samples):
        center = kmeans.cluster_centers_[center_idx]
        distances = np.linalg.norm(embeddings - center, axis=1)
        best_idx = np.argmin(distances)
        selected_indices.append(candidate_indices[best_idx])

    return list(set(selected_indices))
