# Western Leopard Toad (WLT) Priority Analysis

This analysis refocuses the evaluation metrics entirely on the **Western Leopard Toad (WLT)**, prioritizing the detection and recall of this endangered species over general model performance.

## 1. Top Performing Models for WLT

If we isolate WLT-specific metrics, the hierarchy changes compared to the general mAP-based ranking:

### **The Recall Leader: RT-DETR (Clahe Pretrained)**
*   **Peak WLT Recall**: **0.6172** (reached at Cycle 3).
*   **WLT AP50**: Maintained a stable range between **0.30 and 0.35**.
*   **Significance**: For wildlife monitoring, missing an endangered individual is more critical than a false positive. This model variant demonstrated the highest "capture rate" for the target species throughout the active learning progression.

### **The Most Reliable Baseline: YOLO (Clahe Pretrained)**
*   **Cycle 0 WLT AP50**: **0.3416** (The highest starting point for WLT).
*   **Peak WLT Recall**: **0.5234** (reached at Cycle 4).
*   **Significance**: YOLO started strong for the toad class and maintained consistent recall. It is the most robust option for high-speed toad detection.

## 2. Active Learning Impact on WLT

An interesting trend emerged during the active learning cycles:

*   **The "Toad Plateau"**: While overall mAP often increased significantly (driven by "Small Mammals" and "Other Amphibians"), the **WLT AP50 remained stubbornly between 0.28 and 0.35**. 
*   **Querying Efficiency**: The number of WLT instances in the test set is fixed, but the training pool grew from 55 to 59 instances. This indicates that while we are querying more toads, they are likely the "complex boundary cases" (highly camouflaged or partially occluded), which prevents the AP from skyrocketing despite better training data.
*   **Specificity Trade-off**: At Cycle 4, **RT-DETR (Plain Pretrained)** achieved high specificity (0.96) for WLT, but its recall dropped slightly. This suggests the model is becoming more "cautious," which might be undesirable if the goal is maximum monitoring coverage.

## 3. Optimization Recommendations (WLT-First)

1.  **Selection**: Use **RT-DETR (Clahe Pretrained)** for the final deployment if the objective is to maximize the number of toads detected (Highest Recall).
2.  **Thresholding**: Because the **WLT Precision** is generally lower (0.35 - 0.50) than other classes, a class-specific **low confidence threshold** must be used for WLT detections to maintain the high recall capacity of the model.
3.  **Active Learning Bias**: The current AL strategy successfully improved "Small Mammal" performance to near-perfect levels. To break the "Toad Plateau," future cycles should be heavily biased toward Toad uncertainty, even at the expense of other classes.

## 4. Summary Table: WLT Performance at Cycle 4

| Model | Variant | WLT AP50 | WLT Recall | WLT Precision |
|-------|---------|----------|------------|---------------|
| **RT-DETR** | **Clahe Pretrained** | **0.3349** | **0.5469** | 0.4795 |
| **YOLO** | **Clahe Pretrained** | 0.2903 | 0.5234 | 0.3545 |
| **RT-DETR** | **Plain Pretrained** | 0.3169 | 0.4922 | **0.5250** |
| **FASTER_RCNN** | **Clahe Scratch** | 0.1439 | 0.3203 | 0.3661 |
