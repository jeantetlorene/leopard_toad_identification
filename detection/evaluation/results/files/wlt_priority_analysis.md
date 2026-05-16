# Western Leopard Toad (WLT) Priority Analysis

This analysis refocuses the evaluation metrics entirely on the **Western Leopard Toad (WLT)**, prioritizing the detection and recall of this endangered species over general model performance.

## 1. Top Performing Models for WLT

If we isolate WLT-specific metrics, the hierarchy changes compared to the general mAP-based ranking:

### **The Recall Leader: RT-DETR (Clahe Pretrained)**
*   **Peak WLT Recall**: **0.7824** (reached at Cycle 1 and 3).
*   **WLT AP50**: Reached a peak of **0.7430** at Cycle 1.
*   **Significance**: This model variant consistently achieves the highest "capture rate" for the target species. Even at Cycle 0, it maintains a recall of 0.60, making it the most reliable choice for ensuring no individuals are missed.

### **The Most Reliable Baseline: YOLO (Clahe Pretrained)**
*   **Cycle 0 WLT AP50**: **0.7268** (A remarkably high starting point for WLT).
*   **Peak WLT Recall**: **0.7765** (reached at Cycle 1).
*   **Significance**: YOLO demonstrates exceptional fundamental localization for the toad class. Its high Cycle 0 performance suggests that its single-stage architecture is particularly well-suited to the WLT's visual features once labels are correctly synchronized.

## 2. Active Learning Impact on WLT

An interesting trend emerged during the active learning cycles:

*   **The Plateau is Broken**: Following the ground-truth synchronization, the previously observed "Toad Plateau" (0.28-0.35 AP) has been shattered. WLT AP50 now consistently reaches **0.65 - 0.75** across both RT-DETR and YOLO.
*   **Querying Efficiency**: The model is now much more effective at leveraging new data. We see WLT recall jumping significantly in Cycle 1 as the first batch of missed images was added to the training pool.
*   **Precision Stability**: WLT precision has improved dramatically from ~0.40 to **0.75 - 0.85**, indicating that the models are now much better at distinguishing toads from background noise.

## 3. Optimization Recommendations (WLT-First)

1.  **Selection**: Use **RT-DETR (Clahe Pretrained)** for maximum scientific fidelity (Highest Peak Recall of 0.78).
2.  **Thresholding**: With WLT precision now much higher (~0.82), we can afford to use a **standard confidence threshold (0.4 - 0.5)** without being overwhelmed by false positives, while still capturing the vast majority of individuals.
3.  **Future Cycles**: The "Toad Plateau" was a symptom of label noise. Now that the labels are clean, further active learning should focus on the remaining ~20% missed recall, which likely consists of extreme occlusion or motion blur cases.

## 4. Summary Table: WLT Performance at Cycle 4

| Model | Variant | WLT AP50 | WLT Recall | WLT Precision |
|-------|---------|----------|------------|---------------|
| **RT-DETR** | **Clahe Pretrained** | 0.7278 | **0.7471** | **0.8699** |
| **YOLO** | **Clahe Pretrained** | **0.7332** | 0.7000 | 0.8207 |
| **RT-DETR** | **Plain Pretrained** | 0.7277 | 0.6882 | **0.8797** |
| **FASTER_RCNN** | **Clahe Scratch** | 0.4041 | 0.4529 | 0.6875 |
