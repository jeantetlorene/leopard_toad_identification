# Deep Analysis of Leopard Toad Detection Performance

This report synthesizes the results from Architectural, Preprocessing, Transfer Learning, and Active Learning evaluations to identify the most promising model and optimization strategies for Western Leopard Toad (WLT) monitoring.

## 1. Architectural Comparison (The Speed-Accuracy Trade-off)

*   **RT-DETR**: The strongest performer in terms of raw localization capability at Cycle 0 (mAP: 0.7147, mAR: 0.8351). However, it is the most computationally expensive model (2.1 FPS), making it unsuitable for real-time edge deployment but excellent for batch server processing.
*   **YOLO**: Offers a strong balance. While its Cycle 0 mAP (0.3088) is lower than RT-DETR, it is **12x faster** (25.9 FPS). Its recall (0.6938) is excellent for a baseline.
*   **Faster R-CNN**: Consistently underperformed in this domain, showing the lowest mAP and mAR across almost all tests (mAP: 0.3565). It appears less capable of handling the specific cryptic nature of leopard toads in infrared time-lapse data.

## 2. Optimization Pointers

### Preprocessing (CLAHE)
*   **YOLO is a consistent beneficiary**: Applying CLAHE to YOLO (Pretrained) boosted mAP from 0.5366 to 0.5511. CLAHE should be considered a standard preprocessing step if using YOLO to maximize feature extraction.
*   **RT-DETR Sensitivity**: RT-DETR (Pretrained) performed better on "Plain" images at Cycle 0 (mAP 0.6231 vs 0.4766). This suggests that the transformer-based architecture might be sensitive to the local noise amplification that CLAHE can sometimes introduce in high-contrast night shots.

### Transfer Learning
*   **Crucial for YOLO**: Using domain-specific pretrained weights significantly improved YOLO's performance (mAP 0.3720 -> 0.5511). 
*   **Mixed for RT-DETR**: Scratch training actually yielded higher mAP (0.5905) than the Pretrained variant (0.4766) at Cycle 0. This indicates that RT-DETR might require longer fine-tuning or a different learning rate schedule to fully leverage pretrained weights without "forgetting" general features.

### Active Learning Dynamics
*   **The Cycle 3 Anomaly**: Most models (especially YOLO) experienced a significant performance dip at Cycle 3. This often occurs in active learning when the model is presented with a batch of "highly informative" but extremely difficult boundary cases that temporarily confuse the decision boundary.
*   **WLT Specificity**: While "Other Amphibian" and "Small Mammal" metrics reached high peaks (RT-DETR hit 1.0 AP for Small Mammals), the **Western Leopard Toad** remains the most challenging class, typically hovering around 0.30 - 0.35 AP. 

## 3. Which Model is Most Promising?

### **The Winner: RT-DETR (Plain Pretrained)**
If the goal is **maximum scientific accuracy**, RT-DETR is the most promising.
*   It reached the **highest overall performance** in the Active Learning progression (mAP: 0.5596 at Cycle 4).
*   It demonstrates superior "ceiling" performance when given enough data cycles.

### **The Practical Alternative: YOLO (Clahe Pretrained)**
If the goal is **deployment on edge hardware** or processing massive datasets quickly:
*   YOLO reaches a "usable" performance level much faster than Faster R-CNN.
*   With CLAHE and Pretrained weights, it peaked at **mAP 0.47** in Cycle 2, which is excellent for a single-stage detector in such a difficult environment.

## Recommendations
1.  **Adopt RT-DETR** for all final server-side processing of the 5Z camera trap footage.
2.  **Continue Active Learning for WLT**: The gap between Small Mammal accuracy (high) and WLT accuracy (moderate) suggests we should bias the querying mechanism even further toward the Toad class.
3.  **Investigate Cycle 3**: Analyze the specific images added in Cycle 3 to understand why they caused a universal drop in mAP; these are likely the most critical "edge cases" for the project.
