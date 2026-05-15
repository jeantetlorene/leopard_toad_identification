# Deep Analysis of Leopard Toad Detection Performance

This report synthesizes the results from Architectural, Preprocessing, Transfer Learning, and Active Learning evaluations to identify the most promising model and optimization strategies for Western Leopard Toad (WLT) monitoring.

## 1. Architectural Comparison (The Speed-Accuracy Trade-off)

*   **RT-DETR**: The strongest performer in terms of raw localization capability at Cycle 0 (mAP: 0.2446, mAR: 0.5808). However, it is the most computationally expensive model (6.4 FPS), making it unsuitable for real-time edge deployment but excellent for batch server processing.
*   **YOLO**: Offers the best balance. While its Cycle 0 mAP (0.1712) is lower than RT-DETR, it is **14x faster** (88.4 FPS). Its recall (0.4259) is respectable for a baseline.
*   **Faster R-CNN**: Consistently underperformed in this domain, showing the lowest mAP and mAR across almost all tests. It appears less capable of handling the specific cryptic nature of leopard toads in infrared time-lapse data.

## 2. Optimization Pointers

### Preprocessing (CLAHE)
*   **YOLO is the primary beneficiary**: Applying CLAHE to YOLO (Pretrained) boosted mAP from 0.25 to 0.30 and mAR from 0.55 to 0.72. CLAHE should be considered a mandatory preprocessing step if using YOLO.
*   **RT-DETR Sensitivity**: Surprisingly, RT-DETR (Pretrained) performed better on "Plain" images at Cycle 0. This suggests that the transformer-based architecture might be sensitive to the local noise amplification that CLAHE can sometimes introduce.

### Transfer Learning
*   **Crucial for YOLO**: Using domain-specific pretrained weights more than doubled YOLO's performance (mAP 0.14 -> 0.30). 
*   **Mixed for RT-DETR**: Scratch training actually yielded higher mAP than the Pretrained variant at Cycle 0. This indicates that RT-DETR might require longer fine-tuning or a different learning rate schedule to fully leverage pretrained weights without "forgetting" general features.

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
