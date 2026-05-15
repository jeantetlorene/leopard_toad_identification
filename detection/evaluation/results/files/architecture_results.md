# Results: Effect of Architecture (Cycle 0)

This report benchmarks the baseline network paradigms (YOLO, RT-DETR, Faster R-CNN) at Cycle 0, evaluating their computational efficiency and fundamental localization abilities before domain-specific transfer learning.

### Comprehensive Architectural Benchmarking Table
| Architecture | mAP50 | mAP50-95 | mean Average Recall | Params (M) | GFLOPs | Inference Speed (ms) | FPS |
|--------------|-------|----------|---------------------|------------|--------|----------------------|-----|
| YOLO | 0.1712 | 0.1128 | 0.4259 | 20.35 | 67.86 | 11.3 | 88.4 |
| RTDETR | 0.2446 | 0.1819 | 0.5808 | 31.99 | 103.44 | 156.3 | 6.4 |
| FASTER_RCNN | 0.0139 | 0.0518 | 0.2022 | 43.27 | 452.05 | 39.2 | 25.5 |

### Architecture-Specific Confusion Matrices
These matrices cross-reference predicted categories against actual ground-truth labels at a 0.5 confidence threshold, explicitly demonstrating inter-class confusion and background noise vulnerability.

#### YOLO
![YOLO Confusion Matrix](../plots/confusion_matrix_yolo_cycle0.png)

#### RTDETR
![RTDETR Confusion Matrix](../plots/confusion_matrix_rtdetr_cycle0.png)

#### FASTER_RCNN
![FASTER_RCNN Confusion Matrix](../plots/confusion_matrix_faster_rcnn_cycle0.png)

