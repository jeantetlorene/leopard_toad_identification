# Results: Effect of Architecture (Cycle 0)

This report benchmarks the baseline network paradigms (YOLO, RT-DETR, Faster R-CNN) at Cycle 0, evaluating their computational efficiency and fundamental localization abilities before domain-specific transfer learning.

### Comprehensive Architectural Benchmarking Table
| Architecture | mAP50 | mAP50-95 | mean Average Recall | Params (M) | GFLOPs | Inference Speed (ms) | FPS |
|--------------|-------|----------|---------------------|------------|--------|----------------------|-----|
| YOLO | 0.3088 | 0.2980 | 0.6938 | 20.35 | 67.86 | 38.7 | 25.9 |
| RTDETR | 0.7147 | 0.3783 | 0.8351 | 31.99 | 103.44 | 486.8 | 2.1 |
| FASTER_RCNN | 0.3565 | 0.1362 | 0.4938 | 43.27 | 452.05 | 79.1 | 12.6 |

### Architecture-Specific Confusion Matrices
These matrices cross-reference predicted categories against actual ground-truth labels at a 0.5 confidence threshold, explicitly demonstrating inter-class confusion and background noise vulnerability.

#### YOLO
![YOLO Confusion Matrix](../plots/confusion_matrix_yolo_cycle0.png)

#### RTDETR
![RTDETR Confusion Matrix](../plots/confusion_matrix_rtdetr_cycle0.png)

#### FASTER_RCNN
![FASTER_RCNN Confusion Matrix](../plots/confusion_matrix_faster_rcnn_cycle0.png)

