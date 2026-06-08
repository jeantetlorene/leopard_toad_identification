# Results: Effect of Architecture (Cycle 0)

This report benchmarks the baseline network paradigms (YOLO, RT-DETR, Faster R-CNN) at Cycle 0, evaluating their computational efficiency and fundamental localization abilities before domain-specific transfer learning.

### Comprehensive Architectural Benchmarking Table
| Architecture | mAP50 | mAP50-95 | mean Average Recall | Params (M) | GFLOPs | Inference Speed (ms) | FPS |
|--------------|-------|----------|---------------------|------------|--------|----------------------|-----|
| YOLO | 0.5201 | 0.3507 | 0.3938 | 20.35 | N/A | 12.9 | 77.3 |
| RTDETR | 0.6572 | 0.4123 | 0.5526 | 31.99 | N/A | 35.0 | 28.6 |
| FASTER_RCNN | 0.3096 | 0.2046 | 0.4581 | 43.27 | 452.05 | 28.0 | 35.8 |

### Architecture-Specific Confusion Matrices
These matrices cross-reference predicted categories against actual ground-truth labels at a 0.5 confidence threshold, explicitly demonstrating inter-class confusion and background noise vulnerability.

#### YOLO
![YOLO Confusion Matrix](../plots/confusion_matrix_yolo_cycle0.png)

#### RTDETR
![RTDETR Confusion Matrix](../plots/confusion_matrix_rtdetr_cycle0.png)

#### FASTER_RCNN
![FASTER_RCNN Confusion Matrix](../plots/confusion_matrix_faster_rcnn_cycle0.png)

