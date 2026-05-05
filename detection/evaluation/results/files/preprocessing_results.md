# Results: Effect of Preprocessing (Cycle 0)

This report summarizes the impact of CLAHE on Cycle 0 model performance.

### Comparative Performance Table (Cycle 0, Test Set)
| Architecture | Variant | Processing | mAP | Average Recall |
|--------------|---------|------------|-----|----------------|
| FASTER_RCNN | Pretrained | Clahe | 0.0376 | 0.7750 |
| FASTER_RCNN | Pretrained | Plain | 0.0441 | 0.8333 |
| FASTER_RCNN | Scratch | Clahe | 0.0462 | 0.9767 |
| FASTER_RCNN | Scratch | Plain | 0.0624 | 0.8116 |
| RTDETR | Pretrained | Clahe | 0.1829 | 0.9969 |
| RTDETR | Pretrained | Plain | 0.3524 | 1.0000 |
| RTDETR | Scratch | Clahe | 0.3506 | 1.0000 |
| RTDETR | Scratch | Plain | 0.2031 | 1.0000 |
| YOLO | Pretrained | Clahe | 0.3105 | 0.8810 |
| YOLO | Pretrained | Plain | 0.2652 | 0.6918 |
| YOLO | Scratch | Clahe | 0.1571 | 0.7552 |
| YOLO | Scratch | Plain | 0.1876 | 0.5633 |

### Precision-Recall Visualizations

#### FASTER_RCNN
![FASTER_RCNN PR Curve](../plots/pr_curve_faster_rcnn_cycle0.png)

#### RTDETR
![RTDETR PR Curve](../plots/pr_curve_rtdetr_cycle0.png)

#### YOLO
![YOLO PR Curve](../plots/pr_curve_yolo_cycle0.png)

