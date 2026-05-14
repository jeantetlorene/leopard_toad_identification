# Results: Effect of Preprocessing (Cycle 0)

This report summarizes the impact of CLAHE on Cycle 0 model performance.

### Comparative Performance Table (Cycle 0, Test Set)
| Architecture | Variant | Processing | mAP | Average Recall |
|--------------|---------|------------|-----|----------------|
| FASTER_RCNN | Pretrained | Clahe | 0.0670 | 0.2264 |
| FASTER_RCNN | Pretrained | Plain | 0.0494 | 0.2492 |
| FASTER_RCNN | Scratch | Clahe | 0.0879 | 0.4465 |
| FASTER_RCNN | Scratch | Plain | 0.1069 | 0.2825 |
| RTDETR | Pretrained | Clahe | 0.1828 | 0.7613 |
| RTDETR | Pretrained | Plain | 0.3523 | 0.5808 |
| RTDETR | Scratch | Clahe | 0.3506 | 0.7191 |
| RTDETR | Scratch | Plain | 0.2034 | 0.7495 |
| YOLO | Pretrained | Clahe | 0.3050 | 0.7235 |
| YOLO | Pretrained | Plain | 0.2593 | 0.5512 |
| YOLO | Scratch | Clahe | 0.1401 | 0.6413 |
| YOLO | Scratch | Plain | 0.1755 | 0.4576 |

### Precision-Recall Visualizations

#### FASTER_RCNN
![FASTER_RCNN PR Curve](../plots/pr_curve_faster_rcnn_cycle0.png)

#### RTDETR
![RTDETR PR Curve](../plots/pr_curve_rtdetr_cycle0.png)

#### YOLO
![YOLO PR Curve](../plots/pr_curve_yolo_cycle0.png)

