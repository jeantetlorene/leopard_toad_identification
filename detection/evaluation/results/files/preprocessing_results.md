# Results: Effect of Preprocessing (Cycle 0)

This report summarizes the impact of CLAHE on Cycle 0 model performance.

### Comparative Performance Table (Cycle 0, Test Set)
| Architecture | Variant | Processing | mAP | mean Average Recall |
|--------------|---------|------------|-----|---------------------|
| FASTER_RCNN | Pretrained | Clahe | 0.1680 | 0.2956 |
| FASTER_RCNN | Pretrained | Plain | 0.1378 | 0.3287 |
| FASTER_RCNN | Scratch | Clahe | 0.2175 | 0.5387 |
| FASTER_RCNN | Scratch | Plain | 0.2409 | 0.3770 |
| RTDETR | Pretrained | Clahe | 0.4766 | 0.9174 |
| RTDETR | Pretrained | Plain | 0.6231 | 0.7387 |
| RTDETR | Scratch | Clahe | 0.5905 | 0.8446 |
| RTDETR | Scratch | Plain | 0.4838 | 0.9068 |
| YOLO | Pretrained | Clahe | 0.5511 | 0.8575 |
| YOLO | Pretrained | Plain | 0.5366 | 0.6863 |
| YOLO | Scratch | Clahe | 0.3720 | 0.7555 |
| YOLO | Scratch | Plain | 0.3965 | 0.5676 |

### Precision-Recall Visualizations

#### FASTER_RCNN
![FASTER_RCNN PR Curve](../plots/pr_curve_faster_rcnn_cycle0.png)

#### RTDETR
![RTDETR PR Curve](../plots/pr_curve_rtdetr_cycle0.png)

#### YOLO
![YOLO PR Curve](../plots/pr_curve_yolo_cycle0.png)

