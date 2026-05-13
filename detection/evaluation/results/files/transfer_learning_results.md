# Results: Effect of Transfer Learning (Cycle 0)

This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.

### Comprehensive Transfer Learning Performance Table
| Architecture | Variant | mAP50 | mAP50-95 | Average Recall | Trainable Parameters (M) |
|--------------|---------|-------|----------|----------------|--------------------------|
| YOLO | Scratch | 0.1571 | 0.1253 | 0.7552 | 21.78 |
| YOLO | Pretrained | 0.3105 | 0.2655 | 0.8810 | N/A |
| RTDETR | Scratch | 0.3506 | 0.1819 | 1.0000 | 32.81 |
| RTDETR | Pretrained | 0.1829 | 0.1527 | 0.9969 | N/A |
| FASTER_RCNN | Scratch | 0.0462 | 0.0284 | 0.9767 | 43.27 |
| FASTER_RCNN | Pretrained | 0.0376 | 0.0208 | 0.7750 | N/A |
