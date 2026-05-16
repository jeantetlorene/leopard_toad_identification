# Results: Effect of Transfer Learning (Cycle 0)

This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.

### Comprehensive Transfer Learning Performance Table
| Architecture | Variant | mAP50 | mAP50-95 | mean Average Recall | Trainable Parameters (M) |
|--------------|---------|-------|----------|---------------------|--------------------------|
| YOLO | Scratch | 0.3720 | 0.1128 | 0.7555 | 21.78 |
| YOLO | Pretrained | 0.5511 | 0.2615 | 0.8575 | 21.78 |
| RTDETR | Scratch | 0.5905 | 0.1819 | 0.8446 | 32.81 |
| RTDETR | Pretrained | 0.4766 | 0.1527 | 0.9174 | 32.81 |
| FASTER_RCNN | Scratch | 0.2175 | 0.0518 | 0.5387 | 43.27 |
| FASTER_RCNN | Pretrained | 0.1680 | 0.0366 | 0.2956 | 43.27 |
