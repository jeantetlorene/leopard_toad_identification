# Results: Effect of Transfer Learning (Cycle 0)

This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.

### Comprehensive Transfer Learning Performance Table
| Architecture | Variant | mAP50 | mAP50-95 | Average Recall | Trainable Parameters (M) |
|--------------|---------|-------|----------|----------------|--------------------------|
| YOLO | Scratch | 0.1401 | 0.1128 | 0.6413 | 21.78 |
| YOLO | Pretrained | 0.3050 | 0.2615 | 0.7235 | 21.78 |
| RTDETR | Scratch | 0.3506 | 0.1819 | 0.7191 | 32.81 |
| RTDETR | Pretrained | 0.1828 | 0.1527 | 0.7613 | 32.81 |
| FASTER_RCNN | Scratch | 0.0879 | 0.0518 | 0.4465 | 43.27 |
| FASTER_RCNN | Pretrained | 0.0670 | 0.0366 | 0.2264 | 43.27 |
