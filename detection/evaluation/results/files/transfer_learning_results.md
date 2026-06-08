# Results: Effect of Transfer Learning (Cycle 0)

This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.

### Comprehensive Transfer Learning Performance Table
| Architecture | Variant | mAP50 | mAP50-95 | mean Average Recall | Trainable Parameters (M) |
|--------------|---------|-------|----------|---------------------|--------------------------|
| YOLO | Scratch | 0.5201 | 0.3507 | 0.3938 | 21.78 |
| YOLO | Pretrained | 0.8864 | 0.6061 | 0.8013 | 21.78 |
| RTDETR | Scratch | 0.6572 | 0.4123 | 0.5526 | 32.81 |
| RTDETR | Pretrained | 0.5411 | 0.3666 | 0.5332 | 32.81 |
| FASTER_RCNN | Scratch | 0.3096 | 0.2046 | 0.4581 | 43.27 |
| FASTER_RCNN | Pretrained | 0.2315 | 0.1360 | 0.4253 | 43.27 |
