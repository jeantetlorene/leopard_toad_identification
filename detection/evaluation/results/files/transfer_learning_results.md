# Results: Effect of Transfer Learning (Cycle 0)

This report compares baseline architectures trained from scratch against identically configured architectures initialized with domain-specific pre-trained weights.

### Comprehensive Transfer Learning Performance Table
| Architecture | Variant | mAP50 | mAP50-95 | mean Average Recall | Trainable Parameters (M) |
|--------------|---------|-------|----------|---------------------|--------------------------|
| YOLO | Scratch | 0.5201 | 0.3506 | 0.3938 | 21.78 |
| YOLO | Pretrained | 0.8904 | 0.5234 | 0.8189 | 21.78 |
| RTDETR | Scratch | 0.6572 | 0.4263 | 0.5526 | 32.81 |
| RTDETR | Pretrained | 0.5511 | 0.4507 | 0.4892 | 32.81 |
| FASTER_RCNN | Scratch | 0.3096 | 0.1925 | 0.4581 | 43.27 |
| FASTER_RCNN | Pretrained | 0.1735 | 0.1329 | 0.3406 | 43.27 |
