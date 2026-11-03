# ngdet latency benchmark

Device: **NVIDIA GeForce RTX 3060 Ti**. End-to-end `predict()` latency (includes pre/post-processing).

| model | accel | mean ms | p50 ms | p90 ms | FPS | speedup |
|---|---|---|---|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | fp32 | 67.4 | 67.2 | 70.9 | 14.8 | x1.00 |
| torchvision:retinanet_resnet50_fpn_v2 | fp32 | 48.1 | 48.0 | 51.3 | 20.8 | x1.00 |
| yolo:yolo11m.pt | fp32 | 25.3 | 24.9 | 25.9 | 39.6 | x1.00 |
| hf_detr:rtdetr_r50vd | fp32 | 63.5 | 62.5 | 69.0 | 15.8 | x1.00 |