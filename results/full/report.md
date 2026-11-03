# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.492 | 0.236 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.435 | 0.237 |
| yolo:yolo11m.pt | 0.514 | 0.231 |
| hf_detr:PekingU/rtdetr_r50vd | 0.517 | 0.243 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.752 | 0.437 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.690 | 0.449 |
| yolo:yolo11m.pt | 0.721 | 0.431 |
| hf_detr:PekingU/rtdetr_r50vd | 0.754 | 0.449 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.255 |
| torchvision:retinanet_resnet50_fpn_v2 | +0.198 |
| yolo:yolo11m.pt | +0.283 |
| hf_detr:PekingU/rtdetr_r50vd | +0.274 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (5000 imgs, 538.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.492 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.752 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.523 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.286 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.499 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.660 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.232 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.553 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.606 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.438 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.629 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.745 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.531 |
| person | 0.565 |
| cyclist | 0.379 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (5984 imgs, 742.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.437 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.120 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.231 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.110 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.316 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.341 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.318 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.334 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.378 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.451 |
| person | 0.256 |
| cyclist | 0.001 |

### torchvision:retinanet_resnet50_fpn_v2  ×  coco  (5000 imgs, 416.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.435 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.690 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.448 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.218 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.451 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.610 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.219 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.502 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.578 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.386 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.616 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.721 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.478 |
| person | 0.511 |
| cyclist | 0.316 |

### torchvision:retinanet_resnet50_fpn_v2  ×  kitti  (5984 imgs, 622.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.237 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.449 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.130 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.316 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.116 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.332 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.380 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.316 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.384 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.429 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.464 |
| person | 0.246 |
| cyclist | 0.002 |

### yolo:yolo11m.pt  ×  coco  (5000 imgs, 197.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.514 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.721 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.552 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.248 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.535 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.725 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.236 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.560 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.603 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.362 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.646 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.788 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.553 |
| person | 0.590 |
| cyclist | 0.399 |

### yolo:yolo11m.pt  ×  kitti  (5984 imgs, 424.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.231 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.431 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.115 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.221 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.306 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.106 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.300 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.247 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.370 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.457 |
| person | 0.236 |
| cyclist | 0.001 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 385.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.517 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.754 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.542 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.266 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.527 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.726 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.236 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.576 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.667 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.483 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.685 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.829 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.563 |
| person | 0.593 |
| cyclist | 0.395 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (5984 imgs, 681.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.243 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.449 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.131 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.323 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.108 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.353 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.308 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.350 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.397 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.490 |
| person | 0.238 |
| cyclist | 0.002 |
