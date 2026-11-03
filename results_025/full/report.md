# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.484 | 0.235 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.422 | 0.234 |
| yolo:yolo11m.pt | 0.469 | 0.225 |
| hf_detr:PekingU/rtdetr_r50vd | 0.502 | 0.240 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.737 | 0.434 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.663 | 0.441 |
| yolo:yolo11m.pt | 0.640 | 0.412 |
| hf_detr:PekingU/rtdetr_r50vd | 0.728 | 0.441 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.249 |
| torchvision:retinanet_resnet50_fpn_v2 | +0.188 |
| yolo:yolo11m.pt | +0.245 |
| hf_detr:PekingU/rtdetr_r50vd | +0.262 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (5000 imgs, 504.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.484 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.737 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.516 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.280 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.491 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.651 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.231 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.539 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.577 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.406 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.600 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.719 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.522 |
| person | 0.557 |
| cyclist | 0.372 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (5984 imgs, 704.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.434 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.119 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.309 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.109 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.313 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.329 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.305 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.322 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.369 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.448 |
| person | 0.255 |
| cyclist | 0.001 |

### torchvision:retinanet_resnet50_fpn_v2  ×  coco  (5000 imgs, 408.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.422 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.663 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.439 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.206 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.438 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.600 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.215 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.476 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.511 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.305 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.540 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.679 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.462 |
| person | 0.498 |
| cyclist | 0.306 |

### torchvision:retinanet_resnet50_fpn_v2  ×  kitti  (5984 imgs, 598.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.441 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.225 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.128 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.223 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.306 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.109 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.329 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.273 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.329 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.380 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.459 |
| person | 0.243 |
| cyclist | 0.001 |

### yolo:yolo11m.pt  ×  coco  (5000 imgs, 215.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.469 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.640 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.506 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.203 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.490 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.695 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.230 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.507 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.519 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.247 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.559 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.740 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.508 |
| person | 0.544 |
| cyclist | 0.356 |

### yolo:yolo11m.pt  ×  kitti  (5984 imgs, 462.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.225 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.412 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.224 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.109 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.213 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.303 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.104 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.289 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.294 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.211 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.293 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.363 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.446 |
| person | 0.227 |
| cyclist | 0.001 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 430.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.502 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.728 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.526 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.252 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.511 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.710 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.234 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.548 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.587 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.368 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.615 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.773 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.547 |
| person | 0.579 |
| cyclist | 0.378 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (5984 imgs, 695.0s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.240 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.441 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.130 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.229 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.106 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.302 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.254 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.310 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.373 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.484 |
| person | 0.234 |
| cyclist | 0.002 |
