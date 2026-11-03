# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.577 | 0.256 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.526 | 0.257 |
| yolo:yolo11m.pt | 0.579 | 0.228 |
| hf_detr:PekingU/rtdetr_r50vd | 0.602 | 0.242 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.804 | 0.477 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.758 | 0.460 |
| yolo:yolo11m.pt | 0.752 | 0.412 |
| hf_detr:PekingU/rtdetr_r50vd | 0.802 | 0.419 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.320 |
| torchvision:retinanet_resnet50_fpn_v2 | +0.268 |
| yolo:yolo11m.pt | +0.351 |
| hf_detr:PekingU/rtdetr_r50vd | +0.359 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (20 imgs, 9.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.577 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.804 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.656 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.357 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.641 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.710 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.149 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.603 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.670 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.528 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.754 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.732 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.608 |
| person | 0.546 |
| cyclist | 0.000 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (20 imgs, 3.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.256 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.477 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.250 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.209 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.255 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.571 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.099 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.292 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.321 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.343 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.306 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.629 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.491 |
| person | 0.278 |
| cyclist | 0.000 |

### torchvision:retinanet_resnet50_fpn_v2  ×  coco  (20 imgs, 2.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.526 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.758 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.586 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.278 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.647 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.648 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.142 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.559 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.635 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.485 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.741 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.693 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.603 |
| person | 0.448 |
| cyclist | 0.000 |

### torchvision:retinanet_resnet50_fpn_v2  ×  kitti  (20 imgs, 2.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.257 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.460 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.277 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.141 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.276 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.601 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.113 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.302 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.372 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.364 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.376 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.693 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.506 |
| person | 0.260 |
| cyclist | 0.006 |

### yolo:yolo11m.pt  ×  coco  (20 imgs, 2.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.579 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.752 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.619 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.270 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.723 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.744 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.143 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.592 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.632 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.335 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.828 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.757 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.611 |
| person | 0.546 |
| cyclist | 0.000 |

### yolo:yolo11m.pt  ×  kitti  (20 imgs, 1.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.228 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.412 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.096 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.252 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.598 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.093 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.276 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.285 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.207 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.299 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.636 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.480 |
| person | 0.202 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (20 imgs, 8.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.602 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.802 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.648 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.328 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.728 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.751 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.154 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.623 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.689 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.472 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.823 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.776 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.621 |
| person | 0.583 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (20 imgs, 3.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.242 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.419 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.248 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.140 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.259 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.606 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.097 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.300 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.329 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.640 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.514 |
| person | 0.213 |
| cyclist | 0.001 |
