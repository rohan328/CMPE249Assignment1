# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.568 | 0.254 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.496 | 0.253 |
| yolo:yolo11m.pt | 0.543 | 0.217 |
| hf_detr:PekingU/rtdetr_r50vd | 0.581 | 0.240 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.784 | 0.470 |
| torchvision:retinanet_resnet50_fpn_v2 | 0.709 | 0.443 |
| yolo:yolo11m.pt | 0.685 | 0.388 |
| hf_detr:PekingU/rtdetr_r50vd | 0.771 | 0.413 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.314 |
| torchvision:retinanet_resnet50_fpn_v2 | +0.243 |
| yolo:yolo11m.pt | +0.326 |
| hf_detr:PekingU/rtdetr_r50vd | +0.341 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (20 imgs, 23.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.568 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.784 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.650 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.346 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.627 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.710 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.149 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.603 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.644 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.478 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.719 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.732 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.602 |
| person | 0.534 |
| cyclist | 0.000 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (20 imgs, 5.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.254 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.470 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.247 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.208 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.249 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.567 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.099 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.292 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.311 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.335 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.291 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.621 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.483 |
| person | 0.278 |
| cyclist | 0.000 |

### torchvision:retinanet_resnet50_fpn_v2  ×  coco  (20 imgs, 4.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.496 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.709 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.548 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.221 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.640 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.645 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.136 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.513 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.536 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.258 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.692 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.678 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.571 |
| person | 0.420 |
| cyclist | 0.000 |

### torchvision:retinanet_resnet50_fpn_v2  ×  kitti  (20 imgs, 2.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.253 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.443 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.274 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.140 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.266 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.575 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.108 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.293 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.339 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.352 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.342 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.621 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.496 |
| person | 0.258 |
| cyclist | 0.004 |

### yolo:yolo11m.pt  ×  coco  (20 imgs, 3.0s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.543 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.685 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.602 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.229 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.652 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.744 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.143 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.565 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.574 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.255 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.709 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.757 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.589 |
| person | 0.497 |
| cyclist | 0.000 |

### yolo:yolo11m.pt  ×  kitti  (20 imgs, 0.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.217 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.388 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.220 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.091 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.230 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.596 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.093 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.253 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.257 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.185 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.260 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.633 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.467 |
| person | 0.184 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (20 imgs, 16.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.581 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.771 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.626 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.307 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.696 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.740 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.154 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.596 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.635 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.393 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.754 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.754 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.601 |
| person | 0.560 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (20 imgs, 2.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.240 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.413 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.245 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.252 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.601 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.097 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.289 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.294 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.302 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.279 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.633 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.511 |
| person | 0.209 |
| cyclist | 0.000 |
