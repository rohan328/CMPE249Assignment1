# Zero-Shot 2D Object Detection Across COCO and KITTI
## Track A: Experimental Evaluation of Faster R-CNN, RetinaNet, YOLO11, and RT-DETR

**Author:** Rohan Ohlan  
**Course / instructor:** CMPE 249 / Dr. Kai Kai Liu  
**Submission date:** November 2, 2026
**Experiment code and command-log link:** [Repository results archive](https://github.com/rohan328/CMPE249Assignment1/tree/main/results)

> Quantitative accuracy and latency results are drawn from the supplied experiment outputs. The repository results archive contains the metrics, reports, command logs, environment capture, GPU capture, and generated videos used in this report.

## Abstract

This study evaluates four COCO-pretrained 2D detectors on 5,000 COCO validation images and a 5,984-image labeled KITTI subset using the `ngdet` evaluation framework. Predictions and ground truth are mapped to a shared three-class driving taxonomy: vehicle, person, and cyclist. At a confidence threshold of 0.05, RT-DETR-R50 obtains the highest unified AP on COCO (51.71) and KITTI (24.32). YOLO11m provides the lowest measured prediction latency on an NVIDIA GeForce RTX 3060 Ti, averaging 25.26 ms per image (39.59 FPS), with KITTI AP of 23.12. Across models, the COCO-to-KITTI AP decrease ranges from 19.76 to 28.28 percentage points. Cyclist AP on KITTI is almost zero for every detector, exposing an important mismatch between COCO bicycle boxes and KITTI rider-and-bicycle boxes. The experiments examine cross-dataset performance and the speed–accuracy tradeoff. These are custom unified-taxonomy results, distinct from official COCO and KITTI leaderboard metrics. Raising the confidence threshold to 0.25 reduces AP and AR100 in all eight comparisons, changes some model rankings, and leaves RT-DETR first in both domains.

## 1. Research Questions and Model Selection

The main question is whether a detector's accuracy on its pretraining domain predicts its usefulness on an autonomous-driving domain without adaptation. COCO contains varied everyday scenes [5], while KITTI supplies driving imagery [6]. Evaluation across these domains tests transfer under changes in appearance, viewpoint, object scale, and annotation conventions.

Three questions guide the analysis: (1) How do AP and recall change from COCO to KITTI? (2) Which classes and object-size groups show the largest differences? (3) What accuracy is obtained at each measured latency on the available GPU? Fine-tuning is not required for this comparison, and no fine-tuning results are reported.

| Model specification used by ngdet | Detector family and architecture | Selection rationale |
| --- | --- | --- |
| `torchvision:fasterrcnn_resnet50_fpn_v2` | Two-stage, anchor-based ResNet-50/FPN with an RPN and RoI head | Established proposal-based baseline [1, 7, 8] |
| `torchvision:retinanet_resnet50_fpn_v2` | One-stage, anchor-based ResNet-50/FPN with dense classification and regression heads | Tests a dense detector using focal loss [2, 7, 8] |
| `yolo:yolo11m.pt` | One-stage, anchor-free YOLO11 medium with multiscale feature fusion and a dense detection head | Modern practical speed–accuracy configuration [9] |
| `hf_detr:PekingU/rtdetr_r50vd` | ResNet-50-vd, hybrid encoder, and transformer detection decoder | Tests a query-based detector without conventional NMS [4, 10] |

All configurations use COCO-pretrained closed-set detectors, satisfying the requirement for at least three configurations from at least two families. Older CNN models provide architectural baselines alongside YOLO11 and RT-DETR; the study does not claim to rank every current state-of-the-art detector.

Faster R-CNN shares image features between its proposal generator and region classifier [1]. RetinaNet addresses dense foreground/background imbalance with focal loss [2]. DETR introduced set prediction and bipartite assignment to reduce duplicate predictions without conventional NMS [3]; RT-DETR builds an efficient multiscale encoder and query selection mechanism around that approach [4]. These architectural differences motivate the experiments, but do not independently establish the cause of measured performance differences.

## 2. Datasets, Hardware, and Experimental Protocol

### 2.1 Dataset splits and evidence of execution

| Dataset | Images in every model's supplied output | Role | Split information |
| --- | --- | --- | --- |
| COCO 2017 | 5,000 | Same-domain reference for COCO-pretrained detectors | `val2017` images; annotations argument: `datasets/coco2017/annotations/instances_val2017.json` |
| KITTI object detection | 5,984 | Cross-dataset driving-domain evaluation | Reported split file: `ImageSets/train.txt`; reported SHA-256: `A74F0E28EB0F0245BFB256ECDE002CD9A2CCFDC07ECDC0E138349D80E0E1F134` |

The eight entries in `metrics.json` record one completed run for each model–dataset pair, with model identifiers, image counts, confidence thresholds, elapsed times, 12 AP/AR statistics, and per-class AP. Their reported elapsed times sum to 4,008.8 seconds. This is a sum of evaluator-run wall times, not a controlled inference benchmark.

**KITTI split clarification.** The report identifies `ImageSets/train.txt` and supplies a SHA-256 digest for that file; the evaluation output contains 5,984 images, consistent with an 80% split of KITTI's 7,481 labeled images. The upstream `ngdet/datasets.py` defaults to `split='train'`, and the reproduced commands do not override that split. Accordingly, the report treats the run as an evaluation on the **labeled KITTI training partition**, subject to the user-reported `train.txt` path/hash and any local loader changes. The actual list of evaluated IDs has not been supplied for independent verification. No KITTI fine-tuning was performed, so this is a zero-shot cross-dataset test rather than a fitted evaluation on KITTI training examples. Any future KITTI fine-tuning needs a disjoint holdout.

### 2.2 Hardware, software, and checkpoints

| Reproducibility item | Recorded value or required completion |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti, recorded in `latency_report.md` |
| GPU memory capacity / measured peak usage | 8GB |
| CPU, system RAM, operating system | Ryzen 7 5700G, Windows 11 |
| Python / PyTorch / torchvision | 3.12.16 / 2.15.0+cu132 / 0.29.2+cu132 |
| Transformers / Ultralytics / pycocotools | 5.22.1 / 8.4.198 / 2.0.11 |
| ngdet source revision and local modifications | Git commit `8418fc1bae85657190f11749483bf2e729b9702c`; captured local patch is empty |
| Checkpoint identity | Framework/model identifiers are recorded in the commands; resolved cache revisions and checkpoint hashes were not captured |
| Main accuracy confidence threshold | 0.05 for all eight runs, recorded in each output `_meta.score_thr` |
| Additional accuracy threshold | 0.25 for all eight additional runs, recorded in `results_025/full/metrics.json` |
| Accuracy precision / batch size | Default (FP32 / 1) |
| Latency precision | FP32 for all four models, recorded in `latency.json` |
| Latency image, dimensions, warmup, iterations | `datasets/Kitti/training/image_2/000000.png`; 1224 x 370 px (reported); 10 warmup passes; 100 timed passes per model |


The checkpoint names are supported by the respective framework sources [8–10]. The supplied commands and captured provenance establish the model specifications, dataset/threshold arguments, environment package versions, and `ngdet` commit. Resolved download revisions and checkpoint hashes were not captured, so exact weight-file identity remains a limitation.

### 2.3 Preprocessing and inference settings

The following settings were traced through the `ngdet` inference adapters and their underlying framework/checkpoint configurations [8–12]. They describe the **standard FP32 adapter path**, not a common manually imposed input tensor. Each adapter accepts a PIL RGB image and returns detections as absolute-pixel `xyxy` boxes in the original image's coordinate system. `ngdet` then maps the native labels to `driving3` and applies the common confidence cutoff. Because `ngdet` does not override all library parameters, values marked *library default* are inferred from the referenced implementation; the exact installed package revision and loaded checkpoint should be retained for full reproducibility.

**Preprocessing (before the neural network).**

| Detector | Image geometry | Pixel scaling and normalization | Implemented by |
| --- | --- | --- | --- |
| Faster R-CNN ResNet-50-FPN v2 | Preserve aspect ratio; target short edge 800 px and cap long edge at 1333 px | PIL RGB → CHW float `[0,1]`; ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` | `torchvision_det.py` → `to_tensor()` → `GeneralizedRCNNTransform` |
| RetinaNet ResNet-50-FPN v2 | Same 800/1333-px aspect-preserving transform | PIL RGB → CHW float `[0,1]`; same ImageNet mean/std | `torchvision_det.py` → `to_tensor()` → `GeneralizedRCNNTransform` |
| YOLO11m (`yolo11m.pt`) | `imgsz=640` target; aspect-preserving minimum-rectangle, stride-aligned letterbox padding when supported (`rect=True`) | Ultralytics native pixel rescaling; **not** Torchvision's ImageNet transform | `yolo.py` → Ultralytics `YOLO.predict()` |
| RT-DETR ResNet-50-vd (`PekingU/rtdetr_r50vd`) | Resize to **640 × 640** using bilinear interpolation; `do_pad=false` | Rescale by `1/255`; **no mean/std normalization** (`do_normalize=false`). Mean/std values appear in the checkpoint config but are **not applied** | `hf_detr.py` → `AutoImageProcessor` |

**Inference and post-processing (after the neural network).**

| Detector | Native score filtering and duplicate handling | Selection/output limit | Additional unified step |
| --- | --- | --- | --- |
| Faster R-CNN v2 | RPN score floor `0.0` and RPN NMS IoU `0.7`; RoI score floor `0.05` and class-wise box NMS IoU `0.5` | RPN top `1000` proposals pre-NMS and `1000` post-NMS at test time; **100** final detections/image | Original-image absolute `xyxy`; then `ngdet` `score_thr=0.05` or `0.25` and class mapping |
| RetinaNet v2 | Native score floor `0.05`; anchor decoding and class-aware NMS IoU `0.5` | Top `1000` candidates **per feature level** before NMS; **300** detections/image | Original-image absolute `xyxy`; then same shared threshold and mapping |
| YOLO11m | `ngdet` calls `YOLO.predict(..., conf=0.0, half=False)` in FP32 mode; native class-aware NMS IoU `0.7` remains active | `max_det=300` | Boxes rescaled to original-image `xyxy`; then shared threshold and mapping |
| RT-DETR R50-vd | Processor called with `threshold=0.0`; **NMS-free** set-prediction inference | **300 object queries**, not an NMS output cap | `post_process_object_detection(..., target_sizes=[[original_height, original_width]])` restores absolute `xyxy`; then shared threshold and mapping |

These are *library defaults* for geometry, native filters and detection caps wherever `ngdet` does not override them; the explicit adapter arguments are identified in the table. Inputs are therefore **not identical tensors across models**. In particular, YOLO's `imgsz=640` is a target and may produce a smaller rectangular tensor, while RT-DETR resizes directly to 640 × 640.

**Native-versus-shared thresholds.** The common `--score-thr` is applied in `BaseDetector._fold_to_unified()` **after** the model adapter obtains native detections. At both experimental cutoffs, Faster R-CNN and RetinaNet first apply their built-in `0.05` filtering and NMS. YOLO11m instead receives `conf=0.0` but still performs Ultralytics NMS **before** the unified cutoff; RT-DETR uses the processor's `threshold=0.0` and no NMS. Thus, `--score-thr 0.05` and `--score-thr 0.25` are consistent **outer** operating thresholds, not proof of identical internal proposal selection, score calibration, or suppression across architectures. This difference is relevant to the comparisons in Sections 6 and 7.

**Implementation trace and source evidence.** In [`ngdet/detectors/torchvision_det.py`](https://github.com/rohan328/CMPE249Assignment1/blob/main/DeepDataMiningLearning/ngdet/detectors/torchvision_det.py), `TorchvisionDetector.__init__()` loads `get_model_weights(...).DEFAULT` with `get_model(...)`, while `predict()` calls `to_tensor(image)`, `self.model([img_t])` and `_fold_to_unified()`. [`ngdet/detectors/yolo.py`](https://github.com/rohan328/CMPE249Assignment1/blob/main/DeepDataMiningLearning/ngdet/detectors/yolo.py), `YOLODetector.predict()`, supplies `conf=0.0` to `self.model.predict(...)` without overriding `imgsz`, `iou`, `rect`, or `max_det`. [`ngdet/detectors/hf_detr.py`](https://github.com/rohan328/CMPE249Assignment1/blob/main/DeepDataMiningLearning/ngdet/detectors/hf_detr.py), `HFDetrDetector.predict()`, invokes `self.processor(images=image, return_tensors="pt")` and `post_process_object_detection(..., threshold=0.0, target_sizes=...)`. [`ngdet/detectors/base.py`](https://github.com/rohan328/CMPE249Assignment1/blob/main/DeepDataMiningLearning/ngdet/detectors/base.py), `BaseDetector._fold_to_unified()`, performs the common score filter and label mapping. Framework defaults come from [Torchvision Faster R-CNN](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py) and [RetinaNet](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py), [Ultralytics prediction arguments](https://docs.ultralytics.com/modes/predict/), and the [RT-DETR checkpoint's `preprocessor_config.json`](https://huggingface.co/PekingU/rtdetr_r50vd/blob/main/preprocessor_config.json). These source locations can be cited directly alongside [8–12].

**Reproducibility caveat:** The reported preprocessing and unmodified inference defaults were checked against the repository and published framework/checkpoint configurations, whereas the supplied metric JSONs themselves store only the *outer* `score_thr`. The executed repository commit, installed framework versions, and resolved checkpoint revisions should be retained to establish their exact state on the evaluation machine.

## 3. Taxonomy and Metric Definition

### 3.1 Shared label mapping

`ngdet` maps class names rather than relying on a common numeric index. This handles differences between torchvision/Hugging Face COCO indexing and Ultralytics' class indexing. The supplied `report.md` identifies the active preset as `driving3` [11].

| Unified class | COCO categories contributing to the mapping | KITTI labels contributing to the mapping |
| --- | --- | --- |
| Vehicle | car, truck, bus, train, motorcycle | Car, Van, Truck, Tram |
| Person | person | Pedestrian, Person_sitting |
| Cyclist | bicycle | Cyclist |
| Excluded | Categories outside these groups | Misc, DontCare |

The preset contains additional synonyms for other sources, but the table lists the relevant labels for these two datasets. Folding categories creates a coarse evaluation: a car/truck confusion within “vehicle” is no longer counted as a class error. Native-class NMS is applied before folding in relevant adapters, so mapping alone does not guarantee suppression of duplicates across merged classes.

The cyclist mapping is especially consequential. A COCO bicycle box and a KITTI Cyclist box can cover different extents even when the object is correctly recognized. A bicycle-only prediction can fail IoU matching against a rider-plus-bicycle box. Consequently, the measured cyclist difference combines transfer difficulty with label and geometry mismatch.

### 3.2 AP, AR, and evaluator limitations

AP averages interpolated precision across recall levels and IoU thresholds 0.50, 0.55, …, 0.95. AP50 and AP75 isolate looser and stricter overlap requirements. Small, medium, and large use the COCO area ranges based on 32² and 96² pixels; AR1/AR10/AR100 limit evaluated detections per image and category [13]. All tables below express AP and AR on a **0–100 scale**; differences are percentage points (pp). The JSON stores fractions on a 0–1 scale.

The inspected `COCOUnifiedEvaluator.add()` rebuilds annotations from mapped boxes, sets `iscrowd=0`, and calculates area from box width × height. It does not preserve native COCO segmentation areas/crowd semantics or KITTI DontCare and difficulty handling. These results therefore use the COCO evaluation engine with a custom annotation protocol. They are neither official 80-class COCO AP nor official KITTI easy/moderate/hard AP. The COCO–KITTI gap must also be interpreted as a combined domain-and-annotation shift, rather than a controlled estimate of visual domain shift alone.

## 4. Quantitative Results at Confidence Threshold 0.05

### 4.1 Overall accuracy and transfer gap

| Model | COCO AP | KITTI AP | Gap (pp) | AP retained (%) |
| --- | --- | --- | --- | --- |
| Faster R-CNN v2 | 49.15 | 23.61 | 25.54 | 48.04 |
| RetinaNet v2 | 43.48 | 23.72 | 19.76 | 54.55 |
| YOLO11m | 51.40 | 23.12 | 28.28 | 44.97 |
| RT-DETR-R50 | 51.71 | 24.32 | 27.38 | 47.04 |

Gap = COCO AP − KITTI AP; retention = 100 × KITTI AP / COCO AP. RT-DETR ranks first on both datasets, although its COCO lead over YOLO11m is only 0.31 pp. The COCO model spread is 8.22 pp, while the KITTI spread is 1.21 pp. Thus, the evaluated driving task compresses the differences between these particular pretrained configurations. No confidence intervals or repeated-run estimates are available, so small differences are descriptive rather than evidence of statistical significance.

### 4.2 AP at different IoUs and object scales

| Model / dataset | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 49.15 | 75.21 | 52.27 | 28.60 | 49.94 | 65.96 |
| Faster R-CNN v2 / KITTI | 23.61 | 43.73 | 23.44 | 11.98 | 23.12 | 31.22 |
| RetinaNet v2 / COCO | 43.48 | 69.02 | 44.78 | 21.78 | 45.15 | 61.00 |
| RetinaNet v2 / KITTI | 23.72 | 44.85 | 22.72 | 12.95 | 22.85 | 31.57 |
| YOLO11m / COCO | 51.40 | 72.15 | 55.20 | 24.79 | 53.50 | 72.51 |
| YOLO11m / KITTI | 23.12 | 43.14 | 22.77 | 11.48 | 22.05 | 30.59 |
| RT-DETR-R50 / COCO | 51.71 | 75.42 | 54.15 | 26.62 | 52.66 | 72.60 |
| RT-DETR-R50 / KITTI | 24.32 | 44.91 | 23.68 | 13.14 | 23.40 | 32.28 |

### 4.3 Recall at detection caps and object scales

| Model / dataset | AR1 | AR10 | AR100 | AR small | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 23.16 | 55.28 | 60.57 | 43.80 | 62.86 | 74.47 |
| Faster R-CNN v2 / KITTI | 10.96 | 31.62 | 34.13 | 31.84 | 33.44 | 37.84 |
| RetinaNet v2 / COCO | 21.93 | 50.16 | 57.81 | 38.59 | 61.56 | 72.09 |
| RetinaNet v2 / KITTI | 11.56 | 33.19 | 37.95 | 31.57 | 38.38 | 42.85 |
| YOLO11m / COCO | 23.61 | 56.04 | 60.26 | 36.23 | 64.61 | 78.84 |
| YOLO11m / KITTI | 10.62 | 29.97 | 31.41 | 24.71 | 31.37 | 37.05 |
| RT-DETR-R50 / COCO | 23.59 | 57.57 | 66.67 | 48.26 | 68.48 | 82.86 |
| RT-DETR-R50 / KITTI | 10.81 | 31.48 | 35.33 | 30.83 | 34.96 | 39.71 |

### 4.4 Per-class AP

| Model / dataset | Vehicle AP | Person AP | Cyclist AP |
| --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 53.13 | 56.47 | 37.85 |
| Faster R-CNN v2 / KITTI | 45.12 | 25.60 | 0.12 |
| RetinaNet v2 / COCO | 47.80 | 51.07 | 31.58 |
| RetinaNet v2 / KITTI | 46.41 | 24.56 | 0.20 |
| YOLO11m / COCO | 55.29 | 59.01 | 39.88 |
| YOLO11m / KITTI | 45.65 | 23.61 | 0.09 |
| RT-DETR-R50 / COCO | 56.28 | 59.32 | 39.52 |
| RT-DETR-R50 / KITTI | 48.97 | 23.85 | 0.16 |

Tables 4.2–4.4 contain all 12 COCO-style summary statistics and the three per-class AP values for every run. They were calculated directly from the attached JSON rather than copied from published model benchmarks.

## 5. Technical Analysis I: Cross-Dataset Generalization

### 5.1 Class-dependent degradation

Vehicles transfer substantially better than people in this evaluation. Vehicle AP falls by 8.02 pp for Faster R-CNN, 1.39 pp for RetinaNet, 9.64 pp for YOLO11m, and 7.32 pp for RT-DETR. Person AP falls by 30.87, 26.52, 35.41, and 35.47 pp, respectively. KITTI cyclist AP ranges from approximately 0.09 to 0.20, compared with 31.58–39.88 on COCO. This is the largest class-level difference, but the incompatible box definitions prevent interpreting it solely as failure to recognize cyclists.

RetinaNet has the smallest overall gap (19.76 pp) and highest retention (54.55%). However, it starts from the lowest COCO AP and does not obtain the highest KITTI AP. A smaller gap is therefore insufficient by itself to establish the best target-domain detector. RT-DETR has the strongest vehicle AP on KITTI (48.97), whereas Faster R-CNN has the strongest KITTI person AP (25.60).

As a diagnostic, the arithmetic mean of KITTI vehicle and person AP is 35.36 for Faster R-CNN, 35.48 for RetinaNet, 34.63 for YOLO11m, and 36.41 for RT-DETR. This is derived from existing per-class results with the original prediction pipeline unchanged; it is not a rerun under a different taxonomy. It shows that the near-zero cyclist score strongly affects the three-class mean, while substantial vehicle/person transfer gaps remain.

### 5.2 Object scale and localization

Within each dataset, small-object AP is below large-object AP for every model. On KITTI, RT-DETR has the highest small-object AP (13.14), followed by RetinaNet (12.95), Faster R-CNN (11.98), and YOLO11m (11.48). On COCO, Faster R-CNN has the highest small-object AP (28.60), despite not having the highest overall AP. The model ranking consequently depends on both the domain and the evaluated scale.

Measured small-object AP drops range from 8.83 to 16.62 pp, whereas large-object AP drops range from 29.43 to 41.92 pp. Therefore, the tables do not support a claim that small objects show the largest absolute cross-dataset decrease. Small objects remain difficult within KITTI, while the absolute large-object gaps are greater. Different class composition and box definitions within each area bin limit a causal scale-only interpretation.

KITTI AP50 substantially exceeds AP75 for all models. This is consistent with stricter localization exposing additional errors, but AP aggregates misses, classification, ranking, and box quality; the AP50–AP75 difference cannot isolate localization alone. RT-DETR has the highest KITTI AP75 (23.68), while RetinaNet has the highest KITTI AR100 (37.95). These rankings show that recall and precision/localization summaries reward different properties.

## 6. Technical Analysis II: Architecture and Speed–Accuracy Tradeoff

### 6.1 Measured prediction latency

| Model | Mean ms | Median ms | p90 ms | Std. dev. ms | FPS | KITTI AP |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 | 67.42 | 67.17 | 70.90 | 2.28 | 14.83 | 23.61 |
| RetinaNet v2 | 48.12 | 47.96 | 51.26 | 2.51 | 20.78 | 23.72 |
| YOLO11m | 25.26 | 24.91 | 25.90 | 1.72 | 39.59 | 23.12 |
| RT-DETR-R50 | 63.49 | 62.53 | 69.02 | 5.20 | 15.75 | 24.32 |

Device: **NVIDIA GeForce RTX 3060 Ti**. All four entries use the recorded FP32 mode. FPS is 1000 divided by mean latency, not a separate batched-throughput measurement. The inspected `latency.time_predict()` performs warmup, calls `predict()` repeatedly, and synchronizes CUDA before collecting timings [11]. File loading and model loading occur outside the timed prediction loop; preprocessing, prediction, output conversion, and post-processing occur inside it.

The latency command reproduced in Section 8.3 specifies KITTI image `training/image_2/000000.png`, 10 warmup passes, 100 timed passes, `--accel fp32`, `--device cuda`, and `driving3`. The reported image dimensions are 1224 x 370 px. In the evaluated `ngdet` source, `benchmark()` constructs detectors using **`score_thr=0.30`** (there is no latency CLI flag for the threshold), independently of the accuracy runs at 0.05 and 0.25. Consequently, the latency and AP figures characterize the same model configurations but use different score filters.

### 6.2 Interpreting the tradeoff

YOLO11m is approximately 2.51× as fast as RT-DETR and 2.67× as fast as Faster R-CNN in this timing setup. Its KITTI AP is 1.21 pp below RT-DETR. This makes it the strongest measured speed choice, while RT-DETR offers the highest observed target-domain AP. RetinaNet averages 48.12 ms with 23.72 KITTI AP. For the specific objectives of lower mean latency and higher three-class KITTI AP, YOLO11m, RetinaNet, and RT-DETR are nondominated; Faster R-CNN is dominated by RetinaNet and RT-DETR. Faster R-CNN nevertheless retains the highest person AP, so a pedestrian-specific objective changes the decision.

Faster R-CNN includes proposal generation and region-level processing. RetinaNet and YOLO use dense heads over multiscale features, while RT-DETR processes selected queries with a transformer decoder. These differences help explain why the models have different computational profiles, but input size, backbone capacity, implementation, and post-processing also vary. The results do not isolate an architecture-only effect.

The lack of conventional NMS does not make RT-DETR the fastest detector in this experiment: its encoder/decoder and framework overhead still contribute to latency. Likewise, a paper's “real-time” result on a different GPU or optimized backend does not predict this native FP32 result. No FP16, TensorRT, ONNX, or compilation speedup is claimed because only FP32 measurements are supplied.

## 7. Confidence-Threshold Sensitivity and Failure Analysis

### 7.1 Measured threshold sensitivity: 0.05 versus 0.25

The additional `metrics(1).json` and `report(1).md` contain all eight completed evaluations at threshold 0.25. Model identifiers and image counts match the 0.05 outputs: 5,000 COCO images and 5,984 KITTI images per model. The JSON verifies `_meta.score_thr=0.25` for every new entry. Equal counts do not establish identical image IDs, weights, or preprocessing; those remain subject to command-log and manifest verification.

Differences below are calculated from unrounded JSON values as Δ = metric(0.25) − metric(0.05). AP and AR use the same 0–100 scale as Section 4.

| Model / dataset | AP 0.05 | AP 0.25 | ΔAP (pp) | AR100 0.05 | AR100 0.25 | ΔAR100 (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 49.15 | 48.36 | -0.79 | 60.57 | 57.69 | -2.88 |
| Faster R-CNN v2 / KITTI | 23.61 | 23.46 | -0.15 | 34.13 | 32.91 | -1.22 |
| RetinaNet v2 / COCO | 43.48 | 42.21 | -1.27 | 57.81 | 51.08 | -6.72 |
| RetinaNet v2 / KITTI | 23.72 | 23.42 | -0.30 | 37.95 | 32.87 | -5.09 |
| YOLO11m / COCO | 51.40 | 46.93 | -4.47 | 60.26 | 51.92 | -8.34 |
| YOLO11m / KITTI | 23.12 | 22.46 | -0.65 | 31.41 | 29.41 | -2.00 |
| RT-DETR-R50 / COCO | 51.71 | 50.15 | -1.56 | 66.67 | 58.75 | -7.92 |
| RT-DETR-R50 / KITTI | 24.32 | 23.98 | -0.35 | 35.33 | 31.38 | -3.94 |

**Effect on accuracy and model ranking.** AP decreases for every model–dataset pair. YOLO11m has the largest AP decrease on both COCO (−4.47 pp) and KITTI (−0.65 pp). Faster R-CNN is the least sensitive by AP in both domains, decreasing by 0.79 pp and 0.15 pp, respectively. Every model has a larger absolute AP decrease on COCO than on KITTI. These results show that retaining lower-confidence predictions contributes to measured AP; they do not show that every retained low-confidence box is correct.

RT-DETR remains first on both datasets. The COCO order changes from RT-DETR > YOLO11m > Faster R-CNN > RetinaNet at 0.05 to RT-DETR > Faster R-CNN > YOLO11m > RetinaNet at 0.25. KITTI changes from RT-DETR > RetinaNet > Faster R-CNN > YOLO11m to RT-DETR > Faster R-CNN > RetinaNet > YOLO11m. Faster R-CNN's KITTI lead over RetinaNet at 0.25 is only 0.04 pp; without uncertainty estimates, this is a descriptive reversal rather than a statistically established advantage.

**Effect on recall and classes.** AR100 also decreases in every pair. YOLO11m has the largest COCO AR100 decrease (8.34 pp), whereas RetinaNet has the largest KITTI decrease (5.09 pp). On KITTI, the largest absolute per-class AP decrease is in the vehicle class for each model. For YOLO11m, vehicle AP drops by 1.08 pp and person AP by 0.88 pp, while cyclist AP remains near zero. Thus, the measured threshold sensitivity is not concentrated exclusively in pedestrians or cyclists. COCO YOLO11m loses about 4.30–4.66 pp across all three classes.

Raising the cutoff removes lower-score candidates and limits the recall levels available to AP evaluation. Higher precision at a selected operating point is possible, but the supplied AP/AR summaries do not measure operating-point precision or false-positive counts. The measured decreases support using the lower cutoff for this study's AP reporting; they do not identify the best deployment threshold. Score calibration and native filtering differ across model families, so equal numerical cutoffs do not imply equal operating points. No new latency measurements at threshold 0.25 were supplied.

#### Complete AP results at threshold 0.25

| Model / dataset | AP | AP50 | AP75 | AP small | AP medium | AP large |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 48.36 | 73.66 | 51.56 | 27.97 | 49.06 | 65.06 |
| Faster R-CNN v2 / KITTI | 23.46 | 43.43 | 23.38 | 11.94 | 22.80 | 30.86 |
| RetinaNet v2 / COCO | 42.21 | 66.29 | 43.94 | 20.60 | 43.79 | 60.02 |
| RetinaNet v2 / KITTI | 23.42 | 44.08 | 22.52 | 12.85 | 22.29 | 30.55 |
| YOLO11m / COCO | 46.93 | 63.99 | 50.64 | 20.33 | 49.01 | 69.51 |
| YOLO11m / KITTI | 22.46 | 41.23 | 22.39 | 10.94 | 21.27 | 30.25 |
| RT-DETR-R50 / COCO | 50.15 | 72.77 | 52.64 | 25.18 | 51.12 | 71.02 |
| RT-DETR-R50 / KITTI | 23.98 | 44.09 | 23.44 | 13.00 | 22.88 | 31.42 |

#### Complete recall results at threshold 0.25

| Model / dataset | AR1 | AR10 | AR100 | AR small | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 23.06 | 53.87 | 57.69 | 40.55 | 60.04 | 71.88 |
| Faster R-CNN v2 / KITTI | 10.87 | 31.26 | 32.91 | 30.52 | 32.15 | 36.89 |
| RetinaNet v2 / COCO | 21.46 | 47.58 | 51.08 | 30.51 | 54.01 | 67.93 |
| RetinaNet v2 / KITTI | 10.94 | 31.16 | 32.87 | 27.28 | 32.85 | 38.01 |
| YOLO11m / COCO | 23.00 | 50.68 | 51.92 | 24.68 | 55.90 | 74.02 |
| YOLO11m / KITTI | 10.42 | 28.93 | 29.41 | 21.08 | 29.25 | 36.35 |
| RT-DETR-R50 / COCO | 23.38 | 54.81 | 58.75 | 36.78 | 61.50 | 77.29 |
| RT-DETR-R50 / KITTI | 10.60 | 30.20 | 31.38 | 25.40 | 31.00 | 37.28 |

#### Per-class AP at threshold 0.25

| Model / dataset | Vehicle AP | Person AP | Cyclist AP |
| --- | --- | --- | --- |
| Faster R-CNN v2 / COCO | 52.18 | 55.74 | 37.16 |
| Faster R-CNN v2 / KITTI | 44.80 | 25.48 | 0.12 |
| RetinaNet v2 / COCO | 46.19 | 49.83 | 30.63 |
| RetinaNet v2 / KITTI | 45.88 | 24.26 | 0.13 |
| YOLO11m / COCO | 50.84 | 54.35 | 35.58 |
| YOLO11m / KITTI | 44.58 | 22.73 | 0.08 |
| RT-DETR-R50 / COCO | 54.73 | 57.93 | 37.80 |
| RT-DETR-R50 / KITTI | 48.36 | 23.42 | 0.15 |

**0.25-sweep command:** The second accuracy command in Section 8.3 specifies the same models, datasets, taxonomy, `--max-images 0`, device, annotation file, and video settings as the 0.05 sweep. It changes `--score-thr` to `0.25` and the output directory to `results_025/full`. The corresponding command log and aggregate metrics are included in the repository results archive. Retained prediction counts and operating-point precision are not included in the supplied outputs.

### 7.2 Aggregate failure evidence

The metrics establish three aggregate problems: near-zero KITTI cyclist AP, low KITTI small-object AP, and lower KITTI person AP relative to COCO. They do not reveal which individual predictions caused those values. Occlusion, truncation, distant objects, and inconsistent box extents are plausible explanations requiring visual confirmation.

### 7.3 Qualitative evidence

The evaluation generated annotated videos for up to 20 frames per model and dataset in both accuracy sweeps. These videos are retained in `results/full` and `results_025/full` and provide qualitative evidence alongside the aggregate failure analysis above. No individual image-level claim is used to support the quantitative conclusions; the report therefore does not infer a specific failure case from an uninspected frame.

## 8. Limitations, Conclusions, and Reproducibility

### 8.1 Limitations

The shared taxonomy simplifies evaluation but merges vehicle categories and mismatches cyclist geometry. Crowd, ignored-region, difficulty, and area semantics differ from native benchmarks. The evaluated KITTI split is identified as `ImageSets/train.txt`, but the split file itself and its image-ID manifest were not archived. Model-specific resize, normalization, NMS policies, and internal filtering mean this is a comparison of complete inference configurations rather than a controlled, identical-input architecture ablation. Latency covers the recorded timing setup rather than a dataset-wide distribution or deployment pipeline; peak inference memory and accuracy uncertainty were not supplied. Resolved checkpoint revisions and hashes were not captured.

### 8.2 Conclusions

On the supplied runs, RT-DETR-R50 obtains the highest unified AP on both domains, while YOLO11m provides the fastest prediction on the RTX 3060 Ti. Increasing the confidence threshold to 0.25 decreases AP for all four models on both datasets and changes intermediate rankings, while RT-DETR remains first. The best COCO configuration does not eliminate the large drop on KITTI, and the strongest model depends on whether the objective is overall AP, pedestrian AP, recall, or latency. The nearly universal cyclist collapse demonstrates why taxonomy and box-definition checks are essential in cross-dataset evaluation. These results support evaluating complete model-and-data protocols rather than selecting a detector from published benchmark rankings alone.

An appropriate next experiment would preserve a verified KITTI holdout and compare pretrained performance with adaptation on a disjoint training subset. That is future work; no fine-tuning improvement is claimed here.

### 8.3 Execution evidence and commands

| Model | COCO images / seconds | KITTI images / seconds |
| --- | --- | --- |
| Faster R-CNN v2 | 5,000 / 538.5 | 5,984 / 742.7 |
| RetinaNet v2 | 5,000 / 416.3 | 5,984 / 622.6 |
| YOLO11m | 5,000 / 197.7 | 5,984 / 424.5 |
| RT-DETR-R50 | 5,000 / 385.2 | 5,984 / 681.3 |

The elapsed times above are recorded by the evaluator and can include loading, downloads, video generation, and metric computation. They should not be converted into the latency figures in Section 6. Raw evidence supplied for this report is `metrics.json` and `report.md` at threshold 0.05; `metrics(1).json` and `report(1).md` at threshold 0.25; and `latency.json` and `latency_report.md` for the FP32 timing study. The eight 0.25 runs record a combined evaluator wall time of 4,019.5 seconds. Both accuracy sweeps together contain 16 completed model–dataset–threshold evaluations and 8,028.3 seconds of recorded run time; these totals are execution evidence, not controlled speed measurements.

**Evaluation commands recorded in the report.** The following are the command arguments provided for the actual runs, reformatted as valid multiline Bash commands. The two accuracy sweeps differ only in `--score-thr` and `--out-dir`. The submitted command text did **not** provide a `--roots` override; `ngdet` therefore uses the `DEFAULT_ROOTS` configured in the **locally executed** `run_eval.py`. The uploaded report notes that dataset paths were modified locally. Thus the resolved COCO image root and KITTI root **cannot be recovered from the command strings alone**. By contrast, the explicit COCO annotation file and latency image arguments are recorded below. The upstream default root values should not be silently substituted for locally changed paths. See [`ngdet/run_eval.py`](https://github.com/rohan328/CMPE249Assignment1/blob/main/DeepDataMiningLearning/ngdet/run_eval.py) (`DEFAULT_ROOTS`, `--roots`) for resolution logic.

```bash
# Accuracy sweep A: unified COCO and KITTI AP at score threshold 0.05
# Working directory: DeepDataMiningLearning repository root
python -m DeepDataMiningLearning.ngdet.run_eval \
  --models \
    torchvision:fasterrcnn_resnet50_fpn_v2 \
    torchvision:retinanet_resnet50_fpn_v2 \
    yolo:yolo11m.pt \
    hf_detr:PekingU/rtdetr_r50vd \
  --datasets coco kitti \
  --taxonomy driving3 \
  --max-images 0 \
  --score-thr 0.05 \
  --device cuda \
  --video --video-max 20 \
  --coco-ann "datasets/coco2017/annotations/instances_val2017.json" \
  --out-dir results/full

# Accuracy sweep B: same model/data arguments, higher outer score cutoff
python -m DeepDataMiningLearning.ngdet.run_eval \
  --models \
    torchvision:fasterrcnn_resnet50_fpn_v2 \
    torchvision:retinanet_resnet50_fpn_v2 \
    yolo:yolo11m.pt \
    hf_detr:PekingU/rtdetr_r50vd \
  --datasets coco kitti \
  --taxonomy driving3 \
  --max-images 0 \
  --score-thr 0.25 \
  --device cuda \
  --video --video-max 20 \
  --coco-ann "datasets/coco2017/annotations/instances_val2017.json" \
  --out-dir results_025/full

# Latency: native FP32 inference on a single KITTI image
python -m DeepDataMiningLearning.ngdet.latency \
  --models \
    torchvision:fasterrcnn_resnet50_fpn_v2 \
    torchvision:retinanet_resnet50_fpn_v2 \
    yolo:yolo11m.pt \
    hf_detr:PekingU/rtdetr_r50vd \
  --accel fp32 \
  --image "datasets/Kitti/training/image_2/000000.png" \
  --taxonomy driving3 \
  --device cuda \
  --warmup 10 --iters 100 \
  --out-dir results/latency
```

**Reproducibility details resolved by the commands:** Both accuracy runs request all available images (`--max-images 0`), select COCO and KITTI, use the three-class `driving3` taxonomy and CUDA, and save short annotated videos capped at **20 frames per model/dataset**; that frame cap does *not* limit the images evaluated for AP. The expected accuracy outputs include `metrics.json`, `report.md`, per-image prediction-count JSONs, heatmaps, and the selected videos, written into different directories so threshold runs do not overwrite each other. The latency run loads one KITTI image, measures FP32 prediction after 10 warmup iterations, and collects 100 synchronized iterations per model; the output directory is `results/latency`. The model specs above serve as checkpoint identifiers, but their exact resolved download revisions remain unrecorded.

**Additional provenance commands** (recommended for archiving; not established as already executed):

```bash
# Run from the same repository/environment used for inference.
python -m pip freeze > results/requirements-lock.txt
git rev-parse HEAD > results/repository-commit.txt
nvidia-smi > results/gpu.txt

# Replace KITTI_ROOT with the dataset root resolved by the local run_eval.py.
# This command checks the split hash that is reported in Section 2.1.
Get-FileHash .\datasets\Kitti\ImageSets\train.txt \
  > results/kitti-split-sha256.txt
```

The reported KITTI split digest is `A74F0E28EB0F0245BFB256ECDE002CD9A2CCFDC07ECDC0E138349D80E0E1F134`. The actual shell output is archived in `results/repository-commit.txt` and the accuracy/latency `command.log` files. The resolved dataset roots, checkpoint hashes, and evaluated image-ID manifest were not captured.

**Code/configuration archive:** [Repository results archive](https://github.com/rohan328/CMPE249Assignment1/tree/main/results), including the threshold-specific outputs in [`results_025`](https://github.com/rohan328/CMPE249Assignment1/tree/main/results_025). The archive contains the command logs, source commit, local patch capture, environment lockfile, GPU capture, metrics, latency results, and generated videos. Checkpoint hashes and the evaluated split-ID manifest were not captured.

## References

1. S. Ren, K. He, R. Girshick, and J. Sun. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” NeurIPS, 2015. [Original paper](https://arxiv.org/abs/1506.01497).
2. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár. “Focal Loss for Dense Object Detection.” ICCV, 2017. [Original paper](https://arxiv.org/abs/1708.02002).
3. N. Carion et al. “End-to-End Object Detection with Transformers.” ECCV, 2020. [Original paper](https://arxiv.org/abs/2005.12872).
4. Y. Zhao et al. “DETRs Beat YOLOs on Real-time Object Detection.” CVPR, 2024; preprint 2023. [Original paper](https://arxiv.org/abs/2304.08069).
5. T.-Y. Lin et al. “Microsoft COCO: Common Objects in Context.” ECCV, 2014. [Original paper](https://arxiv.org/abs/1405.0312).
6. A. Geiger, P. Lenz, and R. Urtasun. “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR, 2012. [Official benchmark and publication information](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
7. T.-Y. Lin et al. “Feature Pyramid Networks for Object Detection.” CVPR, 2017. [Original paper](https://arxiv.org/abs/1612.03144).
8. PyTorch/Torchvision maintainers. [Official detection source repository](https://github.com/pytorch/vision/tree/main/torchvision/models/detection) and [Faster R-CNN v2 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html).
9. Ultralytics. [YOLO11 official documentation](https://docs.ultralytics.com/models/yolo11/) and [official implementation](https://github.com/ultralytics/ultralytics). YOLO11 architecture/checkpoint details are attributed to the implementation and documentation rather than to an assumed YOLO11 research paper.
10. RT-DETR authors and Hugging Face maintainers. [Official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR), [RT-DETR framework documentation](https://huggingface.co/docs/transformers/model_doc/rt_detr), and [evaluated checkpoint card](https://huggingface.co/PekingU/rtdetr_r50vd).
11. L. Liu / lkk688. *DeepDataMiningLearning: ngdet tutorial and implementation.* [Assigned tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md). Relevant files: `run_eval.py`, `datasets.py`, `taxonomy.py`, `evaluator.py`, `latency.py`, and `detectors/`. Evaluated commit: `8418fc1bae85657190f11749483bf2e729b9702c`.
12. Ultralytics. [Official prediction arguments and preprocessing documentation](https://docs.ultralytics.com/modes/predict/).
13. COCO maintainers. [Official evaluation implementation, `COCOeval`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) and [detection metric definition](https://cocodataset.org/dataset/detection-eval.htm).

## AI Assistance Disclosure

ChatGPT/Codex assisted with interpreting the supplied output files, calculating derived comparisons, organizing the report, and identifying relevant technical sources. The numerical experiment evidence is the supplied JSON, report, command-log, and environment outputs; AI-generated explanations do not substitute for measured results or inspected examples. The final claims are limited to the captured artifacts and the limitations stated in this report.
