# YOLOv8n Baseline Training Report — IBVAP Stage 1

**Project:** SIH26187 — Intelligent Border Video Analytics Platform (IBVAP)  
**Stage:** 1 — Baseline Person Detector  
**Date:** 2026-09-29  
**Status:** Complete

---

## 1. Environment

| Component | Value |
|---|---|
| OS | Windows 11 |
| CPU | 12th Gen Intel Core i5-12500H |
| GPU | None (CPU-only training) |
| RAM | ~15.6 GB |
| Python | 3.13.7 |
| PyTorch | 2.8.0+cpu |
| Ultralytics | 8.3.186 |
| CUDA | Unavailable |

---

## 2. Source Dataset

| Property | Value |
|---|---|
| Sequence | MOT17-05-FRCNN |
| Source path | `MOT17/MOT17/train/MOT17-05-FRCNN/` |
| Resolution | 640 x 480 |
| Frame rate | 14 FPS |
| Total frames | 837 |
| Image format | JPEG (000001.jpg - 000837.jpg) |
| Ground truth | gt/gt.txt (MOT format) |
| Total GT annotations | 8,013 |
| Pedestrian annotations (class=1, mark=1) | 6,917 |
| Non-pedestrian/ignored annotations | 1,096 |
| Unique pedestrian track IDs | 133 |

### Class Distribution in gt.txt

| Class ID | Class Name | Count |
|---|---|---|
| 1 | Pedestrian | 6,917 |
| 2 | Person on vehicle | 315 |
| 3 | Car | 196 |
| 4 | Bicycle | 315 |
| 6 | Non-motorized vehicle | 11 |
| 8 | Distractor | 24 |
| 9 | Occluder | 235 |

Only class 1 (pedestrian) with mark=1 (valid) was used for YOLO training.

---

## 3. Preprocessing

### Annotation Conversion

MOT format (x, y, width, height in pixels) was converted to YOLO normalized format:

```
x_center = (x + width/2) / 640
y_center = (y + height/2) / 480
w_norm   = width / 640
h_norm   = height / 480
```

- YOLO class ID: 0 (single class: person)
- Track IDs were NOT included in YOLO labels
- Bounding boxes were clamped to [0, 1] range
- Zero-area boxes were discarded
- 6,917 valid annotations converted with 0 errors

### Train/Validation Split

Chronological split (no random shuffling to prevent temporal data leakage):

| Split | Frame Range | Frames | Pedestrian Instances |
|---|---|---|---|
| Train | 1 - 585 | 585 (70%) | 4,719 |
| Validation | 586 - 837 | 252 (30%) | 2,198 |
| Total | 1 - 837 | 837 | 6,917 |

### Output Dataset Structure

```
IBVAP/datasets/MOT17_05_YOLO/
  images/
    train/          (585 images)
    val/            (252 images)
  labels/
    train/          (585 label files)
    val/            (252 label files)
  tracking_gt/
    MOT17-05-FRCNN_gt.txt  (original GT preserved)
  data.yaml
```

### Dataset Validation Results

| Check | Result |
|---|---|
| Total images | 837 |
| Train images | 585 |
| Validation images | 252 |
| Train label instances | 4,719 |
| Val label instances | 2,198 |
| Invalid labels | 0 |
| Missing labels | 0 |
| Missing images | 0 |
| Out-of-range bboxes | 0 |
| Malformed lines | 0 |

---

## 4. Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Pretrained weights | yolov8n.pt (COCO pretrained) |
| Parameters | 3,005,843 |
| GFLOPs | 8.1 |
| Layers (fused) | 72 |
| Model size | 6.2 MB |

---

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Image size | 640 |
| Batch size | 4 |
| Device | CPU |
| Workers | 2 |
| Optimizer | Auto (SGD) |
| Initial LR | 0.01 |
| Final LR factor | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Mosaic | 1.0 (disabled at epoch 20) |
| Close mosaic | 10 |
| Augmentations | HSV, translate, scale, flipLR, erasing |

### Command Used

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data=r'C:\Users\Atharva\Downloads\SeemaDrishti\IBVAP\datasets\MOT17_05_YOLO\data.yaml',
    epochs=30,
    imgsz=640,
    batch=4,
    device='cpu',
    workers=2,
    project=r'C:\Users\Atharva\Downloads\SeemaDrishti\IBVAP\training\runs\detect',
    name='mot17_05_baseline',
    exist_ok=True,
    verbose=True
)
```

---

## 6. Training Results

### Final Validation Metrics (best.pt)

| Metric | Value |
|---|---|
| Precision | 0.891 |
| Recall | 0.753 |
| mAP@50 | 0.846 |
| mAP@50-95 | 0.494 |

### Last Epoch Metrics (epoch 30)

| Metric | Value |
|---|---|
| Precision | 0.908 |
| Recall | 0.730 |
| mAP@50 | 0.839 |
| mAP@50-95 | 0.474 |

### Inference Speed (CPU)

| Stage | Time |
|---|---|
| Preprocess | 0.8 ms |
| Inference | 49.2 ms |
| Postprocess | 1.5 ms |
| Total | ~51.5 ms/image (~19 FPS on CPU) |

### Training Duration

| Metric | Value |
|---|---|
| Total time | 1.589 hours (~95 minutes) |
| Avg time per epoch | ~3.2 minutes |

### Loss Progression (Selected Epochs)

| Epoch | Box Loss | Cls Loss | DFL Loss | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 1.546 | 1.505 | 1.476 | 0.849 | 0.683 | 0.749 | 0.390 |
| 5 | 1.270 | 0.985 | 1.281 | 0.913 | 0.712 | 0.801 | 0.454 |
| 10 | 1.138 | 0.842 | 1.208 | 0.877 | 0.725 | 0.810 | 0.460 |
| 14 | 1.079 | 0.753 | 1.161 | 0.891 | 0.753 | 0.846 | 0.494 |
| 20 | 0.999 | 0.687 | 1.123 | 0.907 | 0.726 | 0.818 | 0.471 |
| 25 | 0.868 | 0.583 | 1.033 | 0.890 | 0.735 | 0.826 | 0.480 |
| 30 | 0.795 | 0.534 | 0.997 | 0.908 | 0.730 | 0.839 | 0.474 |

Best model was saved at epoch 14 (highest mAP@50-95 = 0.494).

---

## 7. Output Artifacts

| Artifact | Path |
|---|---|
| YOLO dataset | IBVAP/datasets/MOT17_05_YOLO/ |
| data.yaml | IBVAP/datasets/MOT17_05_YOLO/data.yaml |
| Best weights | IBVAP/training/runs/detect/mot17_05_baseline/weights/best.pt |
| Last weights | IBVAP/training/runs/detect/mot17_05_baseline/weights/last.pt |
| Results CSV | IBVAP/training/runs/detect/mot17_05_baseline/results.csv |
| Training curves | IBVAP/training/runs/detect/mot17_05_baseline/results.png |
| PR curve | IBVAP/training/runs/detect/mot17_05_baseline/BoxPR_curve.png |
| Confusion matrix | IBVAP/training/runs/detect/mot17_05_baseline/confusion_matrix.png |
| Preparation script | IBVAP/scripts/prepare_yolo_dataset.py |
| Tracking GT | IBVAP/datasets/MOT17_05_YOLO/tracking_gt/MOT17-05-FRCNN_gt.txt |
| This report | IBVAP/training/yolov8n_mot17_baseline_report.md |

---

## 8. Limitations and Notes

### What This Model IS

- A baseline person detector trained on a single MOT17 surveillance sequence
- A proof-of-concept for the IBVAP detection pipeline
- A starting point for iterative improvement

### What This Model IS NOT

- Not a border-specific detection model
- Not the final IBVAP production detector
- Not validated for real-time performance (CPU inference only)
- Not trained on border surveillance imagery
- Not a multi-class detector (person only, no vehicles)

### Known Limitations

1. **Single sequence**: Trained on only MOT17-05-FRCNN (837 frames, one camera angle, one location). Poor generalization expected on different scenes.
2. **Temporal data leakage risk**: Even with chronological split, consecutive frames within the training set are highly correlated (14 FPS video). Validation metrics may overestimate real-world performance.
3. **CPU training**: Training was limited to 30 epochs on CPU. More epochs with GPU could improve results.
4. **No image enhancement**: Raw frames used without dehazing, denoising, or super-resolution.
5. **Single class**: Only person class. Vehicles, animals, and other border-relevant objects not detected.
6. **No tracking integration**: Detection only. ByteTrack/tracking evaluation deferred to Stage 2.
7. **COCO pretrained initialization**: Transfer learning from COCO, which includes person class -- favorable bias but not border-specific.
8. **Validation mAP@50-95 plateau**: Best mAP@50-95 (0.494) achieved at epoch 14, suggesting possible overfitting on later epochs despite decreasing training loss.

### Recommendations for Next Steps

1. Train on additional MOT17 sequences for better generalization
2. Add GPU acceleration for faster experimentation
3. Integrate ByteTrack for multi-object tracking evaluation
4. Add vehicle detection classes
5. Experiment with image enhancement preprocessing
6. Evaluate on completely unseen sequences (cross-sequence validation)

---

*This report was generated as part of IBVAP Stage 1. Do not cite these metrics as production-grade performance.*
