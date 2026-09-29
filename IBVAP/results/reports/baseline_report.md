# IBVAP YOLOv8n Baseline Evaluation

*Generated on: 2026-09-28 21:23:01 | SIH26187 Stage 1 Baseline*

## 1. Test Environment
- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.13.7
- **PyTorch:** 2.8.0+cpu
- **CUDA Available:** False
- **GPU:** N/A (CPU Mode)
- **GPU Memory / System RAM:** System RAM: 15.63 GB
- **Ultralytics:** 8.3.186
- **YOLO Model:** YOLOv8n (Pretrained Nano Weights: `yolov8n.pt`)
- **Confidence Threshold:** 0.25

## 2. Test Videos
- **cctv_sample_test.mp4:** Resolution `640x480`, Source FPS `25.0`, Total Frames `75`, Approx Duration `3.0s`

## 3. Runtime Performance

| Video | Resolution | Source FPS | Processing FPS | Avg Latency | Device |
|---|---|---|---|---|---|
| cctv_sample_test.mp4 | 640x480 | 25.0 | 9.28 | 104.62 ms | CPU |

## 4. Detection Statistics

| Video | Class | Detection Count | Mean Confidence | Min | Max |
|---|---|---|---|---|---|
| cctv_sample_test | `sports ball` | 75 | 0.52 | 0.31 | 0.75 |

## 5. Observations
- **Personnel Detection:** No human figures were detected above the 0.25 confidence threshold in the tested clips.
- **Vehicle Detection:** No vehicle classes were detected in the tested footage.
- **Inference Speed:** Average processing throughput across videos was **9.3 FPS** on `CPU`.
- **Small Object / Scale Sensitivity:** Distant boundary perimeter entities and low-resolution objects exhibit lower confidence scores, demonstrating standard generic COCO model limitations in long-range CCTV viewpoints.
- **Lighting & Environmental Factors:** In non-ideal surveillance conditions (shadows, low light, weather), generic features may miss partial occlusions without CCTV-specific domain adaptation.

> **Note:** The above observations reflect empirical runtime logging and qualitative checks. They are not calculated precision/recall accuracy scores.

## 6. Limitations
- **Generic Pretrained Weights:** The evaluated model is stock `yolov8n.pt` trained on MS COCO 80 classes, not specialized for high-angle fixed border surveillance cameras.
- **No Domain Fine-Tuning:** No domain-specific transfer learning has been applied yet.
- **No Ground-Truth Annotations:** Ground-truth bounding box datasets for these test videos are not yet labeled.
- **Accuracy Metrics Excluded:** Accuracy metrics (mAP@50, mAP@50-95, Precision, Recall, F1) are NOT calculated in this baseline report because ground-truth annotations have not yet been created.
- **Hardware Dependency:** Inference FPS and latency depend on the host hardware profile.

## 7. Next Step
> Create a representative CCTV dataset and ground-truth annotations, then compare pretrained YOLOv8n against a CCTV-fine-tuned YOLOv8n model.
