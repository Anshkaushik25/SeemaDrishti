# IBVAP — Intelligent Border Video Analytics Platform
**SIH26187 — AI-Based Intelligent Video Analytics Platform for Border Surveillance Using Existing CCTV Infrastructure**

---

## 📌 Project Overview (Stage 1: Baseline Evaluation)

This repository contains the **First Baseline Stage** of the **IBVAP** project. 

The objective of this stage is to establish an empirical performance and inference baseline of pretrained **YOLOv8n (Nano)** running on actual surveillance/CCTV footage across local hardware before undertaking dataset annotation or domain-specific fine-tuning.

---

## 📁 Project Structure

```
IBVAP/
│
├── videos/                  # Place CCTV / surveillance videos here (.mp4, .avi, .mkv, .mov)
│
├── models/                  # Holds yolov8n.pt pretrained model weights
│
├── baseline/                # Baseline execution modules
│   ├── run_baseline.py      # Master orchestrator script
│   ├── benchmark.py         # Configuration & latency/utilization benchmarks
│   ├── evaluate_video.py    # Sequential video inference & frame annotation
│   └── generate_report.py   # Metrics summary CSV & Markdown report generator
│
├── results/                 # Auto-generated outputs
│   ├── annotated/           # Visual annotated output videos (*_yolov8n_baseline.mp4)
│   ├── metrics/             # Per-video metrics CSV and baseline_summary.csv
│   ├── logs/                # Per-video execution logs (*.log)
│   └── reports/             # Formal baseline_report.md
│
├── dataset/                 # (Reserved for future CCTV dataset & ground-truth annotations)
│
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
└── setup_check.py           # Environment diagnostics script
```

---

## ⚙️ Environment Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Environment
Run the environment check to inspect PyTorch, CUDA, GPU acceleration, and Ultralytics versions:
```bash
python setup_check.py
```

---

## 🚀 Running the Baseline

### 1. Add Test Videos
Place your CCTV / border surveillance video clips into the `videos/` folder:
- Supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`

### 2. Execute Baseline Inference
Run the complete baseline pipeline with a single command:
```bash
python baseline/run_baseline.py
```

The pipeline automatically:
1. Verifies the hardware environment (CUDA GPU / CPU fallback).
2. Downloads and loads pretrained `yolov8n.pt` into `models/`.
3. Processes all videos sequentially with a configurable confidence threshold (`conf = 0.25`).
4. Generates visual annotated bounding box videos under `results/annotated/`.
5. Computes confidence statistics and class distributions per video in `results/metrics/`.
6. Consolidates performance in `results/metrics/baseline_summary.csv`.
7. Generates a comprehensive evaluation report in `results/reports/baseline_report.md`.

---

## 📊 Important Notice on Metrics

> **NOTE:** Accuracy metrics (such as mAP, Precision, Recall, F1) are **intentionally not reported** in this baseline stage because ground-truth annotations have not yet been created for the input video footage. 
>
> The baseline measures **actual runtime performance** (Inference FPS, Frame Latency in ms, Class Distributions, and Hardware Utilization).

---

## ⏭️ Next Steps

1. Collect and curate a representative CCTV border surveillance dataset under `dataset/`.
2. Generate ground-truth annotations (bounding boxes and class labels).
3. Fine-tune YOLOv8n on the CCTV dataset and benchmark against this established baseline.
