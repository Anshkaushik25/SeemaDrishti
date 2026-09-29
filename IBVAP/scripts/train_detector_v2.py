"""
IBVAP Stage 2 — Train Detector V2 Model (CUDA GPU Accelerated)
=================================================================
Trains the multi-class YOLOv8n Detector V2 model on IBVAP/datasets/ibvap_detector_v2/data.yaml:
Target Classes:
0: person
1: car
2: truck
3: bike
4: patrol_vehicle

Configuration:
- imgsz=640
- device=0 (NVIDIA GeForce RTX 3050 Laptop GPU, Automatic Mixed Precision amp=True)
- batch=16
- epochs=30 (Patience=8 early stopping)
- seed=42
- Exports best model to IBVAP/models/best_detector_v2.pt
"""

import os
import sys
import json
import time
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_YAML = PROJECT_DIR / "datasets" / "ibvap_detector_v2" / "data.yaml"
MODELS_DIR = PROJECT_DIR / "models"
EXPERIMENTS_DIR = PROJECT_DIR / "outputs" / "experiments"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("  IBVAP STAGE 2 — TRAIN DETECTOR V2 MODEL (YOLOV8N)")
    print("=" * 75)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f" [INFO] Device: {device} ({device_name}) | AMP: True")
    print(f" [INFO] Dataset Config: {DATA_YAML}")

    if not DATA_YAML.exists():
        print(f" [ERROR] data.yaml not found at: {DATA_YAML}")
        sys.exit(1)

    base_model_path = PROJECT_DIR / "yolov8n.pt"
    if not base_model_path.exists():
        base_model_path = "yolov8n.pt"

    print(f" [INFO] Loading pretrained base model: {base_model_path}")
    model = YOLO(str(base_model_path))

    epochs = 30
    imgsz = 640
    batch = 16
    seed = 42

    print(f" [INFO] Hyperparameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, seed={seed}")
    t0 = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=8,
        seed=seed,
        amp=True,
        workers=2,
        project=str(PROJECT_DIR / "training" / "runs"),
        name="detector_v2_yolov8n",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,            # Never flip upside down for CCTV
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        verbose=True
    )

    elapsed_sec = time.time() - t0
    print(f"\n  [SUCCESS] Training finished in {elapsed_sec / 60.0:.2f} minutes.")

    # Save to best_detector_v2.pt
    runs_best = PROJECT_DIR / "training" / "runs" / "detector_v2_yolov8n" / "weights" / "best.pt"
    target_v2_best = MODELS_DIR / "best_detector_v2.pt"

    if runs_best.exists():
        shutil.copy(str(runs_best), str(target_v2_best))
        print(f" [INFO] Detector V2 Best Model saved to: {target_v2_best}")
    else:
        print(f" [WARN] Best weights file not found at: {runs_best}")

    # Save Experiment Record
    exp_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "IBVAP Detector V2",
        "architecture": "YOLOv8n",
        "pretrained_base": str(base_model_path),
        "target_classes": {0: "person", 1: "car", 2: "truck", 3: "bike", 4: "patrol_vehicle"},
        "imgsz": imgsz,
        "batch_size": batch,
        "epochs": epochs,
        "device": device_name,
        "training_time_sec": round(elapsed_sec, 2),
        "best_weights_path": str(target_v2_best),
        "dataset_config": str(DATA_YAML)
    }

    exp_json_path = EXPERIMENTS_DIR / "exp_detector_v2.json"
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)

    print(f" [INFO] Experiment record saved to: {exp_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
