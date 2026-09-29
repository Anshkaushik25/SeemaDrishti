"""
IBVAP Stage 2 — Multi-Class YOLOv8n Training Script (CUDA GPU Accelerated)
=============================================================================
Trains a new multi-class YOLOv8n object detection model on the unified CCTV dataset for 5 classes:
0: person
1: car
2: truck
3: bike
4: patrol_vehicle

Enforces:
- CUDA GPU execution (device=0, RTX 3050 Laptop GPU, Automatic Mixed Precision amp=True)
- Sequence-level dataset split (data.yaml)
- Realistic CCTV surveillance augmentations (NO 90-degree rotations!)
- Early stopping & reproducible seed=42
- Exports best.pt model to IBVAP/models/best_multiclass.pt and IBVAP/models/best.pt
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
DATA_YAML = PROJECT_DIR / "datasets" / "unified_multiclass" / "data.yaml"
MODELS_DIR = PROJECT_DIR / "models"
EXPERIMENTS_DIR = PROJECT_DIR / "outputs" / "experiments"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("  IBVAP STAGE 2 — MULTI-CLASS YOLOV8N TRAINING")
    print("=" * 70)

    # 1. Environment Audit
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  [INFO] Training Device : {device} ({device_name})")
    print(f"  [INFO] PyTorch Version : {torch.__version__}")
    print(f"  [INFO] Dataset Config  : {DATA_YAML}")

    if not DATA_YAML.exists():
        print(f"  [ERROR] data.yaml not found at: {DATA_YAML}")
        sys.exit(1)

    # 2. Load Base Pretrained Model
    base_model_path = PROJECT_DIR / "yolov8n.pt"
    if not base_model_path.exists():
        base_model_path = "yolov8n.pt"

    print(f"  [INFO] Loading base model: {base_model_path}")
    model = YOLO(str(base_model_path))

    # 3. Training Hyperparameters
    epochs = 15
    imgsz = 640
    batch = 16
    seed = 42

    print(f"  [INFO] Hyperparameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, seed={seed}")
    t0 = time.time()

    # 4. Train Model with Realistic Surveillance Augmentations
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=5,
        seed=seed,
        amp=True,              # Automatic Mixed Precision
        workers=2,
        project=str(PROJECT_DIR / "training" / "runs"),
        name="multiclass_yolov8n",
        exist_ok=True,
        # Realistic CCTV Augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,            # NEVER flip upside down for CCTV
        fliplr=0.5,            # Horizontal flip is realistic
        mosaic=0.5,
        mixup=0.0,
        verbose=True
    )

    elapsed_sec = time.time() - t0
    print(f"\n  [SUCCESS] Training finished in {elapsed_sec / 60.0:.2f} minutes.")

    # 5. Export Best Weights
    runs_best = PROJECT_DIR / "training" / "runs" / "multiclass_yolov8n" / "weights" / "best.pt"
    target_best_multi = MODELS_DIR / "best_multiclass.pt"
    target_best_default = MODELS_DIR / "best.pt"

    if runs_best.exists():
        shutil.copy(str(runs_best), str(target_best_multi))
        shutil.copy(str(runs_best), str(target_best_default))
        print(f"  [INFO] Best weights saved to: {target_best_multi}")
        print(f"  [INFO] Default weights updated: {target_best_default}")
    else:
        print(f"  [WARN] Best weights file not found at: {runs_best}")

    # 6. Save Experiment Record
    exp_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "YOLOv8n",
        "pretrained_weights": str(base_model_path),
        "target_classes": {0: "person", 1: "car", 2: "truck", 3: "bike", 4: "patrol_vehicle"},
        "imgsz": imgsz,
        "batch_size": batch,
        "epochs": epochs,
        "device": device_name,
        "training_time_sec": round(elapsed_sec, 2),
        "best_weights_path": str(target_best_multi),
        "dataset_config": str(DATA_YAML)
    }

    exp_json_path = EXPERIMENTS_DIR / "exp_multiclass_yolov8n.json"
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)

    print(f"  [INFO] Experiment record saved to: {exp_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
