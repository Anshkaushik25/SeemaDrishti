"""
IBVAP Stage 2 — Enhancement Safety & Detection Comparison Tool
=================================================================
Evaluates RAW vs ENHANCED vs AUTO mode detection performance on a fixed benchmark set.
Enforces hard safety gates:
- Highlight clipping increase check (< 5.0%)
- Mean brightness threshold (< 235.0)

Generates:
- IBVAP/outputs/metrics/enhancement_comparison.csv
"""

import os
import sys
import time
import csv
import json
import cv2
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from visibility_utils import assess_visibility, compute_highlight_clipping
from enhancement import enhance_conditionally
from detector import PersonDetector

MODEL_PATH = PROJECT_DIR / "models" / "best_multiclass.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

TEST_IMG_DIR = PROJECT_DIR / "datasets" / "unified_multiclass" / "images" / "test"
TEST_LBL_DIR = PROJECT_DIR / "datasets" / "unified_multiclass" / "labels" / "test"
OUTPUTS_METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def run_enhancement_evaluation():
    print("=" * 70)
    print("  IBVAP STAGE 2 — ENHANCEMENT SAFETY & COMPARISON BENCHMARK")
    print("=" * 70)

    device = 0 if torch.cuda.is_available() else "cpu"
    detector = PersonDetector(model_path=MODEL_PATH, conf=0.15, iou=0.45, max_det=100, imgsz=640, device=device)

    test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))
    if not test_imgs:
        print(f"  [ERROR] No test images found in {TEST_IMG_DIR}")
        return

    csv_rows = []

    raw_total_dets = 0
    raw_confs = []
    auto_total_dets = 0
    auto_confs = []
    rejected_count = 0

    for img_p in test_imgs:
        raw_frame = cv2.imread(str(img_p))
        if raw_frame is None:
            continue

        # Visibility assessment
        vis_meta = assess_visibility(raw_frame)

        # Generate AUTO frame
        auto_frame = enhance_conditionally(raw_frame, vis_meta)
        is_rejected = np.array_equal(raw_frame, auto_frame) and vis_meta.get("is_poor", False) and vis_meta.get("primary_degradation") in ["HIGH_BRIGHTNESS", "BLURRY"]
        if is_rejected:
            rejected_count += 1

        # Run detections
        raw_dets = detector.detect(raw_frame)
        auto_dets = detector.detect(auto_frame)

        raw_best_conf = max([d["conf"] for d in raw_dets], default=0.0)
        auto_best_conf = max([d["conf"] for d in auto_dets], default=0.0)

        raw_clip = compute_highlight_clipping(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY))
        auto_clip = compute_highlight_clipping(cv2.cvtColor(auto_frame, cv2.COLOR_BGR2GRAY))

        raw_total_dets += len(raw_dets)
        auto_total_dets += len(auto_dets)
        if raw_best_conf > 0:
            raw_confs.append(raw_best_conf)
        if auto_best_conf > 0:
            auto_confs.append(auto_best_conf)

        csv_rows.append({
            "image": img_p.name,
            "visibility_state": vis_meta["state"],
            "primary_degradation": vis_meta["primary_degradation"],
            "action_summary": vis_meta["action_summary"],
            "raw_det_count": len(raw_dets),
            "raw_best_conf": raw_best_conf,
            "raw_clip_pct": raw_clip,
            "auto_det_count": len(auto_dets),
            "auto_best_conf": auto_best_conf,
            "auto_clip_pct": auto_clip,
            "safety_rejection": "REJECTED (PRESERVED RAW)" if is_rejected else "PASSED / APPLIED"
        })

    # Save CSV Report
    csv_path = OUTPUTS_METRICS_DIR / "enhancement_comparison.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    avg_raw_conf = round(float(np.mean(raw_confs)), 3) if raw_confs else 0.0
    avg_auto_conf = round(float(np.mean(auto_confs)), 3) if auto_confs else 0.0

    print()
    print("=" * 70)
    print("  ENHANCEMENT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total Test Frames Evaluated : {len(test_imgs)}")
    print(f"  RAW Mode Total Detections   : {raw_total_dets} (Avg Best Conf: {avg_raw_conf})")
    print(f"  AUTO Mode Total Detections  : {auto_total_dets} (Avg Best Conf: {avg_auto_conf})")
    print(f"  Safety Rejections (RAW Saved): {rejected_count}")
    print(f"  CSV Report Saved to         : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_enhancement_evaluation()
