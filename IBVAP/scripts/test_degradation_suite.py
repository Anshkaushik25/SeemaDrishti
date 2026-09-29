"""
IBVAP Stage 2 — Comprehensive Degradation Test Suite & Benchmark
===================================================================
Evaluates the visibility enhancement pipeline across 8 synthetic/real CCTV degradation scenarios:
A. Normal / Good Visibility
B. True Low-Light (Simulated night surveillance)
C. High Brightness + Low Sharpness (Observed problematic screenshot case: Brightness=220.3, Contrast=31.7, Sharpness=7.9)
D. Heavy Haze / Fog
E. High Sensor Noise
F. Motion Blur / Out of Focus
G. Low Light + Noise
H. Low Light + Haze

Saves outputs to:
IBVAP/outputs/original/
IBVAP/outputs/enhanced/
IBVAP/outputs/auto/
IBVAP/outputs/comparison/
IBVAP/outputs/metrics/

Generates benchmark JSON metrics for report inclusion.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
from pathlib import Path

# Setup paths for module import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from visibility_utils import assess_visibility, compute_highlight_clipping
from enhancement import enhance_conditionally, apply_gamma, apply_clahe
from detector import PersonDetector
from diagnostics import DiagnosticsManager

MODEL_PATH = PROJECT_DIR / r"training\runs\detect\mot17_05_baseline\weights\best.pt"
VAL_IMAGES_DIR = PROJECT_DIR / r"datasets\MOT17_05_YOLO\images\val"


def generate_degradation_frames(base_frame):
    """
    Generate 8 test category frames from a base validation frame:
    A. Good / Normal
    B. Low Light (darkened to mean ~45)
    C. High Brightness + Low Sharpness (Observed user case: mean brightness ~220.3, contrast ~31.7, sharpness ~7.9)
    D. Haze / Fog (added dark channel haze simulation)
    E. Noise (added Gaussian noise)
    F. Blur (Gaussian blur kernel)
    G. Low Light + Noise
    H. Low Light + Haze
    """
    h, w = base_frame.shape[:2]
    categories = {}

    # Category A: Good / Normal
    categories["A_good_normal"] = base_frame.copy()

    # Category B: True Low-Light
    dark = cv2.convertScaleAbs(base_frame, alpha=0.25, beta=0)
    categories["B_low_light"] = dark

    # Category C: High Brightness + Low Sharpness (User screenshot case)
    # Target: Brightness ~220.3, Contrast ~31.7, Sharpness ~7.9
    bright_blur = cv2.GaussianBlur(base_frame, (11, 11), 3.0)
    bright_blur = cv2.convertScaleAbs(bright_blur, alpha=0.45, beta=150)
    categories["C_high_bright_low_sharpness"] = bright_blur

    # Category D: Haze / Fog
    haze_bg = np.full_like(base_frame, 210, dtype=np.uint8)
    hazy = cv2.addWeighted(base_frame, 0.45, haze_bg, 0.55, 0)
    categories["D_haze_fog"] = hazy

    # Category E: High Sensor Noise
    noise = np.random.normal(0, 25, base_frame.shape).astype(np.float32)
    noisy = np.clip(base_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    categories["E_high_noise"] = noisy

    # Category F: Motion / Out-of-Focus Blur
    blurry = cv2.GaussianBlur(base_frame, (15, 15), 5.0)
    categories["F_motion_blur"] = blurry

    # Category G: Low Light + Noise
    dark_noisy = np.clip(dark.astype(np.float32) + np.random.normal(0, 20, dark.shape), 0, 255).astype(np.uint8)
    categories["G_low_light_noise"] = dark_noisy

    # Category H: Low Light + Haze
    dark_hazy = cv2.addWeighted(dark, 0.5, np.full_like(dark, 140, dtype=np.uint8), 0.5, 0)
    categories["H_low_light_haze"] = dark_hazy

    return categories


def run_degradation_benchmark():
    print("=" * 70)
    print("  IBVAP STAGE 2 — DEGRADATION BENCHMARK TEST SUITE")
    print("=" * 70)

    # 1. Initialize Detector & Diagnostics
    device_arg = 0 if torch.cuda.is_available() else "cpu"
    detector = PersonDetector(model_path=MODEL_PATH, conf=0.15, iou=0.45, max_det=100, imgsz=640, device=device_arg)
    diagnostics = DiagnosticsManager()

    # Find sample validation image
    val_images = list(VAL_IMAGES_DIR.glob("*.jpg"))
    if not val_images:
        print(f"  [ERROR] No validation images found in {VAL_IMAGES_DIR}")
        return

    sample_img_path = val_images[10]  # Select validation frame
    base_frame = cv2.imread(str(sample_img_path))
    print(f"  Loaded Base Frame: {sample_img_path.name} ({base_frame.shape[1]}x{base_frame.shape[0]})")
    print(f"  Inference Device : {'GPU (CUDA:0)' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Generate 8 Test Categories
    test_cases = generate_degradation_frames(base_frame)
    benchmark_results = []

    for cat_name, raw_frame in test_cases.items():
        print(f"\n▶ Testing Category: [{cat_name}]")

        # Step 1: Visibility Analysis
        vis_meta = assess_visibility(raw_frame)
        print(f"  Visibility State  : {vis_meta['state']}")
        print(f"  Primary Degradation: {vis_meta['primary_degradation']}")
        print(f"  Action Summary    : {vis_meta['action_summary']}")
        print(f"  Metrics           : Brightness={vis_meta['brightness']}, Contrast={vis_meta['contrast']}, Sharpness={vis_meta['sharpness']}, Clip={vis_meta['highlight_clipping']}%")

        # Step 2: Generate Always-Enhanced Frame (Unconstrained Gamma + CLAHE simulation for baseline comparison)
        t0_enh = time.perf_counter()
        always_enhanced = apply_gamma(raw_frame, gamma=1.6)
        always_enhanced = apply_clahe(always_enhanced, clip_limit=3.0)
        enh_latency_ms = (time.perf_counter() - t0_enh) * 1000.0

        # Step 3: Generate AUTO Mode Frame (Degradation-Aware Adaptive Enhancement)
        t0_auto = time.perf_counter()
        auto_frame = enhance_conditionally(raw_frame, vis_meta)
        auto_latency_ms = (time.perf_counter() - t0_auto) * 1000.0

        # Step 4: YOLO Inference on RAW, ALWAYS_ENHANCED, and AUTO
        t0_inf1 = time.perf_counter()
        raw_dets = detector.detect(raw_frame)
        raw_inf_ms = (time.perf_counter() - t0_inf1) * 1000.0

        t0_inf2 = time.perf_counter()
        enh_dets = detector.detect(always_enhanced)
        enh_inf_ms = (time.perf_counter() - t0_inf2) * 1000.0

        t0_inf3 = time.perf_counter()
        auto_dets = detector.detect(auto_frame)
        auto_inf_ms = (time.perf_counter() - t0_inf3) * 1000.0

        # Extract telemetry
        raw_conf = max([d["conf"] for d in raw_dets], default=0.0)
        enh_conf = max([d["conf"] for d in enh_dets], default=0.0)
        auto_conf = max([d["conf"] for d in auto_dets], default=0.0)

        raw_clip = compute_highlight_clipping(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY))
        enh_clip = compute_highlight_clipping(cv2.cvtColor(always_enhanced, cv2.COLOR_BGR2GRAY))
        auto_clip = compute_highlight_clipping(cv2.cvtColor(auto_frame, cv2.COLOR_BGR2GRAY))

        print(f"  [RAW]   Detections: {len(raw_dets)} | Best Conf: {raw_conf:.2f} | Clip: {raw_clip:.1f}%")
        print(f"  [ENH]   Detections: {len(enh_dets)} | Best Conf: {enh_conf:.2f} | Clip: {enh_clip:.1f}%")
        print(f"  [AUTO]  Detections: {len(auto_dets)} | Best Conf: {auto_conf:.2f} | Clip: {auto_clip:.1f}%")

        # Step 5: Render Annotated Frames
        det_raw_img = detector.draw_detections(raw_frame, raw_dets)
        det_enh_img = detector.draw_detections(always_enhanced, enh_dets)
        det_auto_img = detector.draw_detections(auto_frame, auto_dets)

        # Step 6: Save 4-Quadrant Diagnostic Grid
        raw_info = {"count": len(raw_dets), "max_conf": raw_conf}
        enh_info = {"count": len(enh_dets), "max_conf": enh_conf}
        auto_info = {"count": len(auto_dets), "max_conf": auto_conf}
        lat_info = {"enhancement_ms": auto_latency_ms, "yolo_ms": auto_inf_ms, "total_ms": auto_latency_ms + auto_inf_ms}

        frame_filename = f"{cat_name}.jpg"
        diagnostics.generate_quadrant_comparison(
            raw_frame, always_enhanced, auto_frame,
            det_raw_img, det_enh_img, det_auto_img,
            vis_meta, raw_info, enh_info, auto_info, lat_info,
            frame_name=frame_filename
        )

        cat_record = {
            "category": cat_name,
            "visibility_state": vis_meta["state"],
            "primary_degradation": vis_meta["primary_degradation"],
            "action_summary": vis_meta["action_summary"],
            "brightness": vis_meta["brightness"],
            "contrast": vis_meta["contrast"],
            "sharpness": vis_meta["sharpness"],
            "haze_level": vis_meta["haze_level"],
            "noise_level": vis_meta["noise_level"],
            "raw_det_count": len(raw_dets),
            "raw_max_conf": round(raw_conf, 3),
            "raw_clip_pct": raw_clip,
            "enh_det_count": len(enh_dets),
            "enh_max_conf": round(enh_conf, 3),
            "enh_clip_pct": enh_clip,
            "auto_det_count": len(auto_dets),
            "auto_max_conf": round(auto_conf, 3),
            "auto_clip_pct": auto_clip,
            "enhancement_latency_ms": round(auto_latency_ms, 2),
            "yolo_latency_ms": round(auto_inf_ms, 2),
            "total_latency_ms": round(auto_latency_ms + auto_inf_ms, 2),
        }
        benchmark_results.append(cat_record)

    # Save benchmark JSON
    json_path = diagnostics.dir_metrics / "degradation_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print()
    print("=" * 70)
    print("  BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'CATEGORY':<30} | {'PRIMARY':<16} | {'ACTION':<24} | {'RAW CONF':<8} | {'AUTO CONF':<9}")
    print("-" * 95)
    for r in benchmark_results:
        print(f"{r['category']:<30} | {r['primary_degradation']:<16} | {r['action_summary']:<24} | {r['raw_max_conf']:<8} | {r['auto_max_conf']:<9}")
    print("=" * 95)
    print(f"\n  Diagnostic Grid Outputs saved to: {diagnostics.dir_comparison}")
    print(f"  Benchmark JSON saved to         : {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_degradation_benchmark()
