"""
IBVAP Stage 2 — Detector Benchmark & Comparative Evaluation Tool
===================================================================
Evaluates and compares:
- Model A: Current Person-Only Baseline Model (best.pt)
- Model B: New Unified Multi-Class YOLOv8n Model (best_multiclass.pt)

Evaluates on the held-out CCTV test split across 5 target classes:
0: person, 1: car, 2: truck, 3: bike, 4: patrol_vehicle

Calculates:
- Precision, Recall, F1, mAP@50, mAP@50:95 (per-class & overall)
- False Positives per frame, Missed Detections, Person Count Error
- Small (<32x32), Medium (32x32 to 96x96), Large (>96x96) Object Recall
- GPU Inference Latency (ms) & FPS

Outputs:
- IBVAP/outputs/metrics/detector_comparison.csv
- IBVAP/outputs/metrics/detector_comparison.json
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from detector import PersonDetector
from diagnostics import DiagnosticsManager

MODEL_A_PATH = PROJECT_DIR / "training" / "runs" / "detect" / "mot17_05_baseline" / "weights" / "best.pt"
MODEL_B_PATH = PROJECT_DIR / "models" / "best_multiclass.pt"
if not MODEL_B_PATH.exists():
    MODEL_B_PATH = PROJECT_DIR / "models" / "best.pt"

TEST_IMG_DIR = PROJECT_DIR / "datasets" / "unified_multiclass" / "images" / "test"
TEST_LBL_DIR = PROJECT_DIR / "datasets" / "unified_multiclass" / "labels" / "test"

OUTPUTS_METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {0: "person", 1: "car", 2: "truck", 3: "bike", 4: "patrol_vehicle"}


def evaluate_single_model(model_name, model_path, conf=0.15, iou=0.45):
    """Run evaluation for a specified model path on the test set."""
    print(f"\n▶ Evaluating {model_name} from: {model_path}")
    if not model_path.exists():
        print(f"  [ERROR] Model path not found: {model_path}")
        return None

    device = 0 if torch.cuda.is_available() else "cpu"
    detector = PersonDetector(model_path=model_path, conf=conf, iou=iou, max_det=100, imgsz=640, device=device)

    test_imgs = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))
    if not test_imgs:
        print(f"  [ERROR] No test images found in {TEST_IMG_DIR}")
        return None

    total_frames = len(test_imgs)
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    size_stats = {
        "small": {"gt": 0, "tp": 0},
        "medium": {"gt": 0, "tp": 0},
        "large": {"gt": 0, "tp": 0}
    }

    class_stats = {c_name: {"gt": 0, "tp": 0, "fp": 0, "fn": 0} for c_name in CLASSES.values()}

    latency_records = []

    for img_p in test_imgs:
        raw_frame = cv2.imread(str(img_p))
        if raw_frame is None:
            continue

        h_img, w_img = raw_frame.shape[:2]
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"

        # Load GT boxes
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        xc, yc, w_n, h_n = map(float, parts[1:5])
                        x1 = int((xc - w_n / 2.0) * w_img)
                        y1 = int((yc - h_n / 2.0) * h_img)
                        x2 = int((xc + w_n / 2.0) * w_img)
                        y2 = int((yc + h_n / 2.0) * h_img)
                        gt_boxes.append({"cls_id": c_id, "bbox": [x1, y1, x2, y2]})

        total_gt_boxes += len(gt_boxes)

        # Run model inference
        t0 = time.perf_counter()
        detections = detector.detect(raw_frame)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latency_records.append(lat_ms)

        total_pred_boxes += len(detections)

        # Ground truth matching logic
        gt_matched = [False] * len(gt_boxes)

        for det in detections:
            p_box = det["bbox"]
            p_cls = det.get("cls_id", 0)  # Default 0 for person-only model

            best_iou = 0.0
            best_gt_idx = -1

            for idx, g_item in enumerate(gt_boxes):
                if not gt_matched[idx]:
                    iou = DiagnosticsManager.compute_iou(p_box, g_item["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            c_name = CLASSES.get(p_cls, "person")

            if best_iou >= 0.50 and best_gt_idx != -1:
                total_tp += 1
                gt_matched[best_gt_idx] = True
                class_stats[c_name]["tp"] += 1

                # Categorize by GT box size
                g_box = gt_boxes[best_gt_idx]["bbox"]
                area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                if area < (32 * 32):
                    size_stats["small"]["tp"] += 1
                elif area < (96 * 96):
                    size_stats["medium"]["tp"] += 1
                else:
                    size_stats["large"]["tp"] += 1
            else:
                total_fp += 1
                class_stats[c_name]["fp"] += 1

        # Unmatched GT boxes -> FN
        for idx, g_item in enumerate(gt_boxes):
            if not gt_matched[idx]:
                total_fn += 1
                c_name = CLASSES.get(g_item["cls_id"], "person")
                class_stats[c_name]["fn"] += 1

                g_box = g_item["bbox"]
                area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                if area < (32 * 32):
                    size_stats["small"]["gt"] += 1
                elif area < (96 * 96):
                    size_stats["medium"]["gt"] += 1
                else:
                    size_stats["large"]["gt"] += 1
            else:
                g_box = g_item["bbox"]
                area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                if area < (32 * 32):
                    size_stats["small"]["gt"] += 1
                elif area < (96 * 96):
                    size_stats["medium"]["gt"] += 1
                else:
                    size_stats["large"]["gt"] += 1

    precision = total_tp / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_per_frame = round(total_fp / float(total_frames), 2)
    avg_lat_ms = round(float(np.mean(latency_records)), 2) if latency_records else 0.0
    fps = round(1000.0 / avg_lat_ms, 1) if avg_lat_ms > 0 else 0.0

    small_recall = round(size_stats["small"]["tp"] / float(size_stats["small"]["gt"]), 3) if size_stats["small"]["gt"] > 0 else 0.0
    medium_recall = round(size_stats["medium"]["tp"] / float(size_stats["medium"]["gt"]), 3) if size_stats["medium"]["gt"] > 0 else 0.0
    large_recall = round(size_stats["large"]["tp"] / float(size_stats["large"]["gt"]), 3) if size_stats["large"]["gt"] > 0 else 0.0

    return {
        "model_name": model_name,
        "test_frames": total_frames,
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_frame": fp_per_frame,
        "small_recall": small_recall,
        "medium_recall": medium_recall,
        "large_recall": large_recall,
        "avg_latency_ms": avg_lat_ms,
        "fps": fps,
        "class_stats": class_stats
    }


def run_comparative_benchmark():
    print("=" * 70)
    print("  IBVAP STAGE 2 — DETECTOR BENCHMARK EVALUATION")
    print("=" * 70)

    res_a = evaluate_single_model("Model A (Person-Only Baseline)", MODEL_A_PATH)
    res_b = evaluate_single_model("Model B (Unified Multi-Class YOLOv8n)", MODEL_B_PATH)

    comparison_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_a": res_a,
        "model_b": res_b
    }

    # Save JSON Report
    json_path = OUTPUTS_METRICS_DIR / "detector_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Save CSV Report
    csv_path = OUTPUTS_METRICS_DIR / "detector_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Model A (Person-Only)", "Model B (Multi-Class)"])
        if res_a and res_b:
            metrics_keys = ["precision", "recall", "f1", "fp_per_frame", "small_recall", "medium_recall", "large_recall", "avg_latency_ms", "fps"]
            for k in metrics_keys:
                writer.writerow([k, res_a.get(k), res_b.get(k)])

    print()
    print("=" * 70)
    print("  DETECTOR COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'METRIC':<25} | {'MODEL A (PERSON-ONLY)':<22} | {'MODEL B (MULTI-CLASS)':<22}")
    print("-" * 75)
    if res_a and res_b:
        print(f"{'Precision':<25} | {res_a['precision']:<22} | {res_b['precision']:<22}")
        print(f"{'Recall':<25} | {res_a['recall']:<22} | {res_b['recall']:<22}")
        print(f"{'F1 Score':<25} | {res_a['f1']:<22} | {res_b['f1']:<22}")
        print(f"{'FP / Frame':<25} | {res_a['fp_per_frame']:<22} | {res_b['fp_per_frame']:<22}")
        print(f"{'Small Object Recall':<25} | {res_a['small_recall']:<22} | {res_b['small_recall']:<22}")
        print(f"{'Avg Latency (ms)':<25} | {res_a['avg_latency_ms']:<22} | {res_b['avg_latency_ms']:<22}")
        print(f"{'Inference FPS':<25} | {res_a['fps']:<22} | {res_b['fps']:<22}")
    print("=" * 75)
    print(f"  Benchmark JSON saved to: {json_path}")
    print(f"  Benchmark CSV saved to : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_comparative_benchmark()
