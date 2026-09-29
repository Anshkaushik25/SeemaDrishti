"""
IBVAP Stage 2 — Detector V2 Comparative Benchmark Evaluation Script
======================================================================
Evaluates and compares:
- OLD: IBVAP/models/best_multiclass.pt (Multiclass Baseline)
- NEW: IBVAP/models/best_detector_v2.pt (Detector V2 Trained Model)

Evaluates on the held-out Detector V2 CCTV test split across 5 target classes:
0: person, 1: car, 2: truck, 3: bike, 4: patrol_vehicle

Exports:
- IBVAP/outputs/metrics/detector_v2_comparison.csv
- IBVAP/outputs/metrics/detector_v2_comparison.json
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

MODEL_OLD_PATH = PROJECT_DIR / "models" / "best_multiclass.pt"
MODEL_NEW_PATH = PROJECT_DIR / "models" / "best_detector_v2.pt"

TEST_IMG_DIR = PROJECT_DIR / "datasets" / "ibvap_detector_v2" / "images" / "test"
TEST_LBL_DIR = PROJECT_DIR / "datasets" / "ibvap_detector_v2" / "labels" / "test"

OUTPUTS_METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {0: "person", 1: "car", 2: "truck", 3: "bike", 4: "patrol_vehicle"}


def evaluate_model(model_name, model_path, conf=0.15, iou=0.45):
    """Run evaluation for a model path on Detector V2 test dataset."""
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

    class_stats = {c_name: {"gt": 0, "tp": 0, "fp": 0, "fn": 0} for c_name in CLASSES.values()}
    size_stats = {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}}
    latency_records = []

    for img_p in test_imgs:
        raw_frame = cv2.imread(str(img_p))
        if raw_frame is None:
            continue

        h_img, w_img = raw_frame.shape[:2]
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"

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

        t0 = time.perf_counter()
        detections = detector.detect(raw_frame)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latency_records.append(lat_ms)

        total_pred_boxes += len(detections)
        gt_matched = [False] * len(gt_boxes)

        for det in detections:
            p_box = det["bbox"]
            p_cls = det.get("cls_id", 0)

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
                if c_name in class_stats:
                    class_stats[c_name]["tp"] += 1

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
                if c_name in class_stats:
                    class_stats[c_name]["fp"] += 1

        for idx, g_item in enumerate(gt_boxes):
            g_cls_name = CLASSES.get(g_item["cls_id"], "person")
            if g_cls_name in class_stats:
                class_stats[g_cls_name]["gt"] += 1

            if not gt_matched[idx]:
                total_fn += 1
                if g_cls_name in class_stats:
                    class_stats[g_cls_name]["fn"] += 1

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

    per_class_recall = {}
    for c_name, c_data in class_stats.items():
        per_class_recall[c_name] = round(c_data["tp"] / float(c_data["gt"]), 3) if c_data["gt"] > 0 else 0.0

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
        "per_class_recall": per_class_recall,
        "avg_latency_ms": avg_lat_ms,
        "fps": fps
    }


def main():
    print("=" * 75)
    print("  IBVAP STAGE 2 — DETECTOR V2 BENCHMARK EVALUATION")
    print("=" * 75)

    res_old = evaluate_model("Model Baseline (best_multiclass.pt)", MODEL_OLD_PATH)
    res_new = evaluate_model("Model V2 (best_detector_v2.pt)", MODEL_NEW_PATH)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_model": res_old,
        "new_v2_model": res_new
    }

    json_path = OUTPUTS_METRICS_DIR / "detector_v2_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = OUTPUTS_METRICS_DIR / "detector_v2_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "OLD Model (best_multiclass)", "NEW Model V2 (best_detector_v2)"])
        if res_old and res_new:
            writer.writerow(["Precision", res_old["precision"], res_new["precision"]])
            writer.writerow(["Recall", res_old["recall"], res_new["recall"]])
            writer.writerow(["F1 Score", res_old["f1"], res_new["f1"]])
            writer.writerow(["FP / Frame", res_old["fp_per_frame"], res_new["fp_per_frame"]])
            writer.writerow(["GPU Latency (ms)", res_old["avg_latency_ms"], res_new["avg_latency_ms"]])
            writer.writerow(["Inference FPS", res_old["fps"], res_new["fps"]])
            for c_name in CLASSES.values():
                writer.writerow([f"Recall ({c_name})", res_old["per_class_recall"].get(c_name, 0), res_new["per_class_recall"].get(c_name, 0)])

    print("\n" + "=" * 75)
    print("  DETECTOR V2 COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'METRIC':<25} | {'OLD (MULTICLASS BASELINE)':<25} | {'NEW (DETECTOR V2)':<25}")
    print("-" * 80)
    if res_old and res_new:
        print(f"{'Overall Precision':<25} | {res_old['precision']:<25} | {res_new['precision']:<25}")
        print(f"{'Overall Recall':<25} | {res_old['recall']:<25} | {res_new['recall']:<25}")
        print(f"{'Overall F1 Score':<25} | {res_old['f1']:<25} | {res_new['f1']:<25}")
        print(f"{'FP / Frame':<25} | {res_old['fp_per_frame']:<25} | {res_new['fp_per_frame']:<25}")
        print(f"{'GPU Latency (ms)':<25} | {res_old['avg_latency_ms']:<25} | {res_new['avg_latency_ms']:<25}")
        print(f"{'Inference FPS':<25} | {res_old['fps']:<25} | {res_new['fps']:<25}")
        print("-" * 80)
        for c_name in CLASSES.values():
            print(f"{f'Recall ({c_name})':<25} | {res_old['per_class_recall'].get(c_name, 0):<25} | {res_new['per_class_recall'].get(c_name, 0):<25}")
    print("=" * 75)
    print(f"  Benchmark JSON saved to: {json_path}")
    print(f"  Benchmark CSV saved to : {csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
