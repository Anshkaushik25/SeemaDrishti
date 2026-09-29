"""
IBVAP Stage 2 — Diagnostics & Ground-Truth Matching Module
===========================================================
Generates 4-quadrant side-by-side comparison outputs and records detection performance metrics.
Calculates TP, FP, FN, Precision, Recall, F1, and mAP@50 metrics when GT annotations are available.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


class DiagnosticsManager:
    """Manages diagnostic output directories and visual comparisons."""

    def __init__(self, outputs_dir=OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.dir_original = self.outputs_dir / "original"
        self.dir_enhanced = self.outputs_dir / "enhanced"
        self.dir_auto = self.outputs_dir / "auto"
        self.dir_comparison = self.outputs_dir / "comparison"
        self.dir_metrics = self.outputs_dir / "metrics"

        self.create_directories()

    def create_directories(self):
        for d in [self.dir_original, self.dir_enhanced, self.dir_auto, self.dir_comparison, self.dir_metrics]:
            d.mkdir(parents=True, exist_ok=True)

    def render_metrics_panel(self, width, height, visibility_data, raw_det_info, enh_det_info, auto_det_info, latency_info):
        """Render a dark text canvas containing visibility state, primary degradation, actions, det counts, best conf, and latency."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (24, 28, 36)  # Dark slate background

        # Section Header
        cv2.putText(panel, "DIAGNOSTIC METRICS & EVALUATION", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.line(panel, (15, 40), (width - 15, 40), (80, 80, 90), 1)

        y = 65
        state = visibility_data.get("state", "UNKNOWN")
        primary = visibility_data.get("primary_degradation", "UNKNOWN")
        action = visibility_data.get("action_summary", "NONE")
        clip = visibility_data.get("highlight_clipping", 0.0)

        # Visibility metrics
        state_color = (0, 255, 120) if state == "GOOD" else (50, 100, 255)
        cv2.putText(panel, f"VISIBILITY STATE : {state}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        y += 25
        cv2.putText(panel, f"PRIMARY DEGRADATION: {primary}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 255), 1)
        y += 25
        cv2.putText(panel, f"ACTION TAKEN        : {action}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        y += 25
        cv2.putText(panel, f"HIGHLIGHT CLIPPING  : {clip:.1f}%", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        y += 30

        cv2.line(panel, (15, y - 10), (width - 15, y - 10), (80, 80, 90), 1)

        # Detections & Confidence
        raw_cnt = raw_det_info.get("count", 0)
        raw_conf = raw_det_info.get("max_conf", 0.0)
        enh_cnt = enh_det_info.get("count", 0)
        enh_conf = enh_det_info.get("max_conf", 0.0)
        auto_cnt = auto_det_info.get("count", 0)
        auto_conf = auto_det_info.get("max_conf", 0.0)

        cv2.putText(panel, f"RAW   DETECTIONS: {raw_cnt} | BEST CONF: {raw_conf:.2f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
        y += 22
        cv2.putText(panel, f"ENH   DETECTIONS: {enh_cnt} | BEST CONF: {enh_conf:.2f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 1)
        y += 22
        cv2.putText(panel, f"AUTO  DETECTIONS: {auto_cnt} | BEST CONF: {auto_conf:.2f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2)
        y += 30

        cv2.line(panel, (15, y - 10), (width - 15, y - 10), (80, 80, 90), 1)

        # Latency
        enh_lat = latency_info.get("enhancement_ms", 0.0)
        yolo_lat = latency_info.get("yolo_ms", 0.0)
        tot_lat = latency_info.get("total_ms", 0.0)

        cv2.putText(panel, f"ENHANCEMENT LATENCY: {enh_lat:.1f} ms", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        y += 22
        cv2.putText(panel, f"YOLO INFERENCE     : {yolo_lat:.1f} ms", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        y += 22
        cv2.putText(panel, f"TOTAL FRAME LATENCY: {tot_lat:.1f} ms", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 120), 2)

        return panel

    def generate_quadrant_comparison(
        self,
        raw_frame,
        enhanced_frame,
        auto_frame,
        det_raw_frame,
        det_enh_frame,
        det_auto_frame,
        visibility_data,
        raw_det_info,
        enh_det_info,
        auto_det_info,
        latency_info,
        frame_name="frame.jpg"
    ):
        """
        Create a 4-quadrant (2x2 grid) diagnostic comparison image:
        ┌─────────────────┬─────────────────┐
        │ ORIGINAL (RAW)  │ ENHANCED        │
        ├─────────────────┼─────────────────┤
        │ AUTO            │ METRICS         │
        └─────────────────┴─────────────────┘
        """
        h, w = det_raw_frame.shape[:2]

        quad_a = det_raw_frame.copy()
        cv2.putText(quad_a, "1. ORIGINAL (RAW)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        quad_b = det_enh_frame.copy()
        cv2.putText(quad_b, "2. ENHANCED (ALWAYS ENHANCE)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

        quad_c = det_auto_frame.copy()
        cv2.putText(quad_c, "3. AUTO (DEGRADATION-AWARE)", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

        quad_d = self.render_metrics_panel(w, h, visibility_data, raw_det_info, enh_det_info, auto_det_info, latency_info)

        top_row = np.hstack([quad_a, quad_b])
        bottom_row = np.hstack([quad_c, quad_d])
        comparison_grid = np.vstack([top_row, bottom_row])

        # Save individual frames and comparison grid
        cv2.imwrite(str(self.dir_original / frame_name), raw_frame)
        cv2.imwrite(str(self.dir_enhanced / frame_name), enhanced_frame)
        cv2.imwrite(str(self.dir_auto / frame_name), auto_frame)
        cv2.imwrite(str(self.dir_comparison / f"comp_{frame_name}"), comparison_grid)

        return comparison_grid

    @staticmethod
    def compute_iou(boxA, boxB):
        """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def evaluate_ground_truth(self, gt_boxes, pred_boxes, iou_thresh=0.5):
        """
        Evaluate predictions against Ground Truth boxes.
        Returns dict with TP, FP, FN, Precision, Recall, F1.
        """
        tp = 0
        fp = 0
        gt_matched = [False] * len(gt_boxes)

        for p_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                if not gt_matched[idx]:
                    iou = self.compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx != -1:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = len(gt_boxes) - tp
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

