"""
IBVAP Stage 2 — Live Computer-Vision Pipeline
================================================
Unified live vision processing application supporting:
- Video File / Webcam / RTSP CCTV input streams
- Depth-1 Latest-Frame Buffer architecture
- Quality assessment & adaptive low-light/dehazing enhancement
- Multi-person YOLOv8n detection on GPU / CPU (device 0 / cuda:0 / cpu)
- Modes: ORIGINAL / ENHANCED / AUTO
- Diagnostic output generation & telemetry HUD overlay

Usage:
    python IBVAP/scripts/live_vision.py --source 0 --mode AUTO --device 0
    python IBVAP/scripts/live_vision.py --source path/to/video.mp4 --mode ENHANCED --device 0
    python IBVAP/scripts/live_vision.py --source rtsp://192.168.1.100:554/stream1 --mode AUTO --device cpu
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np
import torch
from pathlib import Path

# Setup paths for module import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from source_manager import SourceManager
from frame_buffer import LatestFrameBuffer
from visibility_utils import assess_visibility
from enhancement import enhance_conditionally
from detector import PersonDetector
from diagnostics import DiagnosticsManager


def draw_hud_overlay(frame, mode, src_info, qual, det_count, best_conf, class_counts, metrics, device_name):
    """Draw clean telemetry HUD overlay banner on the output frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent dark overlay box (top-left)
    box_w, box_h = 370, 275
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (10, 10), (10 + box_w, 10 + box_h), (50, 60, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    c_white = (240, 240, 240)
    c_cyan = (255, 200, 0)
    c_green = (0, 230, 120)
    c_orange = (0, 165, 255)

    vis_state = qual.get("state", "GOOD")
    vis_color = c_green if vis_state == "GOOD" else c_orange
    primary = qual.get("primary_degradation", "GOOD")
    action = qual.get("action_summary", "NO ENHANCEMENT (RAW)")
    clip_pct = qual.get("highlight_clipping", 0.0)

    # Class summary text
    class_summary_str = ", ".join([f"{k}:{v}" for k, v in class_counts.items() if v > 0])
    if not class_summary_str:
        class_summary_str = "None"

    # Title
    cv2.putText(annotated, "IBVAP LIVE VISION (MULTI-CLASS)", (20, 32), font, 0.50, c_cyan, 2)
    cv2.line(annotated, (20, 38), (20 + box_w - 20, 38), (60, 70, 90), 1)

    # Metrics Lines
    lines = [
        f"SOURCE: {src_info.get('source_type', 'UNKNOWN')} ({src_info.get('width')}x{src_info.get('height')})",
        f"DEVICE: {device_name} | MODE: {mode.upper()}",
        f"VISIB : {vis_state} | CLIP: {clip_pct:.1f}%",
        f"PRIMARY: {primary}",
        f"ACTION : {action}",
        f"TOTAL DETECTIONS: {det_count} | BEST CONF: {best_conf:.2f}",
        f"CLASSES: {class_summary_str}",
        f"FPS   : Cap {metrics['capture_fps']} | Inf {metrics['inference_fps']}",
        f"LATENCY: {metrics['avg_latency_ms']:.1f} ms | Age: {metrics['avg_frame_age_ms']:.1f} ms",
    ]

    y_offset = 55
    for idx, line in enumerate(lines):
        col = vis_color if idx in [2, 3] else (c_green if idx in [4, 5, 6] else c_white)
        cv2.putText(annotated, line, (20, y_offset), font, 0.38, col, 1)
        y_offset += 23

    return annotated


def main():
    parser = argparse.ArgumentParser(description="IBVAP Stage 2 Live Computer-Vision Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, path to file, or rtsp:// URL)")
    parser.add_argument("--mode", type=str, default="AUTO", choices=["ORIGINAL", "ENHANCED", "AUTO"], help="Runtime mode")
    parser.add_argument("--device", type=str, default="auto", help="Inference device (0, cuda:0, cpu, or auto)")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--save-diagnostics", action="store_true", help="Save 4-quadrant diagnostic comparison images")
    parser.add_argument("--debug-boxes", action="store_true", help="Enable debug mode rendering predicted bounding box details")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after processing N frames (0 for infinite)")

    args = parser.parse_args()

    # Parse numeric webcam index or device if applicable
    source_val = int(args.source) if args.source.isdigit() else args.source
    device_arg = int(args.device) if args.device.isdigit() else args.device

    print()
    print("=" * 60)
    print("  IBVAP STAGE 2 — LIVE COMPUTER-VISION PIPELINE")
    print("=" * 60)
    print(f"  Source       : {args.source}")
    print(f"  Mode         : {args.mode}")
    print(f"  Device Arg   : {args.device}")
    print(f"  Inference    : conf={args.conf}, iou={args.iou}, max_det={args.max_det}, imgsz={args.imgsz}")
    print(f"  Buffer       : Depth-1 Latest Frame Buffer")
    print("=" * 60)
    print()

    # Initialize Source Manager
    src_mgr = SourceManager(source_val)
    if not src_mgr.is_opened():
        print(f"  [ERROR] Could not open input source: {args.source}")
        sys.exit(1)

    src_info = src_mgr.get_info()
    print(f"  Source Opened: {src_info['source_type']} ({src_info['width']}x{src_info['height']})")

    # Initialize Frame Buffer, Detector, and Diagnostics
    buffer = LatestFrameBuffer()
    detector = PersonDetector(conf=args.conf, iou=args.iou, max_det=args.max_det, imgsz=args.imgsz, device=device_arg)
    diagnostics = DiagnosticsManager()

    device_name = f"GPU ({torch.cuda.get_device_name(0)})" if (torch.cuda.is_available() and str(detector.device) != "cpu") else "CPU"

    stop_event = threading.Event()

    # Capture Thread Worker
    def capture_worker():
        while not stop_event.is_set():
            ret, frame = src_mgr.read()
            if not ret or frame is None:
                if src_info["source_type"] == "VIDEO_FILE":
                    print("  End of video stream reached.")
                    stop_event.set()
                    break
                time.sleep(0.02)
                continue

            buffer.push(frame)
            time.sleep(0.005)

    cap_thread = threading.Thread(target=capture_worker, daemon=True)
    cap_thread.start()

    print("  Processing pipeline started. Press 'q' or CTRL+C to exit.")
    print()

    frame_counter = 0

    try:
        while not stop_event.is_set():
            raw_frame, ts, frame_age_ms = buffer.pop()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            t0_inf = time.perf_counter()

            # 1. Quality & Visibility Assessment
            qual = assess_visibility(raw_frame)

            # 2. Determine Detector Frame based on Mode (RAW FRAME SAFETY GUARANTEED)
            if args.mode == "ORIGINAL":
                detector_frame = raw_frame.copy()
            elif args.mode == "ENHANCED":
                detector_frame = enhance_conditionally(raw_frame, qual)
            else:  # AUTO
                detector_frame = enhance_conditionally(raw_frame, qual)

            # 3. Person Detection
            detections = detector.detect(detector_frame)
            inf_latency_ms = (time.perf_counter() - t0_inf) * 1000.0
            buffer.record_latency(inf_latency_ms)

            # Max confidence & class counts calculation
            best_conf = max([d["conf"] for d in detections], default=0.0)
            class_counts = {}
            for d in detections:
                c_name = d.get("class_name", "object")
                class_counts[c_name] = class_counts.get(c_name, 0) + 1

            # 4. Draw Detections
            annotated_frame = detector.draw_detections(detector_frame, detections)

            # 5. Draw Telemetry HUD
            metrics = buffer.get_metrics()
            display_frame = draw_hud_overlay(annotated_frame, args.mode, src_info, qual, len(detections), best_conf, class_counts, metrics, device_name)

            frame_counter += 1

            # Save diagnostics if requested
            if args.save_diagnostics and frame_counter % 30 == 0:
                enhanced_visual = enhance_conditionally(raw_frame, qual)
                auto_visual = detector_frame
                det_raw_list = detector.detect(raw_frame)
                det_enh_list = detector.detect(enhanced_visual)
                det_auto_list = detections

                det_orig = detector.draw_detections(raw_frame, det_raw_list)
                det_enh = detector.draw_detections(enhanced_visual, det_enh_list)
                det_auto = detector.draw_detections(auto_visual, det_auto_list)

                raw_det_info = {"count": len(det_raw_list), "max_conf": max([d["conf"] for d in det_raw_list], default=0.0)}
                enh_det_info = {"count": len(det_enh_list), "max_conf": max([d["conf"] for d in det_enh_list], default=0.0)}
                auto_det_info = {"count": len(det_auto_list), "max_conf": best_conf}
                latency_info = {"enhancement_ms": 0.0, "yolo_ms": inf_latency_ms, "total_ms": inf_latency_ms}

                diagnostics.generate_quadrant_comparison(
                    raw_frame, enhanced_visual, auto_visual,
                    det_orig, det_enh, det_auto,
                    qual, raw_det_info, enh_det_info, auto_det_info, latency_info,
                    frame_name=f"frame_{frame_counter:06d}.jpg"
                )

            # Display GUI window unless headless
            if not args.headless:
                cv2.imshow("IBVAP Live Vision Pipeline", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("  Exit requested by user.")
                    break

            if args.max_frames > 0 and frame_counter >= args.max_frames:
                print(f"  Reached max frames limit ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n  Keyboard interrupt received.")
    finally:
        stop_event.set()
        cap_thread.join(timeout=2.0)
        src_mgr.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("  LIVE VISION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Inference Device       : {device_name}")
    print(f"  Total Processed Frames : {frame_counter}")
    final_metrics = buffer.get_metrics()
    print(f"  Capture Rate           : {final_metrics['capture_fps']} FPS")
    print(f"  Inference Throughput   : {final_metrics['inference_fps']} FPS")
    print(f"  Avg Inference Latency  : {final_metrics['avg_latency_ms']} ms/frame")
    print(f"  Avg Frame Age          : {final_metrics['avg_frame_age_ms']} ms")
    print(f"  Dropped Stale Frames   : {final_metrics['dropped_frames']}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
