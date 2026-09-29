"""
IBVAP — Intelligent Border Video Analytics Platform
SIH26187 — Video Evaluation & Baseline Inference Pipeline
"""

import os
import time
import logging
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from baseline.benchmark import CONF_THRESHOLD, SURVEILLANCE_CLASSES

def setup_video_logger(video_name: str, log_dir: str) -> logging.Logger:
    """Configures a dedicated logger for a specific video processing job."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{video_name}.log")
    
    logger = logging.getLogger(video_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Clear existing handlers to avoid duplicates

    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def evaluate_single_video(
    video_path: str,
    model,
    device: str,
    output_annotated_dir: str,
    output_metrics_dir: str,
    output_logs_dir: str,
    conf_threshold: float = CONF_THRESHOLD
) -> dict:
    """
    Evaluates a single surveillance/CCTV video with pretrained YOLOv8n.
    Measures latency, processing FPS, class distributions, and confidence statistics.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    logger = setup_video_logger(base_name, output_logs_dir)
    start_time_iso = datetime.now().isoformat()

    logger.info(f"Starting baseline inference on: {video_path}")
    logger.info(f"Model: YOLOv8n (Pretrained) | Device: {device} | Confidence Threshold: {conf_threshold}")

    if not os.path.exists(video_path):
        err_msg = f"Video file not found: {video_path}"
        logger.error(err_msg)
        return {"error": err_msg, "video": os.path.basename(video_path)}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err_msg = f"Failed to open video source: {video_path}. File may be corrupted or unsupported."
        logger.error(err_msg)
        cap.release()
        return {"error": err_msg, "video": os.path.basename(video_path)}

    # Video properties
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution_str = f"{source_width}x{source_height}"

    logger.info(f"Video Resolution: {resolution_str} | Source FPS: {source_fps:.2f} | Total Frames: {total_frames}")

    # Output video writer setup
    os.makedirs(output_annotated_dir, exist_ok=True)
    annotated_filename = f"{base_name}_yolov8n_baseline.mp4"
    annotated_filepath = os.path.join(output_annotated_dir, annotated_filename)

    # Use mp4v codec for cross-platform compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(annotated_filepath, fourcc, source_fps, (source_width, source_height))

    # Metrics storage
    detection_records = []
    frame_latencies_ms = []
    processed_frames = 0
    person_count = 0
    vehicle_count = 0

    print(f"\n[Processing] Video: {os.path.basename(video_path)} ({resolution_str}, {total_frames} frames)")

    wall_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            frame_t0 = time.perf_counter()

            # Run YOLOv8 inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                device=device,
                verbose=False
            )

            frame_t1 = time.perf_counter()
            inference_ms = (frame_t1 - frame_t0) * 1000.0
            frame_latencies_ms.append(inference_ms)

            # Process detections on current frame
            annotated_frame = frame.copy()

            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    conf_val = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    detection_records.append({
                        "frame": processed_frames,
                        "class": cls_name,
                        "confidence": conf_val
                    })

                    # Track surveillance classes
                    if cls_name == "person":
                        person_count += 1
                    elif cls_name in ["car", "truck", "bus", "motorcycle"]:
                        vehicle_count += 1

                    # Visual annotations
                    x1, y1, x2, y2 = xyxy
                    color = (0, 165, 255) if cls_name == "person" else (0, 255, 0) if cls_name in ["car", "truck", "bus", "motorcycle"] else (255, 200, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{cls_name} {conf_val:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - 20)), (x1 + w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Add HUD banner on top
            hud_text = f"IBVAP Baseline | YOLOv8n | Frame: {processed_frames}/{total_frames} | Latency: {inference_ms:.1f}ms"
            cv2.putText(annotated_frame, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            out_writer.write(annotated_frame)

            # Progress feedback in console
            if processed_frames % 50 == 0 or processed_frames == total_frames:
                avg_so_far = np.mean(frame_latencies_ms) if frame_latencies_ms else 0
                print(f"  -> Frame {processed_frames}/{total_frames} | Current Latency: {inference_ms:.1f}ms | Avg: {avg_so_far:.1f}ms", end='\r')

    except Exception as e:
        logger.error(f"Error during video processing loop: {e}", exc_info=True)
    finally:
        cap.release()
        out_writer.release()

    wall_end = time.perf_counter()
    elapsed_seconds = wall_end - wall_start
    processing_fps = (processed_frames / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    avg_latency_ms = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0

    print(f"\n[Completed] Processed {processed_frames} frames in {elapsed_seconds:.2f}s ({processing_fps:.2f} FPS, Avg Latency: {avg_latency_ms:.2f}ms)")

    # ----------------------------------------------------
    # CONFIDENCE STATISTICS & CLASS DISTRIBUTION CSV
    # ----------------------------------------------------
    os.makedirs(output_metrics_dir, exist_ok=True)
    metrics_csv_path = os.path.join(output_metrics_dir, f"{base_name}_metrics.csv")

    class_stats = []
    if detection_records:
        df_det = pd.DataFrame(detection_records)
        grouped = df_det.groupby("class")
        for cls_name, grp in grouped:
            class_stats.append({
                "video": base_name,
                "class": cls_name,
                "detection_count": len(grp),
                "mean_confidence": round(float(grp["confidence"].mean()), 4),
                "min_confidence": round(float(grp["confidence"].min()), 4),
                "max_confidence": round(float(grp["confidence"].max()), 4)
            })
    
    # Save per-video metrics CSV
    df_metrics = pd.DataFrame(class_stats)
    df_metrics.to_csv(metrics_csv_path, index=False)

    # Logging summary
    logger.info(f"Processed Frames: {processed_frames} / {total_frames}")
    logger.info(f"Elapsed Time: {elapsed_seconds:.2f}s | Processing FPS: {processing_fps:.2f} | Avg Latency: {avg_latency_ms:.2f}ms")
    logger.info(f"Detected Persons: {person_count} | Detected Vehicles: {vehicle_count}")
    logger.info(f"Annotated Video Output: {annotated_filepath}")
    logger.info(f"Metrics Output: {metrics_csv_path}")
    logger.info("Video baseline evaluation finished successfully.")

    return {
        "video": os.path.basename(video_path),
        "resolution": resolution_str,
        "source_fps": round(source_fps, 2),
        "frame_count": total_frames,
        "processed_frames": processed_frames,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "processing_fps": round(processing_fps, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "device": str(device).upper(),
        "person_detections": person_count,
        "vehicle_detections": vehicle_count,
        "class_stats": class_stats,
        "annotated_path": annotated_filepath,
        "metrics_csv": metrics_csv_path,
        "log_path": os.path.join(output_logs_dir, f"{base_name}.log")
    }
