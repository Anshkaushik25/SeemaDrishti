"""
IBVAP — Intelligent Border Video Analytics Platform
SIH26187 — Master Baseline Execution Pipeline
"""

import os
import sys
import glob
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from setup_check import check_environment
from baseline.benchmark import CONF_THRESHOLD, SUPPORTED_EXTENSIONS, DEFAULT_MODEL_NAME
from baseline.evaluate_video import evaluate_single_video
from baseline.generate_report import generate_summary_csv, generate_markdown_report

def create_project_directories():
    """Ensures all project directory structures exist."""
    dirs = [
        PROJECT_ROOT / "videos",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "results" / "annotated",
        PROJECT_ROOT / "results" / "metrics",
        PROJECT_ROOT / "results" / "logs",
        PROJECT_ROOT / "results" / "reports",
        PROJECT_ROOT / "dataset"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def find_videos(videos_dir: Path) -> list:
    """Discovers all supported surveillance videos in the videos directory."""
    video_files = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(videos_dir.glob(f"*{ext}"))
        video_files.extend(videos_dir.glob(f"*{ext.upper()}"))
    return sorted(list(set(video_files)))

def main():
    print("\n" + "=" * 60)
    print("  IBVAP — INTELLIGENT BORDER VIDEO ANALYTICS PLATFORM")
    print("  SIH26187 Stage 1 Baseline Inference Pipeline")
    print("=" * 60)

    # 1. Environment Check
    env_info = check_environment()
    create_project_directories()

    videos_dir = PROJECT_ROOT / "videos"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"

    # 2. Discover Input Videos
    video_paths = find_videos(videos_dir)
    if not video_paths:
        print(f"\n[Warning] No video files found in '{videos_dir}'.")
        print("Please place your surveillance/CCTV video clips (.mp4, .avi, .mkv, .mov) inside:")
        print(f"  --> {videos_dir}\n")
        
        # If no video is present, let's create a minimal sample test video so the baseline can run end-to-end
        sample_path = videos_dir / "cctv_sample_test.mp4"
        print(f"Generating a lightweight synthetic sample surveillance clip for baseline testing: {sample_path.name}")
        import cv2
        import numpy as np
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_sample = cv2.VideoWriter(str(sample_path), fourcc, 25.0, (640, 480))
        for i in range(75): # 3 seconds at 25 fps
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw synthetic background
            cv2.rectangle(img, (0, 300), (640, 480), (40, 50, 40), -1) # ground
            cv2.rectangle(img, (0, 0), (640, 300), (20, 20, 30), -1)   # sky
            # Moving object simulation
            cx = int(50 + (i * 6.5))
            cy = 340
            cv2.circle(img, (cx, cy), 20, (200, 200, 200), -1) # simulated human/vehicle
            cv2.putText(img, f"CCTV CAM-01 [BORDER SECTOR A] Frame: {i+1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            out_sample.write(img)
        out_sample.release()
        video_paths = [sample_path]

    print(f"\n[Videos] Found {len(video_paths)} surveillance video(s) to evaluate:")
    for vp in video_paths:
        print(f"  - {vp.name}")

    # 3. Load YOLOv8n Pretrained Model
    model_path = models_dir / DEFAULT_MODEL_NAME
    print(f"\n[Model] Loading YOLOv8n model...")
    try:
        # If model is not in models_dir, Ultralytics downloads it, then we load
        model = YOLO(str(model_path) if model_path.exists() else DEFAULT_MODEL_NAME)
        # If downloaded to working dir, move or copy to models_dir if needed
        if not model_path.exists() and os.path.exists(DEFAULT_MODEL_NAME):
            import shutil
            shutil.move(DEFAULT_MODEL_NAME, str(model_path))
            model = YOLO(str(model_path))
        print(f"[Model] Successfully loaded YOLOv8n ({model_path.name if model_path.exists() else 'pretrained weights'})")
    except Exception as e:
        print(f"[Error] Failed to load YOLOv8n model: {e}")
        return

    # 4. Process Each Video Independently
    all_results = []
    device = env_info["device"]

    for vp in video_paths:
        res = evaluate_single_video(
            video_path=str(vp),
            model=model,
            device=device,
            output_annotated_dir=str(results_dir / "annotated"),
            output_metrics_dir=str(results_dir / "metrics"),
            output_logs_dir=str(results_dir / "logs"),
            conf_threshold=CONF_THRESHOLD
        )
        all_results.append(res)

    # 5. Generate Consolidated Metrics & Baseline Report
    summary_csv_path = str(results_dir / "metrics" / "baseline_summary.csv")
    report_md_path = str(results_dir / "reports" / "baseline_report.md")

    generate_summary_csv(all_results, summary_csv_path)
    generate_markdown_report(env_info, all_results, report_md_path)

    # 6. Final Terminal Output
    print("\n" + "=" * 40)
    print("IBVAP YOLOv8n BASELINE COMPLETE")
    print("=" * 40)
    print(f"\nModel: YOLOv8n pretrained")
    print(f"Device: {env_info.get('gpu_name') if env_info.get('cuda_available') else 'CPU'}")
    print(f"\nVideos processed: {len([r for r in all_results if 'error' not in r])}")
    print("\n{:<25} {:<10} {:<15}".format("Video", "FPS", "Latency"))
    print("-" * 50)
    for r in all_results:
        if "error" not in r:
            print("{:<25} {:<10.1f} {:<15.1f} ms".format(
                r.get("video", "N/A"),
                r.get("processing_fps", 0.0),
                r.get("average_latency_ms", 0.0)
            ))
        else:
            print("{:<25} ERROR: {}".format(r.get("video", "N/A"), r.get("error", "")))

    print("\nResults:")
    print("Annotated videos: results/annotated/")
    print("Metrics:          results/metrics/")
    print("Logs:             results/logs/")
    print("Report:           results/reports/baseline_report.md")
    print("\nIMPORTANT:")
    print("No accuracy/mAP/precision/recall was reported because")
    print("ground-truth annotations are not available yet.")
    print("\nDo not start any training.")
    print("Stop after completing this baseline stage.")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    main()
