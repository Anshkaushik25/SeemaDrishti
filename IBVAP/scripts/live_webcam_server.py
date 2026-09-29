"""
IBVAP Stage 2 — Live Webcam Server with Fixed Multi-Person Pipeline
=====================================================================
FastAPI live demonstration server supporting:
- Multi-person detection with conf=0.15, iou=0.45, max_det=100
- Adaptive low-light visibility enhancement (CLAHE + Gamma + Denoising + Dehazing)
- Depth-1 latest-frame buffer latency management
- Side-by-side comparison feed & snapshot saving

Usage:
    python IBVAP/scripts/live_webcam_server.py
"""

import os
import sys
import time
import datetime
import threading
import cv2
import numpy as np
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Setup paths for module import
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from visibility_utils import (
    assess_visibility,
    LOW_LIGHT_THRESHOLD,
    HIGH_BRIGHTNESS_THRESHOLD,
    CONTRAST_THRESHOLD,
    BLUR_THRESHOLD,
    HAZE_THRESHOLD,
)
from enhancement import (
    enhance_conditionally,
    GAMMA,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DENOISE_STRENGTH,
    ENABLE_DEHAZING,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = PROJECT_DIR / r"training\runs\detect\mot17_05_baseline\weights\best.pt"
TEMP_WEB_DIR = PROJECT_DIR / r"temp_web"
SAMPLE_SAVE_DIR = PROJECT_DIR / r"evaluation\live_visibility_samples"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Multi-Person Fixed Inference Parameters
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
MAX_DET = 100
IMGSZ = 320  # Fast CPU inference
DEVICE = "cpu"

HOST = "127.0.0.1"
PORT = 8000
JPEG_QUALITY = 75


# ─── Shared State & Depth-1 Buffer ───────────────────────────────────────────

class AppState:
    """Thread-safe state shared between camera capture thread and FastAPI API endpoints."""

    def __init__(self):
        self.lock = threading.Lock()

        # Depth-1 Frame Buffer (Latest Frame Architecture)
        self.latest_raw_frame = None
        self.latest_frame_ts = 0.0
        self.dropped_frames = 0

        # JPEG Stream Bytes
        self.frame_orig = None
        self.frame_enh = None
        self.frame_comparison = None

        # Snapshot Caches
        self.raw_frame_cache = None
        self.enh_frame_cache = None
        self.comp_frame_cache = None

        # Server Status
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.display_fps = 0.0
        self.avg_frame_age_ms = 0.0
        self.status = "initializing"
        self.camera_running = False

        self.model = None
        self.cap = None
        self.capture_thread = None
        self.inference_thread = None
        self.stop_event = threading.Event()
        self.resolution = f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"

        # Control Mode (AUTO / ORIGINAL / ENHANCED)
        self.mode = "AUTO"

        # Quality & Visibility Metrics
        self.visibility = "GOOD"
        self.primary_degradation = "GOOD"
        self.action_summary = "NO ENHANCEMENT (RAW)"
        self.highlight_clipping = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.sharpness = 0.0
        self.haze_level = 0.0
        self.enhancement_active = False

        # Multi-Person Detections Telemetry
        self.orig_detections = 0
        self.orig_best_conf = 0.0
        self.enh_detections = 0
        self.enh_best_conf = 0.0

        # Latency Metrics (ms)
        self.orig_latency_ms = 0.0
        self.enh_latency_ms = 0.0


state = AppState()


# ─── Depth-1 Camera Capture Loop ─────────────────────────────────────────────

def capture_loop():
    """Continuously capture frames into depth-1 buffer, discarding stale unread frames."""
    cap_counter = 0
    cap_start = time.perf_counter()

    while not state.stop_event.is_set():
        if state.cap is None or not state.cap.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = state.cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        t_cap = time.perf_counter()

        with state.lock:
            if state.latest_raw_frame is not None:
                state.dropped_frames += 1  # Previous frame was overwritten unread

            state.latest_raw_frame = frame
            state.latest_frame_ts = t_cap

        cap_counter += 1
        elapsed = time.perf_counter() - cap_start
        if elapsed >= 1.0:
            with state.lock:
                state.capture_fps = round(cap_counter / elapsed, 1)
            cap_counter = 0
            cap_start = time.perf_counter()

        time.sleep(0.01)  # Prevent tight spin lock


# ─── Depth-1 Worker & Inference Loop ─────────────────────────────────────────

def inference_loop():
    """Pull latest frame from Depth-1 buffer, run quality assessment & dual multi-person YOLO."""
    inf_counter = 0
    inf_start = time.perf_counter()

    while not state.stop_event.is_set():
        raw_frame = None
        frame_ts = 0.0

        with state.lock:
            if state.latest_raw_frame is not None:
                raw_frame = state.latest_raw_frame
                frame_ts = state.latest_frame_ts
                state.latest_raw_frame = None  # Consume frame

        if raw_frame is None:
            time.sleep(0.01)
            continue

        # Compute Frame Age
        frame_age_ms = (time.perf_counter() - frame_ts) * 1000.0
        h, w = raw_frame.shape[:2]

        # Step 1: Quality / Visibility Assessment
        qual = assess_visibility(raw_frame)
        is_poor = qual["is_poor"]

        # Step 2: Determine Enhancement Status based on Mode
        with state.lock:
            current_mode = state.mode

        if current_mode == "AUTO":
            enhanced_frame = enhance_conditionally(raw_frame, qual)
            enhancement_active = not np.array_equal(raw_frame, enhanced_frame)
        elif current_mode == "ENHANCED":
            enhanced_frame = enhance_conditionally(raw_frame, qual)
            enhancement_active = True
        else:  # ORIGINAL
            enhanced_frame = raw_frame.copy()
            enhancement_active = False

        # Step 4: YOLO Inference on ORIGINAL Frame (Multi-Person, max_det=100)
        t0_orig = time.perf_counter()
        try:
            results_orig = state.model.predict(
                source=raw_frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DET,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )[0]
        except Exception as e:
            print(f"  [ERROR] Inference (Orig) failed: {e}")
            continue
        orig_lat = (time.perf_counter() - t0_orig) * 1000.0

        # Step 5: YOLO Inference on ENHANCED Frame (Multi-Person, max_det=100)
        t0_enh = time.perf_counter()
        try:
            results_enh = state.model.predict(
                source=enhanced_frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DET,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )[0]
        except Exception as e:
            print(f"  [ERROR] Inference (Enh) failed: {e}")
            continue
        enh_lat = (time.perf_counter() - t0_enh) * 1000.0

        # Annotate ORIGINAL Frame with ALL Person Detections
        annotated_orig = raw_frame.copy()
        orig_boxes = []
        orig_confs = []
        for box in results_orig.boxes:
            if int(box.cls[0]) == 0:  # person class
                c = float(box.conf[0])
                orig_confs.append(c)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    orig_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(annotated_orig, (x1, y1), (x2, y2), (0, 220, 100), 2)
                    lbl = f"person {c:.2f}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_orig, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, max(y1, lh + 6)), (0, 220, 100), -1)
                    cv2.putText(annotated_orig, lbl, (x1 + 2, max(y1 - 3, lh + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(annotated_orig, f"ORIGINAL (RAW) | {len(orig_boxes)} Persons", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Annotate ENHANCED Frame with ALL Person Detections
        annotated_enh = enhanced_frame.copy()
        enh_boxes = []
        enh_confs = []
        for box in results_enh.boxes:
            if int(box.cls[0]) == 0:  # person class
                c = float(box.conf[0])
                enh_confs.append(c)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    enh_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(annotated_enh, (x1, y1), (x2, y2), (0, 220, 100), 2)
                    lbl = f"person {c:.2f}"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_enh, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, max(y1, lh + 6)), (0, 220, 100), -1)
                    cv2.putText(annotated_enh, lbl, (x1 + 2, max(y1 - 3, lh + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        status_txt = f"ENHANCED ({'ACTIVE' if enhancement_active else 'INACTIVE'}) | {len(enh_boxes)} Persons"
        badge_color = (0, 255, 120) if enhancement_active else (150, 150, 150)
        cv2.putText(annotated_enh, status_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2)

        # Side-by-Side Comparison Feed Body
        annotated_comp = np.hstack([annotated_orig, annotated_enh])

        # Inference FPS Calculation
        inf_counter += 1
        elapsed = time.perf_counter() - inf_start
        current_inf_fps = inf_counter / elapsed if elapsed >= 1.0 else state.inference_fps
        if elapsed >= 1.0:
            inf_counter = 0
            inf_start = time.perf_counter()

        # Encode JPEG Streams
        _, jpeg_orig = cv2.imencode('.jpg', annotated_orig, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        _, jpeg_enh = cv2.imencode('.jpg', annotated_enh, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        _, jpeg_comp = cv2.imencode('.jpg', annotated_comp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # Update Shared State
        with state.lock:
            state.frame_orig = jpeg_orig.tobytes()
            state.frame_enh = jpeg_enh.tobytes()
            state.frame_comparison = jpeg_comp.tobytes()

            state.raw_frame_cache = raw_frame
            state.enh_frame_cache = enhanced_frame
            state.comp_frame_cache = annotated_comp

            state.inference_fps = round(current_inf_fps, 1)
            state.display_fps = round(current_inf_fps, 1)
            state.avg_frame_age_ms = round(frame_age_ms, 1)

            state.visibility = "POOR" if is_poor else "GOOD"
            state.primary_degradation = qual.get("primary_degradation", "GOOD")
            state.action_summary = qual.get("action_summary", "NO ENHANCEMENT (RAW)")
            state.highlight_clipping = qual.get("highlight_clipping", 0.0)
            state.brightness = qual["brightness"]
            state.contrast = qual["contrast"]
            state.sharpness = qual["sharpness"]
            state.haze_level = qual.get("haze_level", 0.0)
            state.enhancement_active = enhancement_active

            state.orig_detections = len(orig_boxes)
            state.orig_best_conf = round(float(np.max(orig_confs)), 2) if orig_confs else 0.0

            state.enh_detections = len(enh_boxes)
            state.enh_best_conf = round(float(np.max(enh_confs)), 2) if enh_confs else 0.0

            state.orig_latency_ms = round(orig_lat, 1)
            state.enh_latency_ms = round(enh_lat, 1)

            state.status = "running"
            state.camera_running = True


def start_camera():
    """Open camera and launch capture + inference threads."""
    if state.cap is not None and state.cap.isOpened():
        return True

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("  [ERROR] Could not open webcam.")
        with state.lock:
            state.status = "camera_error"
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state.cap = cap
    state.resolution = f"{actual_w}x{actual_h}"
    state.stop_event.clear()

    print(f"  Camera opened: {actual_w}x{actual_h}")

    # Launch Capture Thread
    state.capture_thread = threading.Thread(target=capture_loop, daemon=True)
    state.capture_thread.start()

    # Launch Inference Thread
    state.inference_thread = threading.Thread(target=inference_loop, daemon=True)
    state.inference_thread.start()

    return True


def stop_camera():
    """Stop worker threads and release camera."""
    state.stop_event.set()
    if state.capture_thread is not None:
        state.capture_thread.join(timeout=2.0)
        state.capture_thread = None
    if state.inference_thread is not None:
        state.inference_thread.join(timeout=2.0)
        state.inference_thread = None
    if state.cap is not None:
        state.cap.release()
        state.cap = None
    with state.lock:
        state.camera_running = False
        state.status = "stopped"
        state.display_fps = 0.0
        state.frame_orig = None
        state.frame_enh = None
        state.frame_comparison = None
    print("  Camera stopped.")


# ─── FastAPI Web Application ──────────────────────────────────────────────────

app = FastAPI(title="IBVAP Stage 2 Multi-Person Detection & Visibility Enhancement")

app.mount("/static", StaticFiles(directory=str(TEMP_WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = TEMP_WEB_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/video_feed/original")
async def video_feed_original():
    def generate():
        while True:
            with state.lock:
                f = state.frame_orig
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed/enhanced")
async def video_feed_enhanced():
    def generate():
        while True:
            with state.lock:
                f = state.frame_enh
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed")
async def video_feed_side_by_side():
    def generate():
        while True:
            with state.lock:
                f = state.frame_comparison
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def api_status():
    """Return live system telemetry."""
    with state.lock:
        return JSONResponse({
            "model": "YOLOv8n (Person Only)",
            "task": "Multi-Person Detection & Visibility Enhancement",
            "device": "CPU",
            "camera": "Laptop Webcam",
            "resolution": state.resolution,
            "capture_fps": state.capture_fps,
            "inference_fps": state.inference_fps,
            "fps": state.display_fps,
            "frame_age_ms": state.avg_frame_age_ms,
            "dropped_frames": state.dropped_frames,
            "status": state.status,
            "camera_running": state.camera_running,
            "mode": state.mode,
            "visibility": state.visibility,
            "primary_degradation": state.primary_degradation,
            "action_summary": state.action_summary,
            "highlight_clipping": state.highlight_clipping,
            "brightness": state.brightness,
            "contrast": state.contrast,
            "sharpness": state.sharpness,
            "haze_level": state.haze_level,
            "enhancement_active": state.enhancement_active,
            "orig_detections": state.orig_detections,
            "orig_best_conf": state.orig_best_conf,
            "enh_detections": state.enh_detections,
            "enh_best_conf": state.enh_best_conf,
            "orig_latency_ms": state.orig_latency_ms,
            "enh_latency_ms": state.enh_latency_ms,
        })


@app.post("/api/mode")
async def api_set_mode(req: Request):
    body = await req.json()
    new_mode = body.get("mode", "AUTO").upper()
    if new_mode in ["AUTO", "ORIGINAL", "ENHANCED"]:
        with state.lock:
            state.mode = new_mode
        return JSONResponse({"success": True, "mode": new_mode})
    return JSONResponse({"success": False, "error": "Invalid mode"}, status_code=400)


@app.post("/api/camera/start")
async def api_camera_start():
    ok = start_camera()
    return JSONResponse({"success": ok, "message": "Camera started" if ok else "Failed to start camera"})


@app.post("/api/camera/stop")
async def api_camera_stop():
    stop_camera()
    return JSONResponse({"success": True, "message": "Camera stopped"})


@app.post("/api/save_comparison")
async def api_save_comparison():
    SAMPLE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with state.lock:
        raw = state.raw_frame_cache
        enh = state.enh_frame_cache
        comp = state.comp_frame_cache

    if raw is None or enh is None or comp is None:
        return JSONResponse({"success": False, "error": "No frames available"}, status_code=400)

    p_raw = SAMPLE_SAVE_DIR / f"sample_{ts}_raw.jpg"
    p_enh = SAMPLE_SAVE_DIR / f"sample_{ts}_enh.jpg"
    p_comp = SAMPLE_SAVE_DIR / f"sample_{ts}_side_by_side.jpg"

    cv2.imwrite(str(p_raw), raw)
    cv2.imwrite(str(p_enh), enh)
    cv2.imwrite(str(p_comp), comp)

    return JSONResponse({
        "success": True,
        "timestamp": ts,
        "saved_files": [p_raw.name, p_enh.name, p_comp.name],
        "output_directory": str(SAMPLE_SAVE_DIR)
    })


@app.on_event("shutdown")
def shutdown_event():
    stop_camera()
    print("\n  Server shut down. Resources released.")


# ─── Main Entry Point ────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  IBVAP STAGE 2 — MULTI-PERSON DETECTION & VISIBILITY SERVER")
    print("=" * 60)
    print(f"  Model Weight : {MODEL_PATH.name}")
    print(f"  Inference    : conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, max_det={MAX_DET}, imgsz={IMGSZ}")
    print(f"  Architecture : Depth-1 Latest Frame Buffer")
    print(f"  Enhancement  : CLAHE + Gamma + Denoising + Dehazing")
    print()

    if not MODEL_PATH.is_file():
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"  Loading model: {MODEL_PATH.name}")
    from ultralytics import YOLO
    state.model = YOLO(str(MODEL_PATH))
    print("  Model loaded successfully.")
    print()

    print("  Starting camera capture & inference threads...")
    start_camera()
    print()

    print(f"  Open Web UI: http://{HOST}:{PORT}")
    print("  Press CTRL+C to stop.")
    print("=" * 60)
    print()

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
