"""
IBVAP Stage 1 — Live Webcam Person Detection
=============================================
Qualitative live inference test using the trained YOLOv8n baseline model
on the laptop's built-in webcam.

This is NOT a benchmark, accuracy evaluation, or deployment test.
This is a live viewing/inference demonstration only.

Controls:
  Q   - Quit
  ESC - Quit

Project: SIH26187 - Intelligent Border Video Analytics Platform
"""

import sys
import os
import time

# ---- Configuration ----

MODEL_PATH = r"C:\Users\Atharva\Downloads\SeemaDrishti\IBVAP\training\runs\detect\mot17_05_baseline\weights\best.pt"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONF_THRESHOLD = 0.25
IMGSZ = 640
DEVICE = "cpu"
WINDOW_NAME = "IBVAP - YOLOv8n Live Person Detection"
STATUS_INTERVAL = 5.0  # seconds between terminal status prints


def main():
    # ---- Banner ----
    print("=" * 50)
    print("  STAGE 1 LIVE INFERENCE TEST")
    print("=" * 50)
    print(f"  Model:  YOLOv8n")
    print(f"  Task:   Person Detection")
    print(f"  Source: Laptop Webcam")
    print(f"  Device: CPU")
    print("=" * 50)
    print()

    # ---- Check model file ----
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found:")
        print(f"  {MODEL_PATH}")
        sys.exit(1)

    print(f"[1/3] Loading model: {MODEL_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed.")
        print("  Install with: pip install ultralytics")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python package not installed.")
        print("  Install with: pip install opencv-python")
        sys.exit(1)

    model = YOLO(MODEL_PATH)
    print(f"  Model loaded successfully.")
    print(f"  Classes: {model.names}")
    print()

    # ---- Open webcam ----
    print(f"[2/3] Opening webcam (index {CAMERA_INDEX})...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback without DirectShow
        print("  DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print(f"  Camera index: {CAMERA_INDEX}")
        print("  Check that no other application is using the camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Camera opened successfully.")
    print(f"  Requested resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  Actual resolution:    {actual_w}x{actual_h}")
    print()

    # ---- Inference loop ----
    print(f"[3/3] Starting live inference...")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  Press Q or ESC to quit.")
    print()

    frame_count = 0
    fps = 0.0
    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    last_status_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("WARNING: Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            t_start = time.perf_counter()

            # Run YOLOv8 inference
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )

            # Draw detections on frame
            result = results[0]
            annotated = frame.copy()

            person_count = 0
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Only draw person class (id 0)
                if cls_id == 0:
                    person_count += 1
                    # Box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Label
                    label = f"PERSON {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - label_size[1] - 8),
                        (x1 + label_size[0] + 4, y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated, label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2,
                    )

            # Calculate FPS
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.perf_counter()

            # Draw FPS and person count on frame
            info_text = f"FPS: {fps:.1f}  |  Persons: {person_count}"
            cv2.putText(
                annotated, info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2,
            )

            # Stage label
            cv2.putText(
                annotated, "IBVAP Stage 1 - Live Test",
                (10, actual_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1,
            )

            # Show frame
            cv2.imshow(WINDOW_NAME, annotated)

            # Terminal status printout
            now = time.perf_counter()
            if now - last_status_time >= STATUS_INTERVAL:
                print(f"  Camera: OK | Resolution: {actual_w}x{actual_h} | "
                      f"Model: YOLOv8n | Device: CPU | "
                      f"FPS: {fps:.2f} | Persons: {person_count}")
                last_status_time = now

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print()
                print("  User requested exit.")
                break

    except KeyboardInterrupt:
        print()
        print("  Interrupted by user (Ctrl+C).")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print()
        print("  Camera released. Window closed.")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Last observed FPS: {fps:.2f}")
        print()
        print("=" * 50)
        print("  LIVE INFERENCE TEST ENDED")
        print("=" * 50)


if __name__ == "__main__":
    main()
