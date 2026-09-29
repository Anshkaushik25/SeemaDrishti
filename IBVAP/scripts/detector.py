"""
IBVAP Stage 2 — Person Detector Module (GPU / CPU Configurable)
================================================================
YOLOv8n Person Detector wrapper supporting configurable GPU (device=0 / device='cuda:0')
and CPU execution with automatic CUDA fallback.

Features:
- Configurable device (auto-detects CUDA GPU if available)
- Configurable conf=0.15, iou=0.45, max_det=100, imgsz=640
- Multi-person detection extraction
- Detection IDs per frame (Person #1, Person #2, ...) without tracking
- Coordinate boundary safety clamping
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# Default Model Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / r"models\best_detector_v2.pt"
FALLBACK_MULTICLASS_PATH = PROJECT_DIR / r"models\best_multiclass.pt"
FALLBACK_MODEL_PATH = PROJECT_DIR / r"models\best.pt"


class PersonDetector:
    """YOLOv8n Multi-Class Object Detector Wrapper supporting GPU & CPU execution."""

    def __init__(self, model_path=None, conf=0.15, iou=0.45, max_det=100, imgsz=640, device=None):
        if model_path is None:
            if DEFAULT_MODEL_PATH.is_file():
                model_path = DEFAULT_MODEL_PATH
            elif FALLBACK_MULTICLASS_PATH.is_file():
                model_path = FALLBACK_MULTICLASS_PATH
            else:
                model_path = FALLBACK_MODEL_PATH

        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.imgsz = imgsz

        # Device selection logic: Default to GPU (device=0) if CUDA available
        if device is None or str(device).lower() in ["auto", "gpu", "cuda"]:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model weights not found at: {self.model_path}")

        print(f"  [Detector] Loading YOLOv8n weights: {self.model_path.name}")
        print(f"  [Detector] Inference Device: {self.device} (CUDA Available: {torch.cuda.is_available()})")

        self.model = YOLO(str(self.model_path))
        self.names = self.model.names
        print(f"  [Detector] Model classes loaded: {self.names}")

        # Class Color Palette
        self.class_colors = {
            0: (0, 220, 100),   # person (Green)
            1: (255, 140, 0),   # car (Blue)
            2: (200, 0, 200),   # truck (Purple)
            3: (0, 200, 255),   # bike (Yellow/Cyan)
            4: (0, 100, 255),   # patrol_vehicle (Red/Orange)
        }

    def detect(self, frame):
        """
        Run multi-class object detection on input frame.

        Returns list of detection dictionaries:
        [
            {
                "id": 1,
                "cls_id": 0,
                "class_name": "person",
                "label": "person #1 (0.87)",
                "conf": 0.87,
                "bbox": [x1, y1, x2, y2]
            }, ...
        ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # PyTorch CUDA synchronization if running on GPU
        if isinstance(self.device, int) or (isinstance(self.device, str) and "cuda" in self.device.lower()):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        if isinstance(self.device, int) or (isinstance(self.device, str) and "cuda" in self.device.lower()):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        detections = []
        det_idx = 1

        for box in results.boxes:
            cls_id = int(box.cls[0])
            c = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Boundary safety clamping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                c_name = self.names.get(cls_id, f"cls_{cls_id}")
                detections.append({
                    "id": det_idx,
                    "cls_id": cls_id,
                    "class_name": c_name,
                    "label": f"{c_name} #{det_idx} ({c:.2f})",
                    "conf": round(c, 2),
                    "bbox": [x1, y1, x2, y2],
                })
                det_idx += 1

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and detection labels on a copy of the frame.
        Does NOT modify the input frame array.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            cls_id = det.get("cls_id", 0)
            color = self.class_colors.get(cls_id, (0, 220, 100))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, max(y1, lh + 6)), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(y1 - 3, lh + 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

        return annotated
