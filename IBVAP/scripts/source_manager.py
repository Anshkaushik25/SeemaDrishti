"""
IBVAP Stage 2 — Source Manager Module
======================================
Unified input source abstraction supporting:
1. Video files (.mp4, .avi, .mkv)
2. Local Webcams (index integer 0, 1, ...)
3. RTSP / IP CCTV network streams (rtsp://...)

Features automatic reconnection with exponential backoff for network RTSP streams.
"""

import time
import cv2
from pathlib import Path


class SourceManager:
    """Unified Video / Webcam / RTSP Stream Reader with Auto-Reconnect for RTSP."""

    def __init__(self, source_input=0, target_width=640, target_height=480):
        self.source_input = source_input
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        self.source_type = self._determine_source_type(source_input)

        # RTSP reconnect state
        self.reconnect_count = 0
        self.max_reconnect_attempts = 10
        self.last_reconnect_time = 0.0

        self.open()

    def _determine_source_type(self, source_input):
        """Determine whether input is WEBCAM, VIDEO_FILE, or RTSP_STREAM."""
        if isinstance(source_input, int) or (isinstance(source_input, str) and source_input.isdigit()):
            return "WEBCAM"
        elif isinstance(source_input, str) and (source_input.startswith("rtsp://") or source_input.startswith("http://")):
            return "RTSP_STREAM"
        else:
            return "VIDEO_FILE"

    def open(self):
        """Open the video stream or file based on source type."""
        if self.cap is not None:
            self.cap.release()

        src = int(self.source_input) if self.source_type == "WEBCAM" else self.source_input

        if self.source_type == "WEBCAM":
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        elif self.source_type == "RTSP_STREAM":
            # Set OpenCV RTSP environment parameters
            self.cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)

        else:  # VIDEO_FILE
            self.cap = cv2.VideoCapture(str(src))

        return self.is_opened()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        Read next frame. For RTSP, handles automatic reconnection on network drop.
        Returns: (ret, frame)
        """
        if not self.is_opened():
            if self.source_type == "RTSP_STREAM":
                self._handle_reconnect()
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            if self.source_type == "RTSP_STREAM":
                print(f"  [RTSP WARNING] Stream frame dropped. Attempting reconnect...")
                self._handle_reconnect()
                if self.is_opened():
                    ret, frame = self.cap.read()
            elif self.source_type == "VIDEO_FILE":
                # End of video file reached
                pass

        return ret, frame

    def _handle_reconnect(self):
        """Exponential backoff reconnect logic for RTSP streams."""
        now = time.time()
        if now - self.last_reconnect_time < 2.0:
            return False

        self.last_reconnect_time = now
        self.reconnect_count += 1
        backoff = min(10.0, 1.5 ** self.reconnect_count)
        print(f"  [RTSP RECONNECT] Attempt {self.reconnect_count}/{self.max_reconnect_attempts} in {backoff:.1f}s...")
        time.sleep(backoff)

        success = self.open()
        if success:
            print("  [RTSP RECONNECT] Stream re-established successfully.")
            self.reconnect_count = 0
        return success

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_info(self):
        return {
            "source": str(self.source_input),
            "source_type": self.source_type,
            "is_opened": self.is_opened(),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.is_opened() else 0,
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.is_opened() else 0,
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)) if self.is_opened() else 0.0,
        }
