"""
IBVAP Stage 2 — Latest-Frame Buffer Module
===========================================
Thread-safe Depth-1 single-slot frame buffer for real-time low-latency video processing.

Features:
- Fixed Buffer Depth = 1
- Zero unbounded queue latency build-up
- Instant overwriting of unread stale frames
- Capture FPS, Inference FPS, Frame Age (ms), and Dropped Frame metrics tracking
"""

import time
import threading


class LatestFrameBuffer:
    """Single-slot Depth-1 frame buffer separating capture and inference worker threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_slot = None
        self.timestamp_slot = 0.0

        # Performance Metrics
        self.capture_count = 0
        self.inference_count = 0
        self.dropped_frames = 0

        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.last_cap_tick = time.perf_counter()
        self.last_inf_tick = time.perf_counter()

        self.latencies = []
        self.frame_ages = []

    def push(self, frame):
        """
        Push a newly captured frame into the depth-1 buffer slot.
        If the slot already holds an unread frame, it is overwritten and logged as dropped.
        """
        now = time.perf_counter()
        with self.lock:
            if self.frame_slot is not None:
                self.dropped_frames += 1  # Overwrite unread stale frame

            self.frame_slot = frame
            self.timestamp_slot = now
            self.capture_count += 1

            elapsed = now - self.last_cap_tick
            if elapsed >= 1.0:
                self.capture_fps = round(self.capture_count / elapsed, 1)
                self.capture_count = 0
                self.last_cap_tick = now

    def pop(self):
        """
        Pop the latest frame from the depth-1 buffer slot.
        Returns: (frame, capture_timestamp, frame_age_ms) or (None, 0, 0)
        """
        now = time.perf_counter()
        with self.lock:
            if self.frame_slot is None:
                return None, 0.0, 0.0

            frame = self.frame_slot
            ts = self.timestamp_slot
            self.frame_slot = None  # Clear slot after reading

            self.inference_count += 1
            elapsed = now - self.last_inf_tick
            if elapsed >= 1.0:
                self.inference_fps = round(self.inference_count / elapsed, 1)
                self.inference_count = 0
                self.last_inf_tick = now

        frame_age_ms = (now - ts) * 1000.0
        self.frame_ages.append(frame_age_ms)
        if len(self.frame_ages) > 100:
            self.frame_ages.pop(0)

        return frame, ts, frame_age_ms

    def record_latency(self, latency_ms):
        """Record inference latency in milliseconds."""
        self.latencies.append(latency_ms)
        if len(self.latencies) > 100:
            self.latencies.pop(0)

    def get_metrics(self):
        """Return current performance metrics dictionary."""
        avg_latency = float(sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        avg_age = float(sum(self.frame_ages) / len(self.frame_ages)) if self.frame_ages else 0.0

        return {
            "capture_fps": self.capture_fps,
            "inference_fps": self.inference_fps,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_frame_age_ms": round(avg_age, 1),
            "dropped_frames": self.dropped_frames,
        }
