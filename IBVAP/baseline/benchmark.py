"""
IBVAP — Intelligent Border Video Analytics Platform
SIH26187 — Baseline Benchmark & Configuration Module
"""

import os
import time
import psutil

# ----------------------------------------------------
# GLOBAL CONFIGURATION SECTION (Single source of truth)
# ----------------------------------------------------
CONF_THRESHOLD = 0.25               # Default confidence threshold
SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')
DEFAULT_MODEL_NAME = "yolov8n.pt"

# Target classes of interest for border surveillance baseline
SURVEILLANCE_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle']

class BenchmarkTimer:
    """Accurate timing utility for latency & FPS benchmarking."""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def stop(self):
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

def get_system_utilization():
    """Returns current CPU and RAM usage percentage."""
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_info = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": round(ram_info.used / (1024**3), 2),
        "ram_total_gb": round(ram_info.total / (1024**3), 2),
        "ram_percent": ram_info.percent
    }
