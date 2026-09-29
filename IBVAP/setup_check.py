"""
IBVAP — Intelligent Border Video Analytics Platform
SIH26187 — Baseline Environment Setup Check
"""

import sys
import platform
import psutil

def check_environment():
    """Checks the Python, PyTorch, CUDA, and Ultralytics environment."""
    print("=" * 50)
    print("IBVAP ENVIRONMENT CHECK")
    print("=" * 50)

    # 1. Python version
    py_ver = sys.version.split()[0]
    print(f"Python: {py_ver} ({platform.platform()})")

    # 2. PyTorch version & CUDA
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_ver = "NOT INSTALLED"
        cuda_available = False
        torch = None

    print(f"PyTorch: {torch_ver}")

    # 3. Ultralytics version
    try:
        import ultralytics
        ultra_ver = ultralytics.__version__
    except ImportError:
        ultra_ver = "NOT INSTALLED"

    print(f"Ultralytics: {ultra_ver}")
    print(f"CUDA available: {cuda_available}")

    # 4. GPU information
    device_name = "N/A (CPU Mode)"
    gpu_mem = "N/A"
    device_target = "cpu"

    if cuda_available and torch is not None:
        try:
            device_name = torch.cuda.get_device_name(0)
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_mem = f"{mem_bytes / (1024**3):.2f} GB"
            device_target = 0
        except Exception as e:
            device_name = f"Error querying GPU: {e}"
            device_target = "cpu"
    else:
        # System RAM info for CPU mode
        ram_gb = psutil.virtual_memory().total / (1024**3)
        gpu_mem = f"System RAM: {ram_gb:.2f} GB"

    print(f"GPU: {device_name}")
    print(f"GPU memory: {gpu_mem}")
    print("=" * 50)

    return {
        "python": py_ver,
        "pytorch": torch_ver,
        "ultralytics": ultra_ver,
        "cuda_available": cuda_available,
        "gpu_name": device_name,
        "gpu_memory": gpu_mem,
        "device": device_target,
        "os": platform.platform()
    }

if __name__ == "__main__":
    check_environment()
