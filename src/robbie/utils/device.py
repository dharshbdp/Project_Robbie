"""Hardware and device assessment utilities for Project Robbie."""

from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class HardwareReport:
    device_type: str
    device_name: str
    cuda_available: bool
    device_count: int
    vram_total_gb: float
    vram_free_gb: float

def get_hardware_info() -> HardwareReport:
    """Detect available compute resources (CUDA GPU / CPU) and memory."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Memory allocated vs reserved
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            free_vram = max(0.0, total_vram - reserved)
            return HardwareReport(
                device_type="cuda",
                device_name=device_name,
                cuda_available=True,
                device_count=device_count,
                vram_total_gb=round(total_vram, 2),
                vram_free_gb=round(free_vram, 2),
            )
    except ImportError:
        pass

    # Fallback to CPU
    cpu_count = os.cpu_count() or 1
    return HardwareReport(
        device_type="cpu",
        device_name=f"CPU ({cpu_count} logical cores)",
        cuda_available=False,
        device_count=1,
        vram_total_gb=0.0,
        vram_free_gb=0.0,
    )

def get_best_device() -> str:
    """Return 'cuda' if GPU acceleration is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
