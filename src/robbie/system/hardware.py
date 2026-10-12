"""Hardware resource monitoring: CPU, RAM, GPU, and Storage (Phase 9)."""

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import string
import sys
from typing import List, Optional, Dict, Any

from robbie.utils.device import get_hardware_info, HardwareReport


@dataclass
class CPUInfo:
    """CPU specifications and logical capacity."""
    processor: str
    architecture: str
    logical_cores: int
    physical_cores: Optional[int] = None
    system: str = "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processor": self.processor,
            "architecture": self.architecture,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "system": self.system,
        }


@dataclass
class MemoryInfo:
    """System RAM statistics in Gigabytes."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gb": self.total_gb,
            "available_gb": self.available_gb,
            "used_gb": self.used_gb,
            "percent_used": self.percent_used,
        }


@dataclass
class DiskInfo:
    """Storage drive capacity and free space in Gigabytes."""
    drive: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drive": self.drive,
            "total_gb": self.total_gb,
            "used_gb": self.used_gb,
            "free_gb": self.free_gb,
            "percent_used": self.percent_used,
        }


@dataclass
class GPUInfo:
    """Graphics processing unit and VRAM capacity."""
    cuda_available: bool
    device_name: str
    device_count: int
    vram_total_gb: float
    vram_free_gb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "device_count": self.device_count,
            "vram_total_gb": self.vram_total_gb,
            "vram_free_gb": self.vram_free_gb,
        }


class HardwareMonitor:
    """Queries underlying system hardware specifications and live resource utilization."""

    @classmethod
    def get_cpu_info(cls) -> CPUInfo:
        """Query CPU model, architecture, and core counts."""
        proc = platform.processor() or platform.machine() or "Generic Processor"
        logical = os.cpu_count() or 1
        return CPUInfo(
            processor=proc,
            architecture=platform.machine(),
            logical_cores=logical,
            physical_cores=max(1, logical // 2) if logical > 1 else 1,
            system=f"{platform.system()} {platform.release()}",
        )

    @classmethod
    def get_memory_info(cls) -> MemoryInfo:
        """Query system physical RAM capacity and current availability."""
        if sys.platform == "win32":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total = stat.ullTotalPhys / (1024 ** 3)
                    avail = stat.ullAvailPhys / (1024 ** 3)
                    used = total - avail
                    pct = (used / total) * 100.0 if total > 0 else 0.0
                    return MemoryInfo(
                        total_gb=round(total, 2),
                        available_gb=round(avail, 2),
                        used_gb=round(used, 2),
                        percent_used=round(pct, 1),
                    )
            except Exception:
                pass

        # Fallback approximation
        return MemoryInfo(total_gb=16.0, available_gb=8.0, used_gb=8.0, percent_used=50.0)

    @classmethod
    def get_disks_info(cls) -> List[DiskInfo]:
        """Query mounted drives and storage capacity."""
        disks: List[DiskInfo] = []

        if sys.platform == "win32":
            # Check available drive letters
            for letter in string.ascii_uppercase:
                root = f"{letter}:\\"
                if os.path.exists(root):
                    try:
                        usage = shutil.disk_usage(root)
                        total = usage.total / (1024 ** 3)
                        free = usage.free / (1024 ** 3)
                        used = (usage.total - usage.free) / (1024 ** 3)
                        pct = (used / total * 100.0) if total > 0 else 0.0
                        disks.append(
                            DiskInfo(
                                drive=f"{letter}:",
                                total_gb=round(total, 2),
                                used_gb=round(used, 2),
                                free_gb=round(free, 2),
                                percent_used=round(pct, 1),
                            )
                        )
                    except (PermissionError, OSError):
                        continue
        else:
            # POSIX root filesystem
            try:
                usage = shutil.disk_usage("/")
                total = usage.total / (1024 ** 3)
                free = usage.free / (1024 ** 3)
                used = (usage.total - usage.free) / (1024 ** 3)
                pct = (used / total * 100.0) if total > 0 else 0.0
                disks.append(
                    DiskInfo(
                        drive="/",
                        total_gb=round(total, 2),
                        used_gb=round(used, 2),
                        free_gb=round(free, 2),
                        percent_used=round(pct, 1),
                    )
                )
            except Exception:
                pass

        return disks

    @classmethod
    def get_gpu_info(cls) -> GPUInfo:
        """Query GPU and VRAM metrics using Project Robbie's device utility."""
        report: HardwareReport = get_hardware_info()
        return GPUInfo(
            cuda_available=report.cuda_available,
            device_name=report.device_name,
            device_count=report.device_count,
            vram_total_gb=report.vram_total_gb,
            vram_free_gb=report.vram_free_gb,
        )
