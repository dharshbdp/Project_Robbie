"""Computer Awareness, hardware monitoring, environment, and workspace inspection (Phase 9)."""

from robbie.system.hardware import (
    CPUInfo,
    MemoryInfo,
    DiskInfo,
    GPUInfo,
    HardwareMonitor,
)
from robbie.system.processes import (
    ProcessInfo,
    ProcessMonitor,
)
from robbie.system.environment import (
    BatteryInfo,
    NetworkInfo,
    WindowContext,
    EnvironmentMonitor,
)
from robbie.system.workspace import (
    WorkspaceContext,
    WorkspaceMonitor,
)
from robbie.system.inspector import (
    SystemSnapshot,
    ComputerAwareness,
)

__all__ = [
    "CPUInfo",
    "MemoryInfo",
    "DiskInfo",
    "GPUInfo",
    "HardwareMonitor",
    "ProcessInfo",
    "ProcessMonitor",
    "BatteryInfo",
    "NetworkInfo",
    "WindowContext",
    "EnvironmentMonitor",
    "WorkspaceContext",
    "WorkspaceMonitor",
    "SystemSnapshot",
    "ComputerAwareness",
]
