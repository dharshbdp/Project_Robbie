"""SystemSnapshot aggregator and ComputerAwareness orchestrator (Phase 9)."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

from robbie.system.hardware import (
    CPUInfo,
    MemoryInfo,
    DiskInfo,
    GPUInfo,
    HardwareMonitor,
)
from robbie.system.processes import ProcessInfo, ProcessMonitor
from robbie.system.environment import (
    BatteryInfo,
    NetworkInfo,
    WindowContext,
    EnvironmentMonitor,
)
from robbie.system.workspace import WorkspaceContext, WorkspaceMonitor


@dataclass
class SystemSnapshot:
    """Comprehensive snapshot of hardware, system resources, and development context."""
    cpu: CPUInfo
    memory: MemoryInfo
    disks: List[DiskInfo]
    gpu: GPUInfo
    battery: BatteryInfo
    network: NetworkInfo
    window: WindowContext
    workspace: WorkspaceContext
    processes: List[ProcessInfo]
    timestamp: str

    def to_markdown(self) -> str:
        """Format complete system snapshot into clean, structured Markdown."""
        lines = [
            f"### Computer Awareness Report ({self.timestamp})\n",
            f"**Operating System:** {self.cpu.system} ({self.cpu.architecture})  ",
            f"**Active Window:** `{self.window.title}`\n",
            "#### Hardware Resources",
            f"* **CPU:** {self.cpu.processor} ({self.cpu.logical_cores} logical cores)",
            f"* **RAM:** {self.memory.available_gb} GB free / {self.memory.total_gb} GB total ({self.memory.percent_used}% used)",
        ]

        if self.gpu.cuda_available:
            lines.append(f"* **GPU:** {self.gpu.device_name} ({self.gpu.vram_free_gb:.1f} GB free / {self.gpu.vram_total_gb:.1f} GB VRAM)")
        else:
            lines.append("* **GPU:** CUDA not detected (CPU inference mode)")

        # Disks
        disk_items = [f"{d.drive} ({d.free_gb} GB free / {d.total_gb} GB)" for d in self.disks]
        lines.append(f"* **Storage:** {', '.join(disk_items)}\n")

        # Environment
        lines.append("#### Power & Network")
        if self.battery.present:
            charging_tag = "Charging" if self.battery.is_charging else "Discharging"
            lines.append(f"* **Battery:** {self.battery.percent}% ({self.battery.power_source}, {charging_tag})")
        else:
            lines.append(f"* **Power:** {self.battery.power_source}")

        net_status = "Online" if self.network.internet_connected else "Offline"
        lat_tag = f" ({self.network.latency_ms}ms)" if self.network.latency_ms else ""
        lines.append(f"* **Network:** {net_status}{lat_tag} | Hostname: `{self.network.hostname}` | Local IP: `{self.network.local_ip}`\n")

        # Workspace
        lines.append("#### Development Workspace")
        git_clean_tag = "clean" if self.workspace.is_git_clean else f"{self.workspace.modified_files_count} modified"
        branch_tag = f"`{self.workspace.git_branch}` ({self.workspace.git_commit or 'unknown'}, {git_clean_tag})" if self.workspace.git_branch else "not a git repo"
        lines.append(f"* **Project:** `{self.workspace.project_name}` at `{self.workspace.root_path}`")
        lines.append(f"* **Git State:** {branch_tag}")

        lang_strs = [f"{ext}: {count} files" for ext, count in self.workspace.language_counts.items()]
        if lang_strs:
            lines.append(f"* **Codebase:** {', '.join(lang_strs)}\n")

        # Processes
        if self.processes:
            lines.append("#### Top Active Processes")
            for p in self.processes[:5]:
                lines.append(f"* `{p.name}` (PID {p.pid}): **{p.memory_mb:.1f} MB** RAM")

        return "\n".join(lines)

    def quick_summary(self) -> str:
        """One-line concise computer status."""
        gpu_str = f"GPU: {self.gpu.device_name}" if self.gpu.cuda_available else "GPU: CPU-only"
        battery_str = f" | Bat: {self.battery.percent}% ({self.battery.power_source})" if self.battery.present else ""
        git_str = f" | Git: {self.workspace.git_branch or 'n/a'}"
        return (
            f"Host: {self.network.hostname} | "
            f"RAM: {self.memory.available_gb}GB free / {self.memory.total_gb}GB | "
            f"{gpu_str}{battery_str}{git_str}"
        )


class ComputerAwareness:
    """Entry point for full computer and workspace environmental awareness."""

    def __init__(self):
        self.hardware = HardwareMonitor()
        self.process_monitor = ProcessMonitor()
        self.env_monitor = EnvironmentMonitor()
        self.workspace_monitor = WorkspaceMonitor()

    def snapshot(self, include_processes: bool = True, proc_limit: int = 5) -> SystemSnapshot:
        """Capture comprehensive live snapshot of the computer and workspace."""
        cpu = self.hardware.get_cpu_info()
        memory = self.hardware.get_memory_info()
        disks = self.hardware.get_disks_info()
        gpu = self.hardware.get_gpu_info()
        battery = self.env_monitor.get_battery_info()
        network = self.env_monitor.get_network_info()
        window = self.env_monitor.get_active_window()
        workspace = self.workspace_monitor.get_workspace_context()
        procs = self.process_monitor.list_processes(limit=proc_limit) if include_processes else []

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return SystemSnapshot(
            cpu=cpu,
            memory=memory,
            disks=disks,
            gpu=gpu,
            battery=battery,
            network=network,
            window=window,
            workspace=workspace,
            processes=procs,
            timestamp=now_str,
        )

    def quick_summary(self) -> str:
        """Generate compact one-line status string."""
        return self.snapshot(include_processes=False).quick_summary()
