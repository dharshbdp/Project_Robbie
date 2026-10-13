"""System performance diagnostics, bottleneck detection, and hardware troubleshooting (Phase 9/11).

Provides intelligent analysis for:
- Why CPU or GPU usage is high
- Why a game or application's FPS is dropping
- Comprehensive disk capacity and I/O pressure analysis
"""

from dataclasses import dataclass, field
import os
import platform
import shutil
import string
import sys
import time
from typing import List, Dict, Any, Optional

from robbie.system.hardware import HardwareMonitor, CPUInfo, MemoryInfo, GPUInfo, DiskInfo
from robbie.system.processes import ProcessMonitor, ProcessInfo
from robbie.system.environment import EnvironmentMonitor, WindowContext, BatteryInfo


@dataclass
class DiagnosticFinding:
    """A specific detected performance bottleneck or health issue."""
    category: str        # 'cpu', 'gpu', 'ram', 'disk', 'power', 'fps'
    severity: str        # 'info', 'warning', 'critical'
    title: str
    detail: str
    culprits: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class DiagnosticReport:
    """Comprehensive performance troubleshooting analysis."""
    headline: str
    primary_bottleneck: Optional[str]
    findings: List[DiagnosticFinding]
    summary_markdown: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "primary_bottleneck": self.primary_bottleneck,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "culprits": f.culprits,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
            "summary_markdown": self.summary_markdown,
            "metrics": self.metrics,
        }


class SystemDiagnostics:
    """Diagnostic reasoning engine analyzing hardware counters, processes, and environment."""

    @classmethod
    def diagnose_cpu(cls) -> DiagnosticReport:
        """Analyze why CPU usage is high and identify culprit processes."""
        cpu = HardwareMonitor.get_cpu_info()
        procs = ProcessMonitor.list_processes(limit=15)
        mem = HardwareMonitor.get_memory_info()

        findings: List[DiagnosticFinding] = []
        culprits: List[str] = []

        # Categorize top processes
        browser_procs = [p for p in procs if any(b in p.name.lower() for b in ("chrome", "msedge", "firefox", "brave", "opera"))]
        dev_procs = [p for p in procs if any(d in p.name.lower() for d in ("code", "antigravity", "pycharm", "python", "node", "rustc"))]
        sys_procs = [p for p in procs if any(s in p.name.lower() for s in ("msmpeng", "system", "searchindexer", "explorer"))]

        total_tracked_ram = sum(p.memory_mb for p in procs)

        if browser_procs:
            browser_ram = sum(p.memory_mb for p in browser_procs)
            culprits.append(f"Web Browser instances ({len(browser_procs)} processes, {browser_ram:.1f} MB RAM)")
            findings.append(DiagnosticFinding(
                category="cpu",
                severity="warning" if browser_ram > 1500 else "info",
                title="Active Web Browser Workload",
                detail=f"Browser processes ({len(browser_procs)} running) are consuming substantial background threads and {browser_ram:.1f} MB of RAM.",
                culprits=[p.name for p in browser_procs[:3]],
                recommendation="Close unused tabs, background video streams, or heavy web applications.",
            ))

        if dev_procs:
            dev_ram = sum(p.memory_mb for p in dev_procs)
            culprits.append(f"Development / IDE / Language runtimes ({dev_ram:.1f} MB RAM)")
            findings.append(DiagnosticFinding(
                category="cpu",
                severity="info",
                title="Development Environment Activity",
                detail=f"IDEs, language servers, or Python/Node runtimes are active ({dev_ram:.1f} MB RAM).",
                culprits=[p.name for p in dev_procs[:3]],
                recommendation="Pause active file watchers, indexers, or background training scripts if not in use.",
            ))

        if any("msmpeng" in p.name.lower() for p in procs):
            findings.append(DiagnosticFinding(
                category="cpu",
                severity="info",
                title="Windows Defender / Antivirus Scan",
                detail="MsMpEng.exe is currently active in memory, indicating real-time protection scanning or background file analysis.",
                culprits=["MsMpEng.exe"],
                recommendation="Wait a few moments for the antivirus scan to finish, or add heavy dev folders to Windows Defender exclusions.",
            ))

        # Overall summary
        top_3_names = ", ".join(f"{p.name} ({p.memory_mb:.0f}MB)" for p in procs[:3])
        headline = f"CPU Analysis: {cpu.processor} ({cpu.logical_cores} Cores)"
        summary_md = (
            f"### CPU Load Analysis\n\n"
            f"**Processor:** {cpu.processor} with **{cpu.logical_cores} logical cores**.\n"
            f"**Current Memory Pressure:** {mem.percent_used}% ({mem.used_gb:.1f} GB used / {mem.total_gb:.1f} GB total).\n\n"
            f"**Top 3 Resource Consumers:**\n"
            f"{top_3_names}\n\n"
            f"**Primary Observations:**\n"
        )
        for f in findings:
            summary_md += f"* **{f.title}**: {f.detail}\n  *Action:* {f.recommendation}\n"

        return DiagnosticReport(
            headline=headline,
            primary_bottleneck="CPU / Background Processes" if findings else "Normal",
            findings=findings,
            summary_markdown=summary_md,
            metrics={"cpu": cpu.to_dict(), "top_processes": [p.to_dict() for p in procs[:5]]},
        )

    @classmethod
    def diagnose_fps_drop(cls) -> DiagnosticReport:
        """Diagnose why game or rendering FPS is dropping."""
        gpu = HardwareMonitor.get_gpu_info()
        mem = HardwareMonitor.get_memory_info()
        env = EnvironmentMonitor.get_network_info()
        bat = EnvironmentMonitor.get_battery_info()
        win = EnvironmentMonitor.get_active_window()
        procs = ProcessMonitor.list_processes(limit=15)

        findings: List[DiagnosticFinding] = []
        primary_bottleneck = "None Detected"

        # 1. Power & Battery Check
        if bat.present and not bat.is_charging and bat.power_source != "AC Power":
            primary_bottleneck = "Battery Power Throttling"
            findings.append(DiagnosticFinding(
                category="power",
                severity="critical",
                title="Running on Battery (Power Plan Throttling)",
                detail=f"Laptop is operating on battery ({bat.percent}%). Laptops automatically throttle GPU clock speeds and power limits on battery power, cutting game FPS by 50% to 70%.",
                culprits=["Battery Power Mode"],
                recommendation="Plug your laptop into the AC wall adapter and switch the Windows Power Mode to 'Best Performance'.",
            ))

        # 2. VRAM Saturation Check
        if gpu.cuda_available:
            vram_used = gpu.vram_total_gb - gpu.vram_free_gb
            vram_pct = (vram_used / gpu.vram_total_gb) * 100.0 if gpu.vram_total_gb > 0 else 0
            if vram_pct > 85.0:
                primary_bottleneck = "GPU VRAM Bottleneck"
                findings.append(DiagnosticFinding(
                    category="gpu",
                    severity="critical",
                    title="GPU VRAM Saturation",
                    detail=f"GPU VRAM is {vram_pct:.1f}% full ({vram_used:.1f} / {gpu.vram_total_gb:.1f} GB allocated on {gpu.device_name}). When VRAM overflows, game textures swap into system RAM, causing massive micro-stutters and frame rate drops.",
                    culprits=[f"{gpu.device_name} ({vram_pct:.0f}% VRAM used)"],
                    recommendation="Lower game Texture Quality, Shadow Resolution, or disable Ray Tracing to fit inside VRAM.",
                ))
            else:
                findings.append(DiagnosticFinding(
                    category="gpu",
                    severity="info",
                    title="GPU VRAM Headroom OK",
                    detail=f"{gpu.device_name} has {gpu.vram_free_gb:.1f} GB free VRAM out of {gpu.vram_total_gb:.1f} GB total.",
                    culprits=[],
                    recommendation="GPU VRAM capacity is healthy.",
                ))
        else:
            primary_bottleneck = "No Dedicated GPU Detected"
            findings.append(DiagnosticFinding(
                category="gpu",
                severity="warning",
                title="Dedicated GPU / CUDA Not Detected",
                detail="System appears to be running on integrated graphics or CPU rendering, which provides low frame rates in 3D gaming.",
                culprits=["Integrated Graphics"],
                recommendation="Ensure your dedicated NVIDIA/AMD graphics card driver is installed and the game is configured to run on High Performance GPU.",
            ))

        # 3. System RAM Paging Pressure
        if mem.available_gb < 2.0 or mem.percent_used > 88.0:
            if primary_bottleneck == "None Detected":
                primary_bottleneck = "RAM Paging / Memory Exhaustion"
            findings.append(DiagnosticFinding(
                category="ram",
                severity="critical",
                title="High System RAM Pressure (Paging Lag)",
                detail=f"Only {mem.available_gb:.2f} GB of system RAM is free ({mem.percent_used}% utilized). When free RAM drops below 2 GB, Windows aggressively pages memory to the SSD/HDD, causing FPS drops and hitches during gameplay.",
                culprits=[p.name for p in procs[:4]],
                recommendation="Close background memory-heavy apps (browsers, Discord hardware acceleration, IDEs) before gaming.",
            ))

        # 4. Background CPU / Process Interference
        top_hogs = [p for p in procs[:5] if p.memory_mb > 500 and not any(g in p.name.lower() for g in ("game", "steam", "unreal", "unity"))]
        if top_hogs:
            findings.append(DiagnosticFinding(
                category="cpu",
                severity="warning",
                title="Heavy Background Applications Stealing CPU Cycles",
                detail=f"Non-game applications are consuming substantial system resources: {', '.join(p.name for p in top_hogs[:3])}.",
                culprits=[p.name for p in top_hogs],
                recommendation="Close background apps or disable hardware acceleration in Discord/Chrome.",
            ))

        summary_md = (
            f"### Game FPS Drop & Rendering Diagnostic\n\n"
            f"**Active Foreground Window:** `{win.title}`  \n"
            f"**Graphics Card:** {gpu.device_name if gpu.cuda_available else 'Integrated Graphics'}  \n"
            f"**Power Source:** {bat.power_source} ({bat.percent}% Battery)  \n"
            f"**RAM Availability:** {mem.available_gb:.1f} GB Free / {mem.total_gb:.1f} GB Total ({mem.percent_used}% Used)  \n\n"
            f"**Primary Culprit Identified:** **{primary_bottleneck}**\n\n"
            f"**Troubleshooting Checklist:**\n"
        )
        for f in findings:
            status_tag = "[CRITICAL]" if f.severity == "critical" else ("[WARNING]" if f.severity == "warning" else "[INFO]")
            summary_md += f"* {status_tag} **{f.title}**: {f.detail}\n  *Action:* **{f.recommendation}**\n\n"

        return DiagnosticReport(
            headline=f"FPS Diagnostic: {primary_bottleneck}",
            primary_bottleneck=primary_bottleneck,
            findings=findings,
            summary_markdown=summary_md,
            metrics={
                "gpu": gpu.to_dict(),
                "memory": mem.to_dict(),
                "power": bat.to_dict(),
                "active_window": win.title,
            },
        )

    @classmethod
    def diagnose_disk(cls) -> DiagnosticReport:
        """Analyze disk drive storage and identify near-capacity partitions."""
        disks = HardwareMonitor.get_disks_info()
        findings: List[DiagnosticFinding] = []
        primary_bottleneck = "Storage Healthy"

        for d in disks:
            if d.percent_used > 88.0:
                primary_bottleneck = f"Drive {d.drive} Nearly Full ({d.percent_used}%)"
                findings.append(DiagnosticFinding(
                    category="disk",
                    severity="critical" if d.percent_used > 93.0 else "warning",
                    title=f"Drive {d.drive} Low Free Space",
                    detail=f"Drive {d.drive} has only {d.free_gb:.1f} GB free out of {d.total_gb:.1f} GB ({d.percent_used}% utilized). Low disk space impedes Windows virtual memory swap files and slows down file operations.",
                    culprits=[f"Drive {d.drive}"],
                    recommendation=f"Run Windows Disk Cleanup on {d.drive}, empty Recycle Bin, or move large downloads/games to another drive.",
                ))
            else:
                findings.append(DiagnosticFinding(
                    category="disk",
                    severity="info",
                    title=f"Drive {d.drive} Space Healthy",
                    detail=f"Drive {d.drive} has {d.free_gb:.1f} GB free out of {d.total_gb:.1f} GB ({d.percent_used}% used).",
                    culprits=[],
                    recommendation="No action needed.",
                ))

        summary_md = (
            f"### Storage & Disk Drive Diagnosis\n\n"
            f"**Drives Inspected:** {len(disks)} mounted partitions.\n\n"
        )
        for d in disks:
            bar_len = int(d.percent_used / 5)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            summary_md += f"* **Drive {d.drive}**: `[{bar}]` **{d.percent_used}%** ({d.free_gb:.1f} GB free / {d.total_gb:.1f} GB total)\n"

        summary_md += f"\n**Analysis:** {primary_bottleneck}\n"

        return DiagnosticReport(
            headline=f"Disk Usage: {primary_bottleneck}",
            primary_bottleneck=primary_bottleneck,
            findings=findings,
            summary_markdown=summary_md,
            metrics={"disks": [d.to_dict() for d in disks]},
        )
