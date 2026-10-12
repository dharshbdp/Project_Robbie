"""Comprehensive Unit Tests for Computer Awareness Subsystem (Phase 9)."""

import pytest
from unittest.mock import MagicMock, patch

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
from robbie.system.inspector import SystemSnapshot, ComputerAwareness
from robbie.core.conversation.engine import ChatEngine
from robbie.core.conversation.context import ConversationContext


def test_hardware_monitor_cpu():
    cpu = HardwareMonitor.get_cpu_info()
    assert isinstance(cpu, CPUInfo)
    assert cpu.logical_cores > 0
    assert len(cpu.system) > 0
    assert len(cpu.architecture) > 0
    cpu_dict = cpu.to_dict()
    assert "logical_cores" in cpu_dict
    assert "system" in cpu_dict


def test_hardware_monitor_memory():
    mem = HardwareMonitor.get_memory_info()
    assert isinstance(mem, MemoryInfo)
    assert mem.total_gb > 0
    assert mem.available_gb >= 0
    assert 0.0 <= mem.percent_used <= 100.0
    mem_dict = mem.to_dict()
    assert "total_gb" in mem_dict
    assert "available_gb" in mem_dict


def test_hardware_monitor_disks():
    disks = HardwareMonitor.get_disks_info()
    assert isinstance(disks, list)
    assert len(disks) > 0
    for d in disks:
        assert isinstance(d, DiskInfo)
        assert len(d.drive) > 0
        assert d.total_gb > 0


def test_hardware_monitor_gpu():
    gpu = HardwareMonitor.get_gpu_info()
    assert isinstance(gpu, GPUInfo)
    assert isinstance(gpu.cuda_available, bool)
    assert len(gpu.device_name) > 0
    gpu_dict = gpu.to_dict()
    assert "device_name" in gpu_dict
    assert "cuda_available" in gpu_dict


def test_process_monitor_listing():
    procs = ProcessMonitor.list_processes(limit=5)
    assert isinstance(procs, list)
    assert len(procs) <= 5
    if procs:
        assert isinstance(procs[0], ProcessInfo)
        assert procs[0].pid > 0
        assert len(procs[0].name) > 0
        assert procs[0].memory_mb >= 0.0
        # Verify descending sort by memory
        for i in range(len(procs) - 1):
            assert procs[i].memory_mb >= procs[i + 1].memory_mb


def test_process_monitor_find():
    procs = ProcessMonitor.find_process("python", limit=5)
    assert isinstance(procs, list)
    for p in procs:
        assert "python" in p.name.lower()


def test_environment_battery():
    battery = EnvironmentMonitor.get_battery_info()
    assert isinstance(battery, BatteryInfo)
    assert isinstance(battery.present, bool)
    assert 0 <= battery.percent <= 100
    assert battery.power_source in ("AC Power", "Battery", "Unknown")


def test_environment_network():
    net = EnvironmentMonitor.get_network_info()
    assert isinstance(net, NetworkInfo)
    assert len(net.hostname) > 0
    assert len(net.local_ip) > 0
    assert isinstance(net.internet_connected, bool)


def test_environment_active_window():
    window = EnvironmentMonitor.get_active_window()
    assert isinstance(window, WindowContext)
    assert len(window.title) > 0


def test_workspace_monitor():
    workspace = WorkspaceMonitor.get_workspace_context()
    assert isinstance(workspace, WorkspaceContext)
    assert workspace.project_name == "Project_Robbie"
    assert workspace.root_path.exists()
    assert isinstance(workspace.language_counts, dict)
    assert ".py" in workspace.language_counts


def test_computer_awareness_snapshot_and_markdown():
    ca = ComputerAwareness()
    snapshot = ca.snapshot(include_processes=True, proc_limit=3)
    assert isinstance(snapshot, SystemSnapshot)
    assert len(snapshot.timestamp) > 0

    md = snapshot.to_markdown()
    assert "### Computer Awareness Report" in md
    assert "Operating System:" in md
    assert "Hardware Resources" in md
    assert "Power & Network" in md
    assert "Development Workspace" in md

    summary = snapshot.quick_summary()
    assert "Host:" in summary
    assert "RAM:" in summary


def test_chat_engine_system_commands():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    engine = ChatEngine(model=mock_model, tokenizer=mock_tokenizer, device="cpu")

    # /system command
    sys_report = engine.handle_command("/system")
    assert "### Computer Awareness Report" in sys_report
    assert "Hardware Resources" in sys_report

    # /processes command
    proc_report = engine.handle_command("/processes 3")
    assert "Top 3 Processes" in proc_report
    assert "PID" in proc_report

    # /help documentation
    help_text = engine.handle_command("/help")
    assert "/system" in help_text
    assert "/processes" in help_text


def test_chat_engine_natural_language_system_queries():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    engine = ChatEngine(model=mock_model, tokenizer=mock_tokenizer, device="cpu")

    # Capability question
    resp_cap = engine.respond("Can you monitor my system?")
    assert "built-in computer awareness" in resp_cap

    # System specs question
    resp_specs = engine.respond("What are my computer specs?")
    assert "### Computer Awareness Report" in resp_specs

    # RAM query
    resp_ram = engine.respond("How much RAM is free?")
    assert "GB free" in resp_ram
    assert "RAM" in resp_ram

    # Battery query
    resp_bat = engine.respond("What is my battery level?")
    assert "battery" in resp_bat.lower() or "power" in resp_bat.lower()

    # GPU query
    resp_gpu = engine.respond("What GPU do I have?")
    assert "NVIDIA" in resp_gpu or "CPU" in resp_gpu

    # Active window query
    resp_win = engine.respond("What window is active?")
    assert "active foreground window" in resp_win
