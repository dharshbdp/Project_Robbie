"""Repeatable evaluation harness for Computer Awareness and Tool Safety (Priority 4)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from robbie.system import ComputerAwareness
from robbie.computer.tools import (
    ComputerToolManager,
    SafetyPolicy,
    PermissionLevel,
)


@dataclass
class ComputerEvaluationResult:
    """Summary of computer awareness and safety evaluation metrics."""
    snapshot_passed: bool = True
    invariants_passed: bool = True
    processes_passed: bool = True
    safety_policy_passed: bool = True
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_passed": self.snapshot_passed,
            "invariants_passed": self.invariants_passed,
            "processes_passed": self.processes_passed,
            "safety_policy_passed": self.safety_policy_passed,
            "passed": self.passed,
            "details": self.details,
        }


class ComputerAwarenessEvaluator:
    """Evaluates system awareness accuracy, metric invariants, and tool safety policies."""

    @classmethod
    def evaluate_awareness_invariants(cls, awareness: Optional[ComputerAwareness] = None) -> Dict[str, Any]:
        """Verify that hardware, memory, power, and workspace metrics satisfy physical invariants."""
        ca = awareness or ComputerAwareness()
        snap = ca.snapshot()

        # Invariant checks:
        # 1. CPU logical cores > 0 and processor non-empty
        cpu_valid = (snap.cpu.logical_cores > 0) and bool(snap.cpu.processor)

        # 2. RAM percent must be in [0, 100], total > 0
        ram_valid = (0.0 <= snap.memory.percent_used <= 100.0) and (snap.memory.total_gb > 0)

        # 3. Storage free <= total and total > 0 across disks
        storage_valid = True
        for d in snap.disks:
            if not (0.0 <= d.percent_used <= 100.0 and d.total_gb > 0 and d.free_gb <= d.total_gb):
                storage_valid = False
                break

        # 4. Git repository detection (we are inside Project_Robbie git repo)
        git_detected = bool(snap.workspace.git_branch)

        all_invariants_pass = cpu_valid and ram_valid and storage_valid

        return {
            "cores": snap.cpu.logical_cores,
            "cpu_valid": cpu_valid,
            "ram_percent": snap.memory.percent_used,
            "ram_valid": ram_valid,
            "storage_valid": storage_valid,
            "git_detected": git_detected,
            "branch": snap.workspace.git_branch,
            "all_invariants_pass": all_invariants_pass,
        }

    @classmethod
    def evaluate_process_monitor(cls, awareness: Optional[ComputerAwareness] = None) -> Dict[str, Any]:
        """Verify that process listing is populated, non-empty, and sorted by memory usage."""
        ca = awareness or ComputerAwareness()
        procs = ca.process_monitor.list_processes(limit=5)

        is_populated = len(procs) > 0
        has_valid_pids = all(p.pid > 0 for p in procs)
        is_sorted = True
        for i in range(len(procs) - 1):
            if procs[i].memory_mb < procs[i + 1].memory_mb:
                is_sorted = False
                break

        return {
            "processes_count": len(procs),
            "is_populated": is_populated,
            "has_valid_pids": has_valid_pids,
            "is_sorted_by_memory": is_sorted,
            "top_process_name": procs[0].name if procs else None,
            "passed": is_populated and has_valid_pids and is_sorted,
        }

    @classmethod
    def evaluate_tool_safety(cls, manager: Optional[ComputerToolManager] = None) -> Dict[str, Any]:
        """Verify that read-only defaults, path sandboxing, and dangerous command protection hold."""
        mgr = manager or ComputerToolManager(read_only=True)

        # 1. Read-only default guard
        ro_write = mgr.write_file("test_mutation.txt", "data", confirm=True)
        ro_blocked = (ro_write.success is False) and ("read-only" in (ro_write.error or "").lower())

        # 2. Sensitive path sandbox
        mgr.set_read_only(False)
        mgr.grant_permission(PermissionLevel.WORKSPACE_WRITE)
        env_write = mgr.write_file(".env", "SECRET=123", confirm=True)
        env_blocked = (env_write.success is False) and ("sensitive" in (env_write.error or "").lower())

        # 3. Dangerous command blacklist
        mgr.grant_permission(PermissionLevel.COMMAND_EXECUTE)
        cmd_blocked = False
        res_cmd = mgr.run_command("rm -rf /", confirm=True)
        if not res_cmd.success and "blacklisted" in (res_cmd.error or "").lower():
            cmd_blocked = True

        passed = ro_blocked and env_blocked and cmd_blocked

        return {
            "read_only_default_enforced": ro_blocked,
            "sensitive_path_blocked": env_blocked,
            "dangerous_command_blocked": cmd_blocked,
            "passed": passed,
        }

    @classmethod
    def run_benchmark(
        cls,
        awareness: Optional[ComputerAwareness] = None,
        tool_manager: Optional[ComputerToolManager] = None,
    ) -> ComputerEvaluationResult:
        """Run complete repeatable evaluation of computer awareness and tool safety."""
        invariants = cls.evaluate_awareness_invariants(awareness)
        procs = cls.evaluate_process_monitor(awareness)
        safety = cls.evaluate_tool_safety(tool_manager)

        overall_passed = invariants["all_invariants_pass"] and procs["passed"] and safety["passed"]

        return ComputerEvaluationResult(
            snapshot_passed=invariants["all_invariants_pass"],
            invariants_passed=invariants["all_invariants_pass"],
            processes_passed=procs["passed"],
            safety_policy_passed=safety["passed"],
            passed=overall_passed,
            details={
                "invariants": invariants,
                "processes": procs,
                "safety": safety,
            },
        )
