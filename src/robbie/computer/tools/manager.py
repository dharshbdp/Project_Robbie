"""Unified manager for safe computer tools with audit logging and permission lifecycle (Phase 10)."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, List, Optional, Union, Dict, Any

from robbie.computer.tools.policy import SafetyPolicy, PermissionLevel, SafetyViolation
from robbie.computer.tools.file_ops import FileOps, FileToolResult
from robbie.computer.tools.command_runner import CommandRunner, CommandToolResult


class ComputerToolManager:
    """Central gateway for executing computer and workspace tools under strict safety policies."""

    def __init__(
        self,
        read_only: bool = True,
        allowed_roots: Optional[List[Union[str, Path]]] = None,
        confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.policy = SafetyPolicy(
            read_only=read_only,
            allowed_roots=allowed_roots,
            confirm_callback=confirm_callback,
        )
        self.file_ops = FileOps(policy=self.policy)
        self.command_runner = CommandRunner(policy=self.policy)
        self.audit_log: List[Dict[str, Any]] = []

    def set_read_only(self, enabled: bool) -> None:
        """Switch safety mode between read-only and permitted modification."""
        self.policy.read_only = enabled
        if enabled:
            self.policy.revoke(PermissionLevel.WORKSPACE_WRITE)
            self.policy.revoke(PermissionLevel.COMMAND_EXECUTE)
        else:
            self.policy.grant(PermissionLevel.WORKSPACE_WRITE)
            self.policy.grant(PermissionLevel.COMMAND_EXECUTE)

    def grant_permission(self, level: PermissionLevel) -> None:
        """Grant a specific permission level."""
        self.policy.grant(level)

    def revoke_permission(self, level: PermissionLevel) -> None:
        """Revoke a specific permission level."""
        self.policy.revoke(level)

    def set_confirm_callback(self, callback: Optional[Callable[[str, Dict[str, Any]], bool]]) -> None:
        """Set interactive confirmation prompt hook."""
        self.policy.confirm_callback = callback

    def _record_audit(self, action: str, target: str, success: bool, error: Optional[str] = None) -> None:
        self.audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "target": target,
            "success": success,
            "error": error,
            "read_only": self.policy.read_only,
        })
        if len(self.audit_log) > 200:
            self.audit_log = self.audit_log[-200:]

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    def list_dir(self, path: Union[str, Path] = ".") -> FileToolResult:
        res = self.file_ops.list_dir(path)
        self._record_audit("list_dir", str(path), res.success, res.error)
        return res

    def read_file(self, path: Union[str, Path], max_chars: int = 100_000) -> FileToolResult:
        res = self.file_ops.read_file(path, max_chars=max_chars)
        self._record_audit("read_file", str(path), res.success, res.error)
        return res

    def search_files(self, pattern: str, root_dir: Union[str, Path] = ".") -> FileToolResult:
        res = self.file_ops.search_files(pattern, root_dir=root_dir)
        self._record_audit("search_files", f"{pattern} in {root_dir}", res.success, res.error)
        return res

    def write_file(self, path: Union[str, Path], content: str, confirm: bool = False) -> FileToolResult:
        res = self.file_ops.write_file(path, content=content, confirm=confirm)
        self._record_audit("write_file", str(path), res.success, res.error)
        return res

    def delete_file(self, path: Union[str, Path], confirm: bool = False) -> FileToolResult:
        res = self.file_ops.delete_file(path, confirm=confirm)
        self._record_audit("delete_file", str(path), res.success, res.error)
        return res

    def make_dir(self, path: Union[str, Path], confirm: bool = False) -> FileToolResult:
        res = self.file_ops.make_dir(path, confirm=confirm)
        self._record_audit("make_dir", str(path), res.success, res.error)
        return res

    # -------------------------------------------------------------------------
    # Command Execution
    # -------------------------------------------------------------------------

    def run_command(
        self,
        command: str,
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = 10.0,
        confirm: bool = False,
    ) -> CommandToolResult:
        res = self.command_runner.execute(command, cwd=cwd, timeout=timeout, confirm=confirm)
        self._record_audit("run_command", command, res.success, res.error)
        return res

    # -------------------------------------------------------------------------
    # Audit & Status
    # -------------------------------------------------------------------------

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return list(self.audit_log)

    def clear_audit_log(self) -> None:
        self.audit_log.clear()
