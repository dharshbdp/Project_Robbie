"""Controlled shell command execution with blacklist checks, timeouts, and confirmations (Phase 10)."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Optional, Union, Dict, Any

from robbie.computer.tools.policy import SafetyPolicy, SafetyViolation
from robbie.utils.logger import logger


@dataclass
class CommandToolResult:
    """Outcome of a shell command execution."""
    success: bool
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:1000] if self.stdout else "",
            "stderr": self.stderr[:1000] if self.stderr else "",
            "duration_sec": round(self.duration_sec, 3),
            "error": self.error,
        }


class CommandRunner:
    """Safe command execution engine governing shell calls with strict safety policies."""

    MAX_OUTPUT_CHARS = 50_000

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy(read_only=True)

    def execute(
        self,
        command: str,
        cwd: Optional[Union[str, Path]] = None,
        timeout: float = 10.0,
        confirm: bool = False,
    ) -> CommandToolResult:
        """Execute a shell command within timeout and bounds after passing safety validation."""
        start_time = time.perf_counter()
        clean_cmd = command.strip()

        # 1. Safety policy validation
        violation = self.policy.validate_command_execution(
            command=clean_cmd,
            confirmed=confirm,
            details={"cwd": str(cwd) if cwd else None, "timeout": timeout},
        )
        if violation:
            return CommandToolResult(
                success=False,
                command=clean_cmd,
                exit_code=-1,
                stdout="",
                stderr="",
                duration_sec=0.0,
                error=violation.reason,
            )

        # 2. Determine working directory
        work_dir = Path(cwd).resolve() if cwd else Path.cwd()
        if not work_dir.exists():
            return CommandToolResult(
                success=False,
                command=clean_cmd,
                exit_code=-1,
                stdout="",
                stderr="",
                duration_sec=0.0,
                error=f"Working directory does not exist: '{work_dir}'",
            )

        # 3. Controlled execution
        try:
            logger.info(f"Executing controlled command: '{clean_cmd}' in {work_dir}")
            proc = subprocess.run(
                clean_cmd,
                shell=True,
                cwd=str(work_dir),
                timeout=timeout,
                capture_output=True,
                text=True,
                errors="replace",
            )
            duration = time.perf_counter() - start_time
            stdout = proc.stdout[:self.MAX_OUTPUT_CHARS]
            stderr = proc.stderr[:self.MAX_OUTPUT_CHARS]

            return CommandToolResult(
                success=(proc.returncode == 0),
                command=clean_cmd,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_sec=duration,
                error=None if proc.returncode == 0 else f"Process exited with code {proc.returncode}",
            )
        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            return CommandToolResult(
                success=False,
                command=clean_cmd,
                exit_code=-1,
                stdout="",
                stderr="",
                duration_sec=duration,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            return CommandToolResult(
                success=False,
                command=clean_cmd,
                exit_code=-1,
                stdout="",
                stderr="",
                duration_sec=duration,
                error=str(e),
            )
