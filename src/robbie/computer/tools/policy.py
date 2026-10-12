"""Safety policies, permission boundaries, confirmation callbacks, and command blacklisting (Phase 10)."""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
from typing import Callable, Optional, Set, List, Dict, Any, Union, Tuple


class PermissionLevel(str, Enum):
    """Graduated permission levels for computer tool access."""
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    COMMAND_EXECUTE = "command_execute"
    ADMIN = "admin"


@dataclass
class SafetyViolation:
    """Details of a blocked or rejected tool action."""
    action: str
    target: str
    reason: str
    is_blocked: bool = True


class SafetyPolicy:
    """Enforces read-only defaults, path sandboxing, dangerous command blacklisting,

    and mandatory user confirmations before modifying files or executing commands.
    """

    DANGEROUS_COMMAND_PATTERNS = [
        re.compile(r"\brm\s+-[rRfF]{1,3}\b", re.IGNORECASE),
        re.compile(r"\bdel\s+(?:/[a-zA-Z]\s+)*[cC]:\\", re.IGNORECASE),
        re.compile(r"\brmdir\s+/[sS]\s+/[qQ]\b", re.IGNORECASE),
        re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
        re.compile(r"\bmkfs\b", re.IGNORECASE),
        re.compile(r"\bdd\s+if=", re.IGNORECASE),
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),  # Fork bomb
        re.compile(r"\bshutdown(?:\.exe)?\s+/[sSrR]\b", re.IGNORECASE),
        re.compile(r"\bpoweroff\b", re.IGNORECASE),
        re.compile(r"\bnet\s+user\s+", re.IGNORECASE),
        re.compile(r"\bvssadmin\s+delete\b", re.IGNORECASE),
        re.compile(r"\breg\s+delete\b", re.IGNORECASE),
    ]

    SENSITIVE_PATH_PATTERNS = [
        re.compile(r"(?:^|[/\\])\.ssh(?:[/\\]|$)", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])\.aws(?:[/\\]|$)", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])\.gnupg(?:[/\\]|$)", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])id_[a-zA-Z0-9_]+(?:\.pub)?$", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])\.env(?:$|\.[a-zA-Z0-9_]+)", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])credentials(?:\.json)?$", re.IGNORECASE),
        re.compile(r"(?:^|[/\\])secrets\.json$", re.IGNORECASE),
        re.compile(r"^[a-zA-Z]:[/\\]Windows(?:[/\\]|$)", re.IGNORECASE),
        re.compile(r"^[a-zA-Z]:[/\\]Program Files(?: \(x86\))?(?:[/\\]|$)", re.IGNORECASE),
        re.compile(r"^[a-zA-Z]:[/\\]System Volume Information(?:[/\\]|$)", re.IGNORECASE),
    ]

    def __init__(
        self,
        read_only: bool = True,
        allowed_roots: Optional[List[Union[str, Path]]] = None,
        confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.read_only = read_only
        self.allowed_roots = [Path(r).resolve() for r in (allowed_roots or [Path.cwd()])]
        self.granted_permissions: Set[PermissionLevel] = {PermissionLevel.READ_ONLY}
        if not read_only:
            self.granted_permissions.add(PermissionLevel.WORKSPACE_WRITE)
            self.granted_permissions.add(PermissionLevel.COMMAND_EXECUTE)
        self.confirm_callback = confirm_callback

    def grant(self, level: PermissionLevel) -> None:
        """Explicitly grant a permission level."""
        self.granted_permissions.add(level)
        if level in (PermissionLevel.WORKSPACE_WRITE, PermissionLevel.COMMAND_EXECUTE, PermissionLevel.ADMIN):
            self.read_only = False

    def revoke(self, level: PermissionLevel) -> None:
        """Revoke a permission level."""
        self.granted_permissions.discard(level)
        if not any(lvl in self.granted_permissions for lvl in (PermissionLevel.WORKSPACE_WRITE, PermissionLevel.COMMAND_EXECUTE, PermissionLevel.ADMIN)):
            self.read_only = True

    def is_sensitive_path(self, target_path: Union[str, Path]) -> bool:
        """Check if target file or folder is a sensitive system or credential location."""
        path_str = str(Path(target_path).resolve())
        return any(pat.search(path_str) for pat in self.SENSITIVE_PATH_PATTERNS)

    def is_within_allowed_roots(self, target_path: Union[str, Path]) -> bool:
        """Ensure path resolves to a permitted workspace directory root."""
        resolved = Path(target_path).resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def is_dangerous_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """Verify if a shell command matches catastrophic or system-destructive patterns."""
        cmd_strip = command.strip()
        for pat in self.DANGEROUS_COMMAND_PATTERNS:
            if pat.search(cmd_strip):
                return True, pat.pattern
        return False, None

    def validate_file_read(self, target_path: Union[str, Path]) -> Optional[SafetyViolation]:
        """Validate read request."""
        if self.is_sensitive_path(target_path):
            return SafetyViolation(
                action="read",
                target=str(target_path),
                reason="Access to sensitive system or credential path is blocked.",
            )
        return None

    def validate_file_mutation(
        self,
        action: str,
        target_path: Union[str, Path],
        confirmed: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SafetyViolation]:
        """Validate write, modify, or delete request with read-only and confirmation checks."""
        # 1. Read-only default guard
        if self.read_only:
            return SafetyViolation(
                action=action,
                target=str(target_path),
                reason="System operates in read-only mode by default. Modification rejected.",
            )

        # 2. Permission level guard
        if PermissionLevel.WORKSPACE_WRITE not in self.granted_permissions and PermissionLevel.ADMIN not in self.granted_permissions:
            return SafetyViolation(
                action=action,
                target=str(target_path),
                reason="WORKSPACE_WRITE permission has not been explicitly granted.",
            )

        # 3. Sensitive path guard
        if self.is_sensitive_path(target_path):
            return SafetyViolation(
                action=action,
                target=str(target_path),
                reason="Modifications to sensitive system or credential files are prohibited.",
            )

        # 4. Workspace containment guard
        if not self.is_within_allowed_roots(target_path):
            return SafetyViolation(
                action=action,
                target=str(target_path),
                reason=f"Target path lies outside allowed workspace roots: {self.allowed_roots}",
            )

        # 5. User confirmation guard
        if not confirmed:
            if self.confirm_callback is not None:
                prompt_msg = f"Confirm {action.upper()} on '{target_path}'?"
                if not self.confirm_callback(prompt_msg, details or {}):
                    return SafetyViolation(
                        action=action,
                        target=str(target_path),
                        reason="Action cancelled by user during confirmation.",
                    )
            else:
                return SafetyViolation(
                    action=action,
                    target=str(target_path),
                    reason="Explicit confirmation required before modifying files.",
                )

        return None

    def validate_command_execution(
        self,
        command: str,
        confirmed: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SafetyViolation]:
        """Validate command execution request."""
        # 1. Read-only guard
        if self.read_only:
            return SafetyViolation(
                action="execute",
                target=command,
                reason="System operates in read-only mode by default. Command execution rejected.",
            )

        # 2. Permission level guard
        if PermissionLevel.COMMAND_EXECUTE not in self.granted_permissions and PermissionLevel.ADMIN not in self.granted_permissions:
            return SafetyViolation(
                action="execute",
                target=command,
                reason="COMMAND_EXECUTE permission has not been explicitly granted.",
            )

        # 3. Dangerous command blacklist
        is_dangerous, pat = self.is_dangerous_command(command)
        if is_dangerous:
            return SafetyViolation(
                action="execute",
                target=command,
                reason=f"Dangerous command pattern blacklisted: '{pat}'.",
            )

        # 4. User confirmation guard
        if not confirmed:
            if self.confirm_callback is not None:
                prompt_msg = f"Confirm shell execution: '{command}'?"
                if not self.confirm_callback(prompt_msg, details or {}):
                    return SafetyViolation(
                        action="execute",
                        target=command,
                        reason="Command execution cancelled by user during confirmation.",
                    )
            else:
                return SafetyViolation(
                    action="execute",
                    target=command,
                    reason="Explicit confirmation required before running shell commands.",
                )

        return None
