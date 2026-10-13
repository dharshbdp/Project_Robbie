"""Safe workspace file operations with read-only defaults and confirmation enforcement (Phase 10)."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from robbie.computer.tools.policy import SafetyPolicy, SafetyViolation
from robbie.utils.logger import logger


@dataclass
class FileToolResult:
    """Outcome of a file tool execution."""
    success: bool
    action: str
    path: str
    content: Optional[str] = None
    entries: Optional[List[Dict[str, Any]]] = None
    bytes_affected: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "path": self.path,
            "content": self.content,
            "entries": self.entries,
            "entries_count": len(self.entries) if self.entries is not None else None,
            "bytes_affected": self.bytes_affected,
            "error": self.error,
        }


class FileOps:
    """Controlled file system operator enforcing safety boundaries and user confirmation."""

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy(read_only=True)

    def list_dir(self, path: Union[str, Path] = ".") -> FileToolResult:
        """List directory contents (Read-only action, default allowed)."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_read(target)
        if violation:
            return FileToolResult(
                success=False,
                action="list_dir",
                path=str(target),
                error=violation.reason,
            )

        if not target.exists():
            return FileToolResult(
                success=False,
                action="list_dir",
                path=str(target),
                error=f"Directory does not exist: '{target}'",
            )
        if not target.is_dir():
            return FileToolResult(
                success=False,
                action="list_dir",
                path=str(target),
                error=f"Path is not a directory: '{target}'",
            )

        try:
            entries = []
            for item in sorted(target.iterdir()):
                entries.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                })
            return FileToolResult(
                success=True,
                action="list_dir",
                path=str(target),
                entries=entries,
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="list_dir",
                path=str(target),
                error=str(e),
            )

    def read_file(self, path: Union[str, Path], max_chars: int = 100_000) -> FileToolResult:
        """Read text from a file (Read-only action, default allowed)."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_read(target)
        if violation:
            return FileToolResult(
                success=False,
                action="read_file",
                path=str(target),
                error=violation.reason,
            )

        if not target.exists() or not target.is_file():
            return FileToolResult(
                success=False,
                action="read_file",
                path=str(target),
                error=f"File not found or is a directory: '{target}'",
            )

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            return FileToolResult(
                success=True,
                action="read_file",
                path=str(target),
                content=content,
                bytes_affected=len(content.encode("utf-8")),
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="read_file",
                path=str(target),
                error=str(e),
            )

    def search_files(self, pattern: str, root_dir: Union[str, Path] = ".") -> FileToolResult:
        """Glob search for files matching a pattern within workspace."""
        root = Path(root_dir).resolve()
        violation = self.policy.validate_file_read(root)
        if violation:
            return FileToolResult(
                success=False,
                action="search_files",
                path=str(root),
                error=violation.reason,
            )

        try:
            matches = []
            for p in root.rglob(pattern):
                if not self.policy.is_sensitive_path(p):
                    matches.append({
                        "path": str(p),
                        "name": p.name,
                        "is_dir": p.is_dir(),
                    })
            return FileToolResult(
                success=True,
                action="search_files",
                path=str(root),
                entries=matches[:100],  # Bound results
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="search_files",
                path=str(root),
                error=str(e),
            )

    def write_file(
        self,
        path: Union[str, Path],
        content: str,
        confirm: bool = False,
    ) -> FileToolResult:
        """Write text to a file (Requires write permission and confirmation)."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_mutation(
            action="write_file",
            target_path=target,
            confirmed=confirm,
            details={"content_chars": len(content)},
        )
        if violation:
            return FileToolResult(
                success=False,
                action="write_file",
                path=str(target),
                error=violation.reason,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded = content.encode("utf-8")
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return FileToolResult(
                success=True,
                action="write_file",
                path=str(target),
                bytes_affected=len(encoded),
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="write_file",
                path=str(target),
                error=str(e),
            )

    def delete_file(
        self,
        path: Union[str, Path],
        confirm: bool = False,
    ) -> FileToolResult:
        """Delete a file (Requires write permission and confirmation)."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_mutation(
            action="delete_file",
            target_path=target,
            confirmed=confirm,
        )
        if violation:
            return FileToolResult(
                success=False,
                action="delete_file",
                path=str(target),
                error=violation.reason,
            )

        if not target.exists():
            return FileToolResult(
                success=False,
                action="delete_file",
                path=str(target),
                error=f"File does not exist: '{target}'",
            )

        try:
            size = target.stat().st_size if target.is_file() else 0
            if target.is_dir():
                target.rmdir()  # Only empty directory
            else:
                target.unlink()
            return FileToolResult(
                success=True,
                action="delete_file",
                path=str(target),
                bytes_affected=size,
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="delete_file",
                path=str(target),
                error=str(e),
            )

    def make_dir(
        self,
        path: Union[str, Path],
        confirm: bool = False,
    ) -> FileToolResult:
        """Create a new directory inside allowed workspace."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_mutation(
            action="make_dir",
            target_path=target,
            confirmed=confirm,
        )
        if violation:
            return FileToolResult(
                success=False,
                action="make_dir",
                path=str(target),
                error=violation.reason,
            )

        try:
            target.mkdir(parents=True, exist_ok=True)
            return FileToolResult(
                success=True,
                action="make_dir",
                path=str(target),
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="make_dir",
                path=str(target),
                error=str(e),
            )

    def open_in_system(self, path: Union[str, Path]) -> FileToolResult:
        """Open a file or directory in the system default application or File Explorer."""
        target = Path(path).resolve()
        violation = self.policy.validate_file_read(target)
        if violation:
            return FileToolResult(
                success=False,
                action="open_in_system",
                path=str(target),
                error=violation.reason,
            )

        if not target.exists():
            return FileToolResult(
                success=False,
                action="open_in_system",
                path=str(target),
                error=f"Path does not exist: '{target}'",
            )

        try:
            if sys.platform == "win32":
                os.startfile(str(target))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(target)])
            return FileToolResult(
                success=True,
                action="open_in_system",
                path=str(target),
            )
        except Exception as e:
            return FileToolResult(
                success=False,
                action="open_in_system",
                path=str(target),
                error=str(e),
            )

