"""Development workspace context and Git repository awareness (Phase 9)."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Optional, Dict, Any, List

from robbie.utils.logger import logger


@dataclass
class WorkspaceContext:
    """Current repository, git status, and workspace file structure."""
    root_path: Path
    project_name: str
    git_branch: Optional[str]
    git_commit: Optional[str]
    is_git_clean: bool
    modified_files_count: int
    untracked_files_count: int
    language_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_path": str(self.root_path),
            "project_name": self.project_name,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "is_git_clean": self.is_git_clean,
            "modified_files_count": self.modified_files_count,
            "untracked_files_count": self.untracked_files_count,
            "language_counts": self.language_counts,
        }


class WorkspaceMonitor:
    """Gathers context on the active development workspace and git repository."""

    @classmethod
    def get_workspace_context(cls, root_path: Optional[Path] = None) -> WorkspaceContext:
        """Analyze current project repository root, git state, and source files."""
        if root_path is None:
            # Default to Project_Robbie workspace directory
            root_path = Path.cwd()
        root_path = Path(root_path).resolve()

        project_name = root_path.name
        git_branch: Optional[str] = None
        git_commit: Optional[str] = None
        modified_count = 0
        untracked_count = 0
        is_clean = True

        # Check git state
        git_dir = root_path / ".git"
        if git_dir.exists():
            try:
                # 1. Branch
                res_b = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(root_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res_b.returncode == 0 and res_b.stdout.strip():
                    git_branch = res_b.stdout.strip()

                # 2. Commit hash
                res_c = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(root_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res_c.returncode == 0 and res_c.stdout.strip():
                    git_commit = res_c.stdout.strip()

                # 3. Status
                res_s = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(root_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res_s.returncode == 0:
                    status_lines = [l for l in res_s.stdout.splitlines() if l.strip()]
                    modified_count = sum(1 for l in status_lines if not l.startswith("??"))
                    untracked_count = sum(1 for l in status_lines if l.startswith("??"))
                    is_clean = len(status_lines) == 0
            except Exception as e:
                logger.debug(f"Failed to query git status: {e}")

        # Scan active source code languages in project (exclude .venv and data)
        ext_counter: Counter = Counter()
        scan_dirs = [root_path / "src", root_path / "tests"]
        for d in scan_dirs:
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file() and not p.name.startswith("."):
                        if "__pycache__" in p.parts or p.suffix.lower() in (".pyc", ".pyo"):
                            continue
                        ext = p.suffix.lower()
                        if ext:
                            ext_counter[ext] += 1

        language_counts = dict(ext_counter.most_common(6))

        return WorkspaceContext(
            root_path=root_path,
            project_name=project_name,
            git_branch=git_branch,
            git_commit=git_commit,
            is_git_clean=is_clean,
            modified_files_count=modified_count,
            untracked_files_count=untracked_count,
            language_counts=language_counts,
        )
