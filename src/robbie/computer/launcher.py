"""Safe desktop application launcher for Project Robbie (Phase 10/11).

Launches native Windows applications (Google Chrome, Photos, File Explorer,
Notepad, Task Manager, Settings, Calculator, etc.) via shell protocols and verified paths.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple, Any


@dataclass
class AppLaunchResult:
    """Outcome of an application launch attempt."""
    success: bool
    app_name: str
    target: str
    message: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "app_name": self.app_name,
            "target": self.target,
            "message": self.message,
            "error": self.error,
        }


class AppLauncher:
    """Intelligent Windows application launcher with alias mapping and URI protocol resolution."""

    # Curated mapping of common application names and Windows shell execution targets
    APP_REGISTRY: Dict[str, Tuple[str, str]] = {
        # Browsers
        "chrome": ("Google Chrome", "chrome.exe"),
        "google chrome": ("Google Chrome", "chrome.exe"),
        "edge": ("Microsoft Edge", "msedge.exe"),
        "microsoft edge": ("Microsoft Edge", "msedge.exe"),
        "firefox": ("Mozilla Firefox", "firefox.exe"),
        "brave": ("Brave Browser", "brave.exe"),

        # System & File Utilities
        "file manager": ("File Explorer", "explorer.exe"),
        "explorer": ("File Explorer", "explorer.exe"),
        "file explorer": ("File Explorer", "explorer.exe"),
        "files": ("File Explorer", "explorer.exe"),
        "my pc": ("File Explorer", "explorer.exe"),
        "this pc": ("File Explorer", "explorer.exe"),
        "task manager": ("Task Manager", "taskmgr.exe"),
        "taskmgr": ("Task Manager", "taskmgr.exe"),
        "settings": ("Windows Settings", "ms-settings:"),
        "windows settings": ("Windows Settings", "ms-settings:"),
        "control panel": ("Control Panel", "control.exe"),

        # Productivity & Media
        "photos": ("Photos", "ms-photos:"),
        "photo viewer": ("Photos", "ms-photos:"),
        "notepad": ("Notepad", "notepad.exe"),
        "calculator": ("Calculator", "calc.exe"),
        "calc": ("Calculator", "calc.exe"),
        "paint": ("Paint", "mspaint.exe"),
        "camera": ("Camera", "microsoft.windows.camera:"),
        "voice recorder": ("Voice Recorder", "ms-soundrecorder:"),
        "spotify": ("Spotify", "spotify:"),

        # Developer Tools
        "vs code": ("Visual Studio Code", "code.cmd"),
        "vscode": ("Visual Studio Code", "code.cmd"),
        "code": ("Visual Studio Code", "code.cmd"),
        "terminal": ("Windows Terminal", "wt.exe"),
        "powershell": ("PowerShell", "powershell.exe"),
        "cmd": ("Command Prompt", "cmd.exe"),
    }

    @classmethod
    def launch(cls, query: str) -> AppLaunchResult:
        """Resolve and safely launch an application by name or phrase.

        Examples:
            launch("Google Chrome")
            launch("photos")
            launch("file manager")
        """
        cleaned = query.strip().lower()
        if cleaned.startswith("open "):
            cleaned = cleaned[len("open "):].strip()
        if cleaned.startswith("launch "):
            cleaned = cleaned[len("launch "):].strip()
        if cleaned.startswith("start "):
            cleaned = cleaned[len("start "):].strip()

        if not cleaned:
            return AppLaunchResult(
                success=False,
                app_name="",
                target="",
                message="No application name specified.",
                error="Empty application query",
            )

        # 1. Check direct registry alias
        display_name, target = cls.APP_REGISTRY.get(cleaned, (cleaned.title(), cleaned))

        # 2. Partial matching in registry
        if target == cleaned:
            for alias, (disp, reg_target) in cls.APP_REGISTRY.items():
                if alias in cleaned or cleaned in alias:
                    display_name, target = disp, reg_target
                    break

        if sys.platform != "win32":
            # POSIX fallback
            try:
                subprocess.Popen([target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return AppLaunchResult(
                    success=True,
                    app_name=display_name,
                    target=target,
                    message=f"Launched application '{display_name}'.",
                )
            except Exception as e:
                return AppLaunchResult(
                    success=False,
                    app_name=display_name,
                    target=target,
                    message=f"Failed to launch '{display_name}': {e}",
                    error=str(e),
                )

        # Windows Launch Logic
        try:
            # URI protocol handler (e.g. ms-settings:, ms-photos:)
            if ":" in target and not Path(target).is_absolute() and not target.endswith(".exe"):
                os.startfile(target)
                return AppLaunchResult(
                    success=True,
                    app_name=display_name,
                    target=target,
                    message=f"Successfully opened {display_name}!",
                )

            # Executable file or command
            # Try startfile first
            try:
                os.startfile(target)
                return AppLaunchResult(
                    success=True,
                    app_name=display_name,
                    target=target,
                    message=f"Successfully launched {display_name}!",
                )
            except (FileNotFoundError, OSError):
                pass

            # Try locating via shutil.which
            which_path = shutil.which(target) or shutil.which(f"{target}.exe")
            if which_path:
                subprocess.Popen(
                    [which_path],
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                    close_fds=True,
                )
                return AppLaunchResult(
                    success=True,
                    app_name=display_name,
                    target=which_path,
                    message=f"Successfully launched {display_name}!",
                )

            # Fallback to shell start
            cmd = f'start "" "{target}"'
            subprocess.Popen(
                cmd,
                shell=True,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return AppLaunchResult(
                success=True,
                app_name=display_name,
                target=target,
                message=f"Sent start command for {display_name}!",
            )

        except Exception as e:
            return AppLaunchResult(
                success=False,
                app_name=display_name,
                target=target,
                message=f"Could not open {display_name}: {e}",
                error=str(e),
            )
