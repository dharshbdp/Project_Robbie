"""Process inspection and active application monitoring (Phase 9)."""

import csv
from dataclasses import dataclass
import io
import subprocess
import sys
from typing import List, Dict, Any, Optional

from robbie.utils.logger import logger


@dataclass
class ProcessInfo:
    """Information about an active running process."""
    name: str
    pid: int
    memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pid": self.pid,
            "memory_mb": self.memory_mb,
        }


class ProcessMonitor:
    """Queries running operating system processes without requiring third-party libraries."""

    @classmethod
    def list_processes(cls, limit: int = 15) -> List[ProcessInfo]:
        """List active processes ordered by memory utilization descending."""
        if sys.platform == "win32":
            return cls._list_windows_processes(limit=limit)
        else:
            return cls._list_posix_processes(limit=limit)

    @classmethod
    def find_process(cls, query: str, limit: int = 10) -> List[ProcessInfo]:
        """Search running processes by name substring."""
        all_procs = cls.list_processes(limit=100)
        q = query.lower()
        matched = [p for p in all_procs if q in p.name.lower()]
        return matched[:limit]

    @classmethod
    def _list_windows_processes(cls, limit: int = 15) -> List[ProcessInfo]:
        try:
            cmd = ["tasklist", "/FO", "CSV", "/NH"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            reader = csv.reader(io.StringIO(res.stdout))
            procs: List[ProcessInfo] = []

            for row in reader:
                if len(row) >= 5:
                    name = row[0].strip()
                    try:
                        pid = int(row[1].strip())
                    except ValueError:
                        pid = 0

                    mem_raw = row[4].replace(" K", "").replace(",", "").replace(".", "").strip()
                    try:
                        mem_kb = int(mem_raw)
                        mem_mb = round(mem_kb / 1024.0, 1)
                    except ValueError:
                        mem_mb = 0.0

                    procs.append(ProcessInfo(name=name, pid=pid, memory_mb=mem_mb))

            # Sort by memory descending
            procs.sort(key=lambda p: p.memory_mb, reverse=True)
            return procs[:limit]
        except Exception as e:
            logger.warning(f"Failed to query Windows processes via tasklist: {e}")
            return []

    @classmethod
    def _list_posix_processes(cls, limit: int = 15) -> List[ProcessInfo]:
        try:
            cmd = ["ps", "-eo", "pid,rss,comm", "--sort=-rss"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = res.stdout.strip().splitlines()[1:]  # Skip header
            procs: List[ProcessInfo] = []

            for line in lines:
                parts = line.strip().split(None, 2)
                if len(parts) >= 3:
                    try:
                        pid = int(parts[0])
                        rss_kb = int(parts[1])
                        name = parts[2]
                        procs.append(ProcessInfo(name=name, pid=pid, memory_mb=round(rss_kb / 1024.0, 1)))
                    except ValueError:
                        continue

            return procs[:limit]
        except Exception as e:
            logger.warning(f"Failed to query POSIX processes via ps: {e}")
            return []
