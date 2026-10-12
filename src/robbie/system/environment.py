"""Battery, network, and active window environment context monitoring (Phase 9)."""

import ctypes
from dataclasses import dataclass
import socket
import sys
import time
from typing import Optional, Dict, Any

from robbie.utils.logger import logger


@dataclass
class BatteryInfo:
    """System battery charge and power source state."""
    present: bool
    percent: int
    is_charging: bool
    power_source: str  # "AC Power" or "Battery"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "present": self.present,
            "percent": self.percent,
            "is_charging": self.is_charging,
            "power_source": self.power_source,
        }


@dataclass
class NetworkInfo:
    """Network connection, hostname, and online availability."""
    hostname: str
    local_ip: str
    internet_connected: bool
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "local_ip": self.local_ip,
            "internet_connected": self.internet_connected,
            "latency_ms": self.latency_ms,
        }


@dataclass
class WindowContext:
    """Currently focused application window."""
    title: str

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title}


class EnvironmentMonitor:
    """Monitors battery power, network state, and focused application window."""

    @classmethod
    def get_battery_info(cls) -> BatteryInfo:
        """Query system power and battery percentage."""
        if sys.platform == "win32":
            try:
                class SYSTEM_POWER_STATUS(ctypes.Structure):
                    _fields_ = [
                        ("ACLineStatus", ctypes.c_byte),
                        ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte),
                        ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong),
                        ("BatteryFullLifeTime", ctypes.c_ulong),
                    ]

                stat = SYSTEM_POWER_STATUS()
                if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(stat)):
                    # BatteryFlag 128 = No system battery (desktop)
                    present = stat.BatteryFlag != 128 and stat.BatteryLifePercent != 255
                    pct = int(stat.BatteryLifePercent) if stat.BatteryLifePercent != 255 else 100
                    # ACLineStatus: 1 = Online (AC), 0 = Offline (Battery)
                    is_ac = stat.ACLineStatus == 1
                    is_charging = bool(stat.BatteryFlag & 8) or is_ac
                    power_source = "AC Power" if is_ac else "Battery"

                    return BatteryInfo(
                        present=present,
                        percent=pct if present else 100,
                        is_charging=is_charging,
                        power_source=power_source,
                    )
            except Exception as e:
                logger.debug(f"Failed to query Windows battery status: {e}")

        # Desktop or unsupported fallback
        return BatteryInfo(
            present=False,
            percent=100,
            is_charging=True,
            power_source="AC Power",
        )

    @classmethod
    def get_network_info(cls, check_internet: bool = True) -> NetworkInfo:
        """Query hostname, IP addresses, and verify internet connectivity."""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            hostname = "localhost"
            local_ip = "127.0.0.1"

        internet_connected = False
        latency_ms: Optional[float] = None

        if check_internet:
            try:
                t0 = time.perf_counter()
                # Fast socket connection test to Cloudflare DNS (1.1.1.1:53)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.2)
                sock.connect(("1.1.1.1", 53))
                sock.close()
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                internet_connected = True
            except Exception:
                internet_connected = False

        return NetworkInfo(
            hostname=hostname,
            local_ip=local_ip,
            internet_connected=internet_connected,
            latency_ms=latency_ms,
        )

    @classmethod
    def get_active_window(cls) -> WindowContext:
        """Query title of the foreground application window."""
        if sys.platform == "win32":
            try:
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                if hwnd:
                    buf = ctypes.create_unicode_buffer(512)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
                    title = buf.value.strip()
                    if title:
                        return WindowContext(title=title)
            except Exception as e:
                logger.debug(f"Failed to query active foreground window: {e}")

        return WindowContext(title="Active Terminal / Workspace")
