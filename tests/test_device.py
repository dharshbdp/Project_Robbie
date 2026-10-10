"""Tests for device detection and hardware reporting."""

from robbie.utils.device import get_hardware_info, get_best_device, HardwareReport

def test_hardware_info():
    report = get_hardware_info()
    assert isinstance(report, HardwareReport)
    assert report.device_type in ("cuda", "cpu")
    assert report.device_count >= 1
    assert isinstance(report.device_name, str)

def test_best_device():
    device = get_best_device()
    assert device in ("cuda", "cpu")
