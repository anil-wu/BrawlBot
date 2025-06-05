"""
adb_control.py
~~~~~~~~~~~~~~
纯 ADB 注入：tap / swipe / key / drag(持续按住再抬起)。
"""
from __future__ import annotations
import subprocess, shlex, time
from typing import Optional


class AdbControl:
    def __init__(self, serial: Optional[str] = None):
        self.serial = serial

    # ─────────── 内部工具 ───────────
    def _adb(self, cmd: str) -> None:
        base = ["adb"] + (["-s", self.serial] if self.serial else [])
        full = base + shlex.split(cmd)
        res = subprocess.run(full, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(f"ADB FAILED: {' '.join(full)}\n{res.stderr}")

    # ─────────── 基础注入 ───────────
    def tap(self, x: int, y: int):
        self._adb(f"shell input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, dur_ms: int = 300):
        self._adb(f"shell input swipe {x1} {y1} {x2} {y2} {dur_ms}")

    def drag(self, x: int, y: int, hold_ms: int = 100):
        """
        摇杆“按住不动”——DOWN→sleep→UP。
        """
        self._adb(f"shell sendevent /dev/input/event0 3 57 0")  # start touch
        self._adb(f"shell input swipe {x} {y} {x} {y} {hold_ms}")

    def key(self, keycode: int):
        self._adb(f"shell input keyevent {keycode}")
