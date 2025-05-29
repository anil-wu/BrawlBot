"""
adb_control.py
~~~~~~~~~~~~~~
利用 `adb shell input …` 直接在设备上注入点击 / 滑动。
优点：无需依赖 scrcpy 控制端口；缺点：额外一次 adb 往返、时延略高。
"""

import subprocess
import shlex
from typing import Tuple, Optional


class AdbControl:
    """基于 adb 的输入注入。"""

    def __init__(self, serial: Optional[str] = None):
        """
        :param serial: 设备序列号；为空时使用 adb 默认设备。
        """
        self.serial = serial

    # —— 内部工具 ————————————————————
    def _adb(self, cmd: str) -> None:
        """
        执行 adb 子命令；若失败抛出 RuntimeError。
        """
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]
        # `shell` 后面的命令用一个字符串传入，避免双层列表
        full_cmd = base + shlex.split(cmd)
        res = subprocess.run(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"ADB cmd failed: {' '.join(full_cmd)}\n{res.stderr.strip()}"
            )

    # —— 对外 API ————————————————————
    def tap(self, x: int, y: int) -> None:
        """
        单击 (x, y)。
        """
        self._adb(f"shell input tap {x} {y}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """
        滑动：从 (x1, y1) → (x2, y2)，持续 duration_ms 毫秒。
        """
        self._adb(f"shell input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    # —— 其他高级操作可按需扩展 ——————
    # def key(self, keycode: int): ...
    # def long_press(self, x, y, hold_ms): ...
