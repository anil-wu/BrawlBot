from __future__ import annotations
import shlex, subprocess, time
from pathlib import Path
from typing import Optional

class ScrcpyLauncher:
    def __init__(
        self,
        *,
        serial: Optional[str] = None,
        video_port: int = 27183,
        server_jar: str | Path = "scrcpy-server.jar",
        server_version: str = "3.2",
        server_opts: str = (
            "tunnel_forward=true audio=false control=false "
            "cleanup=false raw_stream=true max_size=1080 power_on=true"
        ),
        unlock_screen: bool = True,
    ):
        self.serial = serial
        self.video_port = video_port
        self.server_jar = Path(server_jar)
        self.server_version = server_version
        self.server_opts = server_opts
        self.unlock_screen = unlock_screen

        self._adb_base = ["adb"] + (["-s", self.serial] if self.serial else [])
        self._server_proc: subprocess.Popen | None = None

    # ———————————— 外部接口 ————————————
    def launch(self) -> None:
        self._push_jar()
        self._start_server()
        self._wake_and_unlock()
        self._setup_forward()
        print("[Launcher] 环境启动完成 (forward 模式) ✔")

    def stop(self) -> None:
        self._remove_forward()
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
        print("[Launcher] 环境已停止 ✔")

    # 支持 with 语法
    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ———————————— 步骤细节 ————————————
    def _push_jar(self):
        print("[Launcher] 正在推送 scrcpy-server.jar …")
        self._adb(f"push {self.server_jar} /data/local/tmp/")

    def _start_server(self):
        print("[Launcher] 启动 scrcpy-server …")
        cmd = (
            f"CLASSPATH=/data/local/tmp/{self.server_jar.name} "
            "app_process / com.genymobile.scrcpy.Server "
            f"{self.server_version} {self.server_opts}"
        )
        self._server_proc = subprocess.Popen(
            self._adb_base + ["shell", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)

    def _wake_and_unlock(self):
        print("[Launcher] 点亮屏幕 …")
        self._adb("shell input keyevent 224")           # WAKEUP
        if self.unlock_screen:
            self._adb("shell input swipe 540 1800 540 400 200")

    def _setup_forward(self):
        """tcp:video_port  →  localabstract:scrcpy"""
        self._adb(f"forward tcp:{self.video_port} localabstract:scrcpy")
        print(f"[Launcher] adb forward tcp:{self.video_port} ✔")

    def _remove_forward(self):
        self._adb(f"--remove forward tcp:{self.video_port}", hide_err=True)

    # ———————————— ADB 工具 ————————————
    def _adb(self, sub_cmd: str, *, hide_err: bool = False):
        full = self._adb_base + shlex.split(sub_cmd)
        subprocess.run(
            full,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL if hide_err else None,
        )
