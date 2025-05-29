from __future__ import annotations
import socket
import struct
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional
from scrcpy_video import VideoDecoder
from adb_control       import AdbControl
from scrcpy_control import ControlSocket
import av

CTRL_INJECT_TOUCH = 2
SCRCPY_HEADER_FMT = ">BIIII64s"

class ScrcpyClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 1234,
        control_port: int | None = 27184,
        *,
        serial: str | None = None,
        tunnel_forward: bool = True,
        start_server: bool = True,
        server_jar: str | Path = "scrcpy-server-v3.2.jar",
        server_version: str = "3.2",
        server_opts: str = "tunnel_forward=true audio=false control=false cleanup=false raw_stream=true max_size=1920 power_on=true control=false",
        on_frame: Optional[Callable] = None,
        resize_to: Optional[tuple[int, int]] = tuple[864, 1920],
    ) -> None:
        self.host = host
        self.port = port
        self.control_port = control_port
        self.serial = serial
        self.tunnel_forward = tunnel_forward
        self.start_server = start_server
        self.server_jar = Path(server_jar)
        self.server_version = server_version
        self.server_opts = server_opts
        self.on_frame = on_frame
        self.resize_to = resize_to

        self._video_sock: socket.socket | None = None
        self._adb_control: AdbControl | None = None
        self._container: av.container.InputContainer | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._server_proc: subprocess.Popen | None = None

    # ────────────────────────── 公共 API ──────────────────────────

    def frame(self)-> np.ndarray:
        return self._decoder.read()

    def start(self):
        if self.start_server:
            self._start_scrcpy_server()
        self._setup_adb_tunnel()
        # 初始化视频解码器（内部会自行建立到 host:port 的连接）
        self._decoder = VideoDecoder(self.host, self.port, resize=self.resize_to)
        self._adb_control = AdbControl(serial="N7AIPVDEIFPRRSL7")

    def stop(self):
        self._running.clear()
        if self._reader_thread:
            self._reader_thread.join(2)
        if self._container:
            self._container.close()
        for s in (self._video_sock, self._control_sock):
            if s:
                s.close()
        self._release_adb_tunnel()
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()

    # ────────────────────────── 控制助手 ──────────────────────────
    def tap(self, x: int, y: int):
        if not self._adb_control:
            raise RuntimeError("未启用控制通道")
        self._adb_control.tap(x, y)

    def swipe(self,
              x1: int, y1: int,
              x2: int, y2: int,
              duration: float = .3):
        if not self._adb_control:
            raise RuntimeError("未启用控制通道")
        self._adb_control.swipe(x1, y1, x2, y2, duration)

    def key(self, keycode: int):
        if not self.control:
            raise RuntimeError("未启用控制通道")
        self.control.key(keycode)

    # ────────────────────────── 设备端 server ─────────────────────────

    def _start_scrcpy_server(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]

        # 推送 server jar
        subprocess.run(
            ["adb", "push", str(self.server_jar), "/data/local/tmp/"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # 后台启动 server
        cmd = (
            f"CLASSPATH=/data/local/tmp/{self.server_jar.name} "
            "app_process / com.genymobile.scrcpy.Server "
            f"{self.server_version} {self.server_opts} "
            f"control_port={self.control_port or 0}"
        )

        self._server_proc = subprocess.Popen(
            base + ["shell", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 点亮屏幕
        subprocess.run(
            ["adb", "shell","input keyevent 26"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print("[ADB] scrcpy-server started on device")

    # ────────────────────────── ADB 隧道 ─────────────────────────

    def _setup_adb_tunnel(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]
        if self.tunnel_forward:
            subprocess.run(base + ["forward", f"tcp:{self.port}", "localabstract:scrcpy"], check=True)
            if self.control_port is not None:
                subprocess.run(base + ["forward", f"tcp:{self.control_port}", "localabstract:scrcpy"], check=True)
            print(f"[ADB] forward tcp:{self.port}")
        else:
            subprocess.run(base + ["reverse", f"tcp:{self.port}", f"tcp:{self.port}"], check=True)
            if self.control_port is not None:
                subprocess.run(base + ["reverse", f"tcp:{self.control_port}", f"tcp:{self.control_port}"], check=True)
            print(f"[ADB] reverse tcp:{self.port}")

    def _release_adb_tunnel(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]
        cmd = "forward" if self.tunnel_forward else "reverse"
        for p in (self.port, self.control_port):
            if p is None:
                continue
            subprocess.run(base + ["--remove", cmd, f"tcp:{p}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
