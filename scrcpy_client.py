from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
from scrcpy_video import VideoDecoder
from adb_control import AdbControl

class ScrcpyClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 1234,
        *,
        serial: str | None = None,
        tunnel_forward: bool = True,
        start_server: bool = True,
        server_jar: str | Path = "scrcpy-server.jar",
        server_version: str = "3.2",
        server_opts: str = "tunnel_forward=true audio=false control=false cleanup=false raw_stream=true max_size=1920",
        resize_to: Optional[tuple[int, int]] = (864, 1080),
    ) -> None:
        self.host = host
        self.port = port
        self.serial = serial
        self.tunnel_forward = tunnel_forward
        self.start_server = start_server
        self.server_jar = Path(server_jar)
        self.server_version = server_version
        self.server_opts = server_opts
        self.resize_to = resize_to

        self._adb_control: AdbControl | None = None
        self._server_proc: subprocess.Popen | None = None
        self._decoder: VideoDecoder | None = None

    def frame(self) -> np.ndarray:
        if not self._decoder:
            raise RuntimeError("视频解码器未初始化")
        return self._decoder.read()

    def start(self):
        if self.start_server:
            self._start_scrcpy_server()
        self._setup_adb_tunnel()
        self._decoder = VideoDecoder(self.host, self.port, resize=self.resize_to)
        self._adb_control = AdbControl(serial=self.serial)

    def stop(self):
        if self._adb_control:
            self._adb_control = None
        self._release_adb_tunnel()
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
        if  self._decoder:
            self._decoder.close()

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
        if not self._adb_control:
            raise RuntimeError("未启用控制通道")
        self._adb_control.key(keycode)

    def _start_scrcpy_server(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]

        subprocess.run(
            base + ["push", str(self.server_jar), "/data/local/tmp/"],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        cmd = (
            f"CLASSPATH=/data/local/tmp/{self.server_jar.name} "
            "app_process / com.genymobile.scrcpy.Server "
            f"{self.server_version} {self.server_opts}"
        )

        self._server_proc = subprocess.Popen(
            base + ["shell", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            base + ["shell", "input keyevent 26"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print("[ADB] scrcpy-server started on device")

    def _setup_adb_tunnel(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]

        if self.tunnel_forward:
            subprocess.run(
                base + ["forward", f"tcp:{self.port}", "localabstract:scrcpy"],
                check=True
            )
            print(f"[ADB] forward tcp:{self.port}")
        else:
            subprocess.run(
                base + ["reverse", f"tcp:{self.port}", f"tcp:{self.port}"],
                check=True
            )
            print(f"[ADB] reverse tcp:{self.port}")

    def _release_adb_tunnel(self):
        base = ["adb"]
        if self.serial:
            base += ["-s", self.serial]

        cmd = "forward" if self.tunnel_forward else "reverse"
        subprocess.run(
            base + [cmd, "--remove", f"tcp:{self.port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )