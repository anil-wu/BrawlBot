"""Thin wrapper around the scrcpy *control* socket (port + 1)."""
import socket
import struct
import time

class ControlSocket:
    # Touch actions (android.view.MotionEvent constants)
    ACTION_DOWN = 0
    ACTION_UP   = 1
    ACTION_MOVE = 2

    # Key example (see KeyEvent.java)
    KEYCODE_HOME = 3

    def __init__(self, host: str = "127.0.0.1", port: int = 27184):
        print(f"ControlSocket Connecting to {host}:{port}")
        self.sock = socket.create_connection((host, port))

    # ――― Touch helpers ―――
    def _touch_msg(self, pointer_id: int, action: int, x: int, y: int,
                   pressure: float = 1.0, display_id: int = 0) -> bytes:
        """Build a TOUCH message as defined in control_msg.h (type = 2)."""
        return struct.pack(
            ">BqBIIIH",
            2,                   # type = SCRCPY_MSG_TYPE_INJECT_TOUCH
            pointer_id,
            action,
            x,
            y,
            int(pressure * 65535),
            display_id,
        )

    def tap(self, x: int, y: int, pointer_id: int = 0) -> None:
        self.sock.sendall(self._touch_msg(pointer_id, self.ACTION_DOWN, x, y))
        time.sleep(0.015)  # 15 ms ― just enough for the down‑up pair
        self.sock.sendall(self._touch_msg(pointer_id, self.ACTION_UP,   x, y))

    # ――― Key helpers ―――
    def key(self, keycode: int, action: int = 0) -> None:
        """Send simple key event (type 0)."""
        msg = struct.pack(">BIIB", 0, keycode, 0, action)
        self.sock.sendall(msg)

    def swipe(self,
              x1: int, y1: int, x2: int, y2: int,
              duration: float = .3,
              steps: int = 12) -> None:
        """
        模拟滑动：duration 秒内分 steps 次 MOVE。
        """
        if steps < 1:
            steps = 1
        self._send(self._pack_touch(x1, y1, ACTION_DOWN))
        dt = duration / steps
        for i in range(1, steps):
            xi = int(x1 + (x2 - x1) * i / steps)
            yi = int(y1 + (y2 - y1) * i / steps)
            time.sleep(dt)
            self._send(self._pack_touch(xi, yi, ACTION_MOVE))
        time.sleep(dt)
        self._send(self._pack_touch(x2, y2, ACTION_UP))

    def close(self):
        self.sock.close()