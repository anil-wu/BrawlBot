# video_decoder.py
from __future__ import annotations
import av, cv2, threading, time
import numpy as np
from typing import Optional, Tuple


class VideoDecoder:
    """
    后台线程不断解码，只保留一张“最新帧”。
    read() 随取随用，不会落后于实时画面。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 27183,
        resize: Optional[Tuple[int, int]] = None,
    ):
        url = f"tcp://{host}:{port}"
        self.url = url
        self.resize = resize
        self.link_av()

    # ---------------- 公共接口 ----------------
    def read(self, *, block: bool = True, timeout: float = 1.0) -> np.ndarray:
        """
        返回最新 RGB ndarray。
        参数
        ----
        block   : True 阻塞到拿到第一帧；False 立即返回 None
        timeout : 首帧最大等待秒数
        """
        start = time.time()
        while self._last_frame is None:
            if not block or (time.time() - start) > timeout:
                return None
            time.sleep(0.005)

        with self._lock:
            return self._last_frame.copy()       # 防止外部修改原帧

    def close(self):
        self._running = False
        self._thread.join()
        self.container.close()

    def link_av(self):
        self.container = av.open(self.url, options={"fflags": "nobuffer"})
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # ---------------- 内部线程 ----------------
    def _reader_loop(self):
        for packet in self.container.demux(self.stream):
            if not self._running:
                break
            for frame in packet.decode():
                img = frame.to_ndarray(format="rgb24")
                if self.resize:
                    img = cv2.resize(img, self.resize,
                                     interpolation=cv2.INTER_AREA)
                with self._lock:
                    self._last_frame = img
