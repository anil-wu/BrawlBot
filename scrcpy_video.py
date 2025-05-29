"""Lightweight H.264 video reader that pulls frames directly from the scrcpy
video socket and optionally resizes them for RL input."""
import av
import cv2
import numpy as np
from typing import Tuple, Optional
class VideoDecoder:
    def __init__(self,
                 host: str = "localhost",
                 port: int = 27183,
                 resize: Optional[Tuple[int, int]] = (84, 84)) -> None:
        """Connect to ``tcp://<host>:<port>?listen`` and yield RGB frames.

        Args:
            host: IP where scrcpy is listening (typically localhost).
            port: Video socket port (see --port when spawning scrcpy).
            resize: (width, height) ― if given, frames are resized with
                OpenCV INTER_AREA.  ``None`` keeps the original size.
        """
        url = f"tcp://{host}:{port}"
        # `fflags=nobuffer` greatly reduces latency (<2 frames behind live).
        print("开始链接", url)
        self.container = av.open(url, options={"fflags": "nobuffer"})
        print("链接成功")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"  # allow ffmpeg to multithread
        self.resize = resize
        print("decode stream start")
        self._frames = self.container.decode(self.stream)
        print("decode stream end ")

    def read(self) -> np.ndarray:
        """Return next RGB frame as numpy uint8 array (H, W, 3)."""
        frame = next(self._frames)             # may raise StopIteration
        img = frame.to_ndarray(format="rgb24")
        # if self.resize is not None:
        #     img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        return img

    def close(self) -> None:
        self.container.close()