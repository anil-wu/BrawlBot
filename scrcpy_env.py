"""Gymnasium‑compatible environment that exposes an Android screen through
scrcpy and injects touch events back to the device."""
from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Deque

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from scrcpy_video import VideoDecoder
from scrcpy_control import ControlSocket


class ScrcpyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self,
                 host: str = "127.0.0.1",
                 video_port: int = 27183,
                 control_port: int = 27184,
                 frame_stack: int = 4,
                 resize: Tuple[int, int] = (84, 84)):
        super().__init__()
        self.decoder = VideoDecoder(host, video_port, resize=resize)
        self.ctrl = ControlSocket(host, control_port)
        self.resize = resize

        # Discrete "tap on a 3 × 3 grid" + no‑op (index 0).
        self._grid = [(0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
                      (0.25, 0.50), (0.50, 0.50), (0.75, 0.50),
                      (0.25, 0.75), (0.50, 0.75), (0.75, 0.75)]
        self.action_space = Discrete(len(self._grid) + 1)

        # Observation = stacked RGB frames (C, H, W)
        c, h, w = frame_stack, resize[1], resize[0]
        self.observation_space = Box(0, 255, (c, h, w, 3), np.uint8)
        self._frames: Deque[np.ndarray] = deque(maxlen=frame_stack)

    # ――― Gym API ―――
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        # Quick soft‑reset: press HOME to minimise game menus.
        self.ctrl.key(ControlSocket.KEYCODE_HOME)
        self._frames.clear()
        f = self.decoder.read()
        for _ in range(self._frames.maxlen):
            self._frames.append(f)
        return self._get_obs(), {}

    def step(self, action: int):
        if action > 0:
            x_ratio, y_ratio = self._grid[action - 1]
            w, h = self.resize
            self.ctrl.tap(int(x_ratio * w), int(y_ratio * h))
        frame = self.decoder.read()
        self._frames.append(frame)

        # TODO ===== game‑specific termination / reward logic here =====
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self._frames, axis=0)

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        cv2.imshow("ScrcpyEnv", self._frames[-1])
        cv2.waitKey(1)

    def close(self):
        self.decoder.close()
        self.ctrl.close()
        cv2.destroyAllWindows()