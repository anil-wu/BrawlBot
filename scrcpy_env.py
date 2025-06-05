from __future__ import annotations
import time
import numpy as np
import cv2
from collections import deque
from typing import Deque, Tuple, Optional
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from scrcpy_video import VideoDecoder
from adb_control import AdbControl
from game_detector import GameDetector, GameState

# ——————————— 屏幕&摇杆参数 ———————————
# SCREEN_W, SCREEN_H = 2712, 1220
JOY_CX, JOY_CY = 400, 930        # 摇杆中心
JOY_R = 180                      # 半径

# 八方向向量（单位圆）
DIR_VECS = np.array([
    (-1,  0), (1,  0), (0, -1), (0,  1),
    (-1, -1), (1, -1), (-1, 1), (1, 1)
], dtype=np.float32)

DIR_VECS /= np.linalg.norm(DIR_VECS, axis=1, keepdims=True)

ATTACK_BTN = (2280, 750)
SKILL_BTN  = (1940, 890)

class ScrcpyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 decoder: VideoDecoder,
                 ctrl: AdbControl,
                 detector: GameDetector,
                 resize: Tuple[int, int],
                 frame_stack: int = 4):
        super().__init__()
        self.decoder = decoder #VideoDecoder(host, video_port, resize=resize)
        self.ctrl = ctrl #AdbControl(serial)
        self.detector = detector
        self.resize = resize
        self.frame_stack = frame_stack          # ← 保存一下，后面要用

        # === 动作空间同原来 ===
        self.action_space = Discrete(11)

        # —— 关键：把 observation_space 改成 (C, H, W) ——
        h, w = resize[1], resize[0]
        c = frame_stack * 3                     # 3 通道 × 堆叠帧
        self.observation_space = Box(0, 255, (c, h, w), np.uint8)

        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)
        self.frame_num = 0

    def execute_battle_flow(self):
        isStart = False
        while not isStart:  # 等待游戏开始
            time.sleep(1)
            frame = self.decoder.read()
            dets = self.detector.detect(frame)

            if len(dets) == 0:
                continue
            for d in dets:
                cls_name = d["cls"]
                print(cls_name)
                pot = self.detector.bbox_center2screen_pos(d["xyxy"])
                if cls_name == "BattleBtn":
                    print(cls_name, pot)
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "QuitBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "ContinueBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "AgainBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "StartGameTips":
                    isStart =True
                    break

    # -------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.frames.clear()

        self.execute_battle_flow()

        f = self.decoder.read()
        for _ in range(self.frames.maxlen):
            self.frames.append(f)
        return self._obs(), {}

    def step(self, action: int):
        print(f"执行游戏动作: {action}")
        self.frame_num +=1
        reward = 0.0

        # —— 动作 → 触控 ——
        if 1 <= action <= 8:                       # 摇杆方向
            dx, dy = DIR_VECS[action - 1]
            tx = int(JOY_CX + dx * JOY_R)
            ty = int(JOY_CY + dy * JOY_R)
            self.ctrl.swipe(JOY_CX, JOY_CY, tx, ty, dur_ms=100)
        elif action == 9:                          # 普通攻击
            self.ctrl.tap(*ATTACK_BTN)
        elif action == 10:                         # 技能
            self.ctrl.tap(*SKILL_BTN)


        # —— 获取新帧 ——
        frame = self.decoder.read()
        self.save_frame(self.frame_num, frame)
        self.frames.append(frame)

        dets = self.detector.detect(frame)
        terminated =  self._is_game_over(dets)



        # state = get_game_state(frame, dets)
        # state = get_game_state(frame, action, state)

        # —— 计算奖励 / 终止（需按游戏逻辑改写） ——
        reward += self._compute_reward(dets, action)
        # if state == GameState.DEFEAT:
        #     reward -= 30

        truncated = False
        info = {}

        return self._obs(), reward, terminated, truncated, info

    # -------------- 工具 --------------
    def _obs(self) -> np.ndarray:
        """
        返回 (C, H, W) uint8 张量，满足 stable-baselines3 CnnPolicy 需求：
            - 先把 (stack, H, W, 3) 转为 (stack, 3, H, W)
            - 再合并 stack 与 3 两个通道维度 → (stack*3, H, W)
        """
        arr = np.stack(self.frames, axis=0)               # (k, H, W, 3)
        arr = arr.transpose(0, 3, 1, 2)                   # (k, 3, H, W)
        c, h, w = arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3]
        return arr.reshape(c, h, w)                       # (k*3, H, W)

    def _compute_reward(self, dets, action:int) -> float:


        return 0

    def _is_game_over(self, dets) -> bool:
        if len(dets) == 0:
            print("没有检测到物体")
            return False
        for d in dets:
            cls_name = d["cls"]
            print("_is_game_over",cls_name, cls_name == "DefeatTips")
            if cls_name == "DefeatTips":
                print("游戏结束")
                return True
        return False

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        cv2.imshow("ScrcpyEnv", self.frames[-1])
        cv2.waitKey(1)
    def save_frame(self, frame_num, frame):
        path = "./frames/frame_" + str(frame_num) + ".png"
        cv2.imwrite(path, frame)

    def close(self):
        self.decoder = None
        # cv2.destroyAllWindows()
