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
from env_launcher import ScrcpyLauncher
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
                 launch:ScrcpyLauncher,
                 resize: Tuple[int, int],
                 frame_stack: int = 4):
        super().__init__()
        self.decoder = decoder #VideoDecoder(host, video_port, resize=resize)
        self.ctrl = ctrl #AdbControl(serial)
        self.detector = detector
        self.launch = launch
        self.resize = resize
        self.frame_stack = frame_stack          # ← 保存一下，后面要用

        self.batel_num = 0

        # === 动作空间同原来 ===
        self.action_space = Discrete(11)

        self.last_action = None  # 新增：记录上一个有效动作
        self.action_counter = 0  # 新增：连续执行同一动作的计数器

        # —— 关键：把 observation_space 改成 (C, H, W) ——
        h, w = resize[1], resize[0]
        c = frame_stack * 3                     # 3 通道 × 堆叠帧
        self.observation_space = Box(0, 255, (c, h, w), np.uint8)

        self.frames: Deque[np.ndarray] = deque(maxlen=frame_stack)
        self.frame_num = 0

    def execute_battle_flow(self):
        isStart = False
        while not isStart:  # 等待游戏开始
            time.sleep(0.1)
            if not self.ctrl.check_adb_link() :
                break

            frame = self.decoder.read()
            if frame is None: continue
            dets = self.detector.detect(frame)

            if len(dets) == 0:
                continue
            for d in dets:
                cls_name = d["cls"]
                if d["conf"] < 0.8:
                    continue
                pot = self.detector.bbox_center2screen_pos(d["xyxy"])
                if cls_name == "BattleBtn" :
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "QuitBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "ContinueBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "ExitCheckout":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "AgainBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name in ["SkillCD", "SkillFull"]:
                    isStart =True
                    break

    # -------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        print("重置环境")
        super().reset(seed=seed)
        while True:
            time.sleep(1)
            # 加入循环体？等待重连
            if not self.ctrl.check_adb_link() :
                print("等待设备重新连接")
                while True :
                    time.sleep(1)
                    if self.ctrl.check_adb_link() :
                        print("设备已连接")
                        time.sleep(10)
                        self.launch.launch()
                        self.decoder.link_av()
                        break

            self.frames.clear()
            print("尝试启动战斗")
            self.execute_battle_flow()
            f = self.decoder.read()
            for _ in range(self.frames.maxlen):
                self.frames.append(f)

            self.last_action = None  # 新增：重置动作记录
            self.action_counter = 0  # 新增：重置计数器

            if self.ctrl.check_adb_link() :
                break
            else:
                print("设备已断开连接 ")

        self.batel_num +=1
        print("开始战斗... 场次", self.batel_num)
        return self._obs(), {}

    def step(self, action: int):
        self.frame_num +=1
        print(f"Setp: {self.frame_num} --------------------------")
        print(f"执行动作: {action}")

        terminated = False
        truncated = False
        info = {}
        reward = 0.0

        # —— 动作 → 触控 ——
        if 1 <= action <= 8:                       # 摇杆方向
            dx, dy = DIR_VECS[action - 1]
            tx = int(JOY_CX + dx * JOY_R)
            ty = int(JOY_CY + dy * JOY_R)
            self.ctrl.swipe(JOY_CX, JOY_CY, tx, ty, dur_ms=300)
        elif action == 9:                          # 普通攻击
            self.ctrl.tap(*ATTACK_BTN)
        elif action == 10:                         # 技能
            self.ctrl.tap(*SKILL_BTN)

        if  not self.ctrl.check_adb_link() :
            terminated = True
            obs = self._obs()
            self.decoder.close()
            print("[train_agent] ADB连接已断开，已安全退出。")
            return obs, reward, terminated, truncated, info

        time.sleep(0.1)


        # —— 获取新帧 ——
        frame = self.decoder.read()
        if frame is None:
            print("帧读取失败")

        self.save_frame(self.frame_num, frame)
        self.frames.append(frame)

        dets = self.detector.detect(frame)
        terminated =  self._is_game_over(dets)

        if terminated:
            reward += self._settlement_reward()
        else:
            reward += self._compute_reward(dets, action)

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

    def _settlement_reward(self)->float:
        print("进入奖励结算环节")
        is_settlement_status = False
        while True:
            time.sleep(1)
            if not self.ctrl.check_adb_link() :
                break
            frame = self.decoder.read()
            dets = self.detector.detect(frame)

            if len(dets) == 0:
                continue

            for d in dets:
                cls_name = d["cls"]
                if is_settlement_status:
                    if cls_name == "RankedFirst":
                        print("战斗胜利， 加分!")
                        return 1000
                    elif cls_name == "RankedSecond":
                        print("位列第二名， 加分!")
                        return 100
                    else:
                        print("战斗失败， 扣分!")
                        return -1000

                pot = self.detector.bbox_center2screen_pos(d["xyxy"])
                if cls_name == "QuitBtn":
                    self.ctrl.tap(*pot)
                    continue

                if cls_name == "ContinueBtn":
                    is_settlement_status = True
                    continue

    def _compute_reward(self, dets, action:int) -> float:
        reward = 0

        if action in [1, 8]:  # 左/左上
            reward += 1  # 鼓励其他方向

        if action in [9, 10]:  # 左/左上
            reward += 1  # 鼓励其他方向

        if action == 0:
            print("未有操作，扣分!")
            reward -= 2

        if action == self.last_action:
            self.action_counter += 1
            print(f"连续执行动作{action} ({self.action_counter}/5)")

            # 连续5次执行同一动作时扣分
            if self.action_counter >= 5:
                print(f"⚠️ 连续5次执行动作{action}，扣分！")
                self.action_counter = 0  # 重置计数器
                reward -= 10  # 重复惩罚
        else:
            # 动作改变时重置计数器
            self.action_counter = 1
            self.last_action = action

        if len(dets) == 0:
            return reward

        for d in dets:
            cls_name = d["cls"]
            if cls_name == "DefeatTips":
                continue

            if cls_name == "EnemyBloodLoss":
                print("敌方掉血，加分")
                reward += 10
                continue

            if cls_name == "HeroBloodLoss":
                print("玩家掉血，扣分")
                reward -= 5
                continue

            # if cls_name == "SkillCD" and action == 10:
            #     print("操作冷却中的技能，扣分!")
            #     reward -= 5
            #     continue

            if cls_name == "LowHP":
                print("血量过低， 扣分!")
                reward -= 5
                continue

            if cls_name == "KillEnemy" and d["conf"] >=0.85:
                print("击杀敌人， 加分!")
                reward += 100
                print(d)
                continue

        return reward

    def _is_game_over(self, dets) -> bool:
        if len(dets) == 0:
            return False

        for d in dets:
            cls_name = d["cls"]

            if cls_name in ["DefeatTips", "ContinueBtn"]:
                # 游戏失败会直接跳转到失败界面，然后需要确认退出
                # 游戏胜利时候会直接出现有继续的按钮的页面
                print("游戏结束")
                return True

            if cls_name in ["SkillCD", "SkillFull"]:
                return False

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
