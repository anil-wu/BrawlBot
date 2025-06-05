from __future__ import annotations
import argparse
import time
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher

def main(config):
    SCREEN_DIM = config["display"]["screen"]
    FRAME_DIM = config["display"]["frame"]
    ADB_BRIDGE = config["adb_bridge"]

    host  = config["adb_bridge"]["host"]
    port  = config["adb_bridge"]["port"]
    serial = config["adb_bridge"]["serial"]

    steps = 1_000_000


    # 1️⃣ 启动 scrcpy-server 与 ADB forward
    launcher = ScrcpyLauncher(
        serial = serial,
        video_por t= port,
    )

    try:
        with launcher:
            # 2️⃣ 创建单环境（用 DummyVecEnv 适配 SB3）
            def make_env():
                return ScrcpyEnv(
                    host = host,
                    video_port = port,
                    serial = serial,
                )

            vec_env = DummyVecEnv([make_env])

            # # 3️⃣ 训练 PPO
            model = PPO(
                "CnnPolicy",
                vec_env,
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                clip_range=0.2,
                tensorboard_log="./runs",
                verbose=1,
            )
            model.learn(total_timesteps = steps)
            model.save("ppo_scrcpy_single")

            vec_env.close()
    except KeyboardInterrupt:
        print("\n[train_agent] 手动中断，已安全退出。")



if __name__ == "__main__":

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    main(config)
