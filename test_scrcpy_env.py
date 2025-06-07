
from __future__ import annotations
import argparse
import time
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scrcpy_video import VideoDecoder
from adb_control import AdbControl
from game_detector import GameDetector, GameState


from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher

def main(config):
    steps = 1_000_000
    SCREEN_DIM = config["display"]["screen"]
    FRAME_DIM = config["display"]["frame"]
    ADB_BRIDGE = config["adb_bridge"]

    host  = config["adb_bridge"]["host"]
    port  = config["adb_bridge"]["port"]
    serial = config["adb_bridge"]["serial"]

    k =  SCREEN_DIM[0]/FRAME_DIM[0] #2.511111111111111
    print(k)

    # 1️⃣ 启动 scrcpy-server 与 ADB forward
    launcher = ScrcpyLauncher(
        serial = serial,
        video_port= port,
    )
    try:
        with launcher:
            resize = (FRAME_DIM[0], FRAME_DIM[1])
            decoder = VideoDecoder(host, port, resize=resize)
            detector = GameDetector(k=k)
            ctrl = AdbControl(serial)
            vec_env = ScrcpyEnv(
                decoder,
                ctrl,
                detector,
                resize,
                )

            Episode = 1
            while Episode <= 5:
                print(f"\n[train_agent] 开始训练第 {Episode} 轮。")
                vec_env.execute_battle_flow()
                start_train = True
                while start_train:
                    obs, r, done, trunc, info = vec_env.step(vec_env.action_space.sample())
                    if done:
                        Episode += 1
                        start_train =  False
                        print("\n[train_agent] 训练结束，已安全退出。")

            print(f"[train_agent] 训练完成，已安全退出。")

    except KeyboardInterrupt:
        print("\n[train_agent] 手动中断，已安全退出。")
    finally:
        decoder.close()
        launcher.stop()



if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        main(config)
