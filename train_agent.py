import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # 新增导入
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # 新增回调函数

from adb_control import AdbControl
from scrcpy_env import ScrcpyEnv
from scrcpy_video import VideoDecoder
from env_launcher import ScrcpyLauncher
from game_detector import GameDetector, GameState

def main(config):
    SCREEN_DIM = config["display"]["screen"]
    FRAME_DIM = config["display"]["frame"]
    ADB_BRIDGE = config["adb_bridge"]

    host  = config["adb_bridge"]["host"]
    port  = config["adb_bridge"]["port"]
    serial = config["adb_bridge"]["serial"]

    k =  SCREEN_DIM[0]/FRAME_DIM[0]

    # 1️⃣ 启动 scrcpy-server 与 ADB forward
    launcher = ScrcpyLauncher(
        serial = serial,
        video_port = port,
    )

    try:
        with launcher:
            resize = (FRAME_DIM[0], FRAME_DIM[1])
            decoder = VideoDecoder(host, port, resize=resize)
            detector = GameDetector(k=k)
            ctrl = AdbControl(serial)

            # 2️⃣ 创建单环境（用 DummyVecEnv 适配 SB3）
            def make_env():
                return ScrcpyEnv(
                    decoder,
                    ctrl,
                    detector,
                    resize,
                )

            vec_env = DummyVecEnv([make_env])

            # ===== 修复监控包装 =====
            # 使用默认监控
            vec_env = VecMonitor(vec_env)

            # ===== 新增回调函数 =====
            # 模型检查点回调（每500步保存一次）
            checkpoint_callback = CheckpointCallback(
                save_freq=20,
                save_path="./checkpoints/",
                name_prefix="brawl_model"
            )

            # 评估回调（每200步评估一次）
            eval_callback = EvalCallback(
                vec_env,
                best_model_save_path="./best_models/",
                eval_freq=10,
                deterministic=True,
                render=False
            )

            steps = 100
            # # 3️⃣ 训练 PPO
            model = PPO(
                "CnnPolicy",
                vec_env,
                n_steps=steps,
                batch_size=64,
                learning_rate=3e-4,
                clip_range=0.2,
                tensorboard_log="./runs",  # 确保TensorBoard日志目录存在
                verbose=1,
            )

            # ===== 训练时传入回调 =====
            model.learn(
                total_timesteps=steps,
                callback=[checkpoint_callback, eval_callback],  # 添加回调
                tb_log_name="荒野乱斗智能玩家"  # TensorBoard实验名称
            )
            model.save("BrawlStars")
            vec_env.close()

    except KeyboardInterrupt:
        print("\n[train_agent] 手动中断，已安全退出。")


if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        main(config)