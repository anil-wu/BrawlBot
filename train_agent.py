import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # 新增导入
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # 新增回调函数

from adb_control import AdbControl
from scrcpy_env import ScrcpyEnv
from scrcpy_video import VideoDecoder
from env_launcher import ScrcpyLauncher
from game_detector import GameDetector, GameState


from stable_baselines3.common.callbacks import BaseCallback

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial_coef=0.5, final_coef=0.01, decay_steps=10000):
        super().__init__()
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        current_coef = self.initial_coef - (self.initial_coef - self.final_coef) * progress
        self.model.ent_coef = current_coef
        self.logger.record("train/ent_coef", current_coef)
        return True

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
                save_freq=500,
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

            # 在回调列表中添加
            entropy_callback = EntropyScheduleCallback(
                initial_coef=0.5,
                final_coef=0.01,
                decay_steps=10000
            )

            steps = 2048
            # # 3️⃣ 训练 PPO
            model = PPO(
                "CnnPolicy",
                vec_env,
                n_steps=steps,
                batch_size=64,
                n_epochs=10,    # 添加epoch参数
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.1,  # 增加熵系数促进探索
                gamma=0.99,     # 提高长期回报考量
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
    finally:
        decoder.close()
        launcher.stop()  # 确保环境停止
        if 'vec_env' in locals():
            vec_env.close()


if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        main(config)