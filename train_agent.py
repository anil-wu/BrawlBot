"""Parallel PPO training harness using Stable‑Baselines3."""
import argparse
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from scrcpy_env import ScrcpyEnv


def make_env(rank: int, host: str, base_video_port: int, base_control_port: int):
    """Utility for SubprocVecEnv ― each worker gets its own port pair."""
    def _init():
        env = ScrcpyEnv(
            host=host,
            video_port=base_video_port + 2 * rank,
            control_port=base_control_port + 2 * rank,
        )
        set_random_seed(rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="how many envs")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="total training timesteps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--video-port", type=int, default=27183)
    parser.add_argument("--control-port", type=int, default=27184)
    args = parser.parse_args()

    env_fns = [make_env(i, args.host, args.video_port, args.control_port)
               for i in range(args.workers)]
    vec_env = SubprocVecEnv(env_fns, start_method="fork")

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
    model.learn(total_timesteps=args.steps)
    model.save("ppo_scrcpy")

    vec_env.close()


if __name__ == "__main__":
    main()