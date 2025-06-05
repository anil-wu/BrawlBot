import time
from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher

launcher = ScrcpyLauncher(
    video_port=1234,
)
try:
    with launcher:
        env = ScrcpyEnv(
                    host="localhost",
                    video_port=1234,
                    serial=None,
                )

        Episode = 1
        while Episode <= 5:
            # time.sleep(5)

            print("第 {Episode} 局游戏开始")
            obs, _ = env.reset()
            done = False

            while not done:
                obs, r, done, trunc, info = env.step(env.action_space.sample())

            print("第 {Episode} 局游戏结束")
            Episode += 1

            # evn.close()

except KeyboardInterrupt:
    print("\n[train_agent] 手动中断，已安全退出。")
finally:
    cli.stop()