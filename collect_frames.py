import time
import yaml
import cv2
from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher
from scrcpy_video import VideoDecoder
from adb_control import AdbControl

r = 0.7
ATTACK_BTN = (2280, 750)
SKILL_BTN  = (1940, 890)
DEFEAT_BTN = (1403, 1133)
GO_ON_BTN  = (2450, 1130)
PLAY_AGIN_BTN  = (2000, 1130)
BATTLE_BTN  = (1720/r, 790/r)

if __name__ == "__main__":

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

        SCREEN_DIM = config["display"]["screen"]
        FRAME_DIM = config["display"]["frame"]
        ADB_BRIDGE = config["adb_bridge"]

        host  = config["adb_bridge"]["host"]
        port  = config["adb_bridge"]["port"]
        serial = config["adb_bridge"]["serial"]

        k =  SCREEN_DIM[0]/FRAME_DIM[0] #2.511111111111111

        def save_frame(episode, frame_num, frame):

                path = "./frames/"+str(episode)+"_frame_" + str(frame_num) + ".png"
                cv2.imwrite(path, frame)

        launcher = ScrcpyLauncher(
            video_port=1234,
        )
        try:
            with launcher:
                decoder = VideoDecoder("localhost", 1234, resize=(FRAME_DIM[0], FRAME_DIM[1]))
                ctrl = AdbControl(None)
                Episode = 2
                frame_num = 0
                while True :
                    time.sleep(0.1)
                    frame_num += 1
                    frame = decoder.read()
                    if frame is None: continue
                    save_frame(Episode, frame_num, frame)

        except KeyboardInterrupt:
            print("\n[train_agent] 手动中断，已安全退出。")


