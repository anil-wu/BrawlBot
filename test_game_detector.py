import time
import cv2
from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher
from scrcpy_video import VideoDecoder
from adb_control import AdbControl
from game_detector import GameState, GameDetector

import yaml  # 需要安装PyYAML: pip install PyYAML

with open('config.yaml') as f:
    config = yaml.safe_load(f)

SCREEN_DIM = config["display"]["screen"]
FRAME_DIM = config["display"]["frame"]
ADB_BRIDGE = config["adb_bridge"]

print(ADB_BRIDGE)

def bbox_center2screen_pos(bbox, k):
    # 中心点（图片坐标系）
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # 统一放大
    X_screen = cx * k
    Y_screen = cy * k
    return (int(X_screen), int(Y_screen))

def get_game_state(frame, detector, ctrl:AdbControl):
    dets = detector.detect(frame)
    k = SCREEN_DIM[0] / FRAME_DIM[0]
    print(k)
    # 收集坐标
    coords = {}        # dict[str, list[tuple]]
    for d in dets:
        cls_name = d["cls"]
        # if cls_name in INTERESTED_CLASSES:
        print(cls_name)
        # if cls_name == "BattleBtn":
        pot = bbox_center2screen_pos(d["xyxy"], k=k)
        ctrl.tap(*pot)

def save_frame(episode, frame_num, frame):

        path = "./frames/"+str(episode)+"_frame_" + str(frame_num) + ".png"
        cv2.imwrite(path, frame)

launcher = ScrcpyLauncher(
    video_port=1234,
)
try:
    with launcher:
        decoder = VideoDecoder(ADB_BRIDGE["host"], ADB_BRIDGE["port"], resize=(FRAME_DIM[0], FRAME_DIM[1]))
        gameDetector = GameDetector()
        ctrl = AdbControl(None)
        Episode = 0
        frame_num = 0
        while True :
            time.sleep(0.1)
            frame_num += 1
            frame = decoder.read()
            if frame is None: continue

            get_game_state(frame, gameDetector, ctrl)
            # save_frame(Episode, frame_num, frame)
            # state = get_game_state(frame)
            # print(f"[game_state] 当前状态: {state}")
            # if state == GameState.LOBBY:
            #     #ctrl.tap(*BATTLE_BTN)
            # elif state == GameState.DEFEAT:
            #     #ctrl.tap(*DEFEAT_BTN)
            # elif state == GameState.GO_ON:
            #     ctrl.tap(*GO_ON_BTN)
            # elif state == GameState.SETTLE:
            #     ctrl.tap(*PLAY_AGIN_BTN)


except KeyboardInterrupt:
    print("\n[train_agent] 手动中断，已安全退出。")
