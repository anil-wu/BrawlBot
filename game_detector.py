# game_detector.py
from ultralytics import YOLO
from enum import Enum

# ───────────────────── 枚举定义 ─────────────────────
class GameState(str, Enum):
    # VICTORY = "victory"
    FAILE  = "faile"
    SETTLE  = "settle"
    WATTINT_START = "waittingStart"
    PLAYING  = "Playing"
    GAME_OVER  = "gameOver"

    LOBBY = "lobby"

class GameDetector:
    def __init__(self, weight="best.pt", device=0, k=1):
        self.device = device            # 0 / "0,1" / "cpu"
        self.model = YOLO(weight)       # 不再手动 .to()
        self.k = k                      # 屏幕分辨率与帧图片分辨率的比例

    def detect(self, frame, conf=0.4, imgsz=1080):
        result = self.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            # device=self.device,
            verbose=False
        )[0]

        return [
            {
                "cls":  result.names[int(b.cls)],
                "conf": float(b.conf),
                "xyxy": tuple(int(v) for v in b.xyxy[0])
            }
            for b in result.boxes
        ]

    def bbox_center2screen_pos(self, bbox):
        # 中心点（图片坐标系）
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # 统一放大
        print(cx, cy, self.k)
        X_screen = cx * self.k
        Y_screen = cy * self.k
        return (int(X_screen), int(Y_screen))

    # def get_game_state(self, frame, ctrl:AdbControl):
    #     dets = self.detect(frame)
    #     # 收集坐标
    #     coords = {}        # dict[str, list[tuple]]
    #     for d in dets:
    #         cls_name = d["cls"]
    #         # if cls_name in INTERESTED_CLASSES:
    #         print(cls_name)
    #         # if cls_name == "BattleBtn":
    #         coords[cls_name] = bbox_center2screen_pos(d["xyxy"], k=self.k)
    #         pot = bbox_center2screen_pos(d["xyxy"], k=self.k)
    #         ctrl.tap(*pot)
    #     return coords
