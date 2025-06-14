import time
import math

from env_launcher   import ScrcpyLauncher
from adb_control import AdbControl

import yaml  # 需要安装PyYAML: pip install PyYAML

with open('config.yaml') as f:
    config = yaml.safe_load(f)

SCREEN_DIM = config["display"]["screen"]
FRAME_DIM = config["display"]["frame"]
ADB_BRIDGE = config["adb_bridge"]

print(ADB_BRIDGE)

launcher = ScrcpyLauncher(
    video_port=1234,
)
try:
    with launcher:


        # 圆周运动参数
        center_x, center_y = 400, 930  # 圆心坐标
        radius = 180                    # 圆周半径
        angle_step = math.pi / 18       # 每次增加10度（π/18弧度 ≈ 0.1745弧度）
        current_angle = 0               # 当前角度（弧度制）

        ctrl = AdbControl(None)
        ctrl.touch_down(center_x, center_y)
        while True:
            # 计算圆周上的坐标点
            x = center_x + radius * math.cos(current_angle)
            y = center_y + radius * math.sin(current_angle)

            # 移动手指到新坐标
            ctrl.touch_move(x, y)

            # 增加角度（10度）
            current_angle += angle_step

            # 保持角度在0-2π范围内防止溢出
            if current_angle >= 2 * math.pi:
                current_angle -= 2 * math.pi

            time.sleep(0.1)  # 等待0.1秒



except KeyboardInterrupt:
    print("\n[train_agent] 手动中断，已安全退出。")
