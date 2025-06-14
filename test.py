import time
import cv2
from scrcpy_env     import ScrcpyEnv
from env_launcher   import ScrcpyLauncher
from scrcpy_video import VideoDecoder
from adb_control import AdbControl
from game_detector import GameState, GameDetector
from checker_monitor import ColorCheckerMonitor

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
        decoder = VideoDecoder(ADB_BRIDGE["host"], ADB_BRIDGE["port"], resize=(FRAME_DIM[0], FRAME_DIM[1]))


        frame = decoder.read()

        while frame is None :
            frame = decoder.read()
            time.sleep(0.1)

        # 初始化FPS计算相关变量
        frame_count = 0
        start_time = time.time()
        fps = 0

        # 创建可调整大小的显示窗口
        window_name = "Movement Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, FRAME_DIM[0], FRAME_DIM[1])

        roi = (int(FRAME_DIM[0]/2), int(FRAME_DIM[1]/2), 300, 150)
        monitor= ColorCheckerMonitor(roi)
        while True :

            # 读取帧
            frame = decoder.read()
            if frame is None:
                continue
            time.sleep(0.05)
            # 计算FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:  # 每0.5秒更新一次FPS
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # 检测移动
            moved, offset, vis = monitor.check_movement(frame)

            # 在图像上显示FPS
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示结果（窗口大小已预先设置）
            cv2.imshow(window_name, frame)

            # 打印检测结果
            print(f"\r移动: {moved}, 偏移: {offset}, FPS: {fps:.1f}", end="", flush=True)

            # 检测退出按键
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            # 可选：添加窗口大小调整快捷键
            elif key & 0xFF == ord('1'):
                cv2.resizeWindow(window_name, 640, 480)  # 小尺寸
            elif key & 0xFF == ord('2'):
                cv2.resizeWindow(window_name, 800, 600)  # 中等尺寸
            elif key & 0xFF == ord('3'):
                cv2.resizeWindow(window_name, 1024, 768)  # 大尺寸
            elif key & 0xFF == ord('0'):
                cv2.resizeWindow(window_name, FRAME_DIM[0], FRAME_DIM[1])  # 原始尺寸

        cv2.destroyAllWindows()

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("\n[train_agent] 手动中断，已安全退出。")

finally:
    launcher.stop()