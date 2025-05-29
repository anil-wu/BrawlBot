import cv2
import time
import argparse
from pathlib import Path
from scrcpy_client import ScrcpyClient

parser = argparse.ArgumentParser(description="测试 VideoDecoder 延迟与 FPS")
parser.add_argument("--outdir", default="frames", help="保存帧的目录")
args = parser.parse_args()

cli = ScrcpyClient(serial="N7AIPVDEIFPRRSL7", server_jar="scrcpy-server.jar", server_version="3.2", tunnel_forward=True)
cli.start()

start_time = last_time = time.time()
total_frames = sec_frames = 0
while True:
    # time.sleep(5)
    # cli.tap(540, 1200)
    frame = cli.frame()
    now = time.time()
    total_frames += 1
    sec_frames += 1

    # 每秒打印一次即时 FPS 并保存当前帧
    if now - last_time >= 1.0:
        cur_fps = sec_frames / (now - last_time)
        print(f"[1s FPS] {cur_fps:.2f}")
        fname = Path(args.outdir) / f"frame_{int(now)}.jpg"
        cv2.imwrite(str(fname), frame)
        last_time = now
        sec_frames = 0
        cli.swipe(540, 1600, 540, 400, 100)