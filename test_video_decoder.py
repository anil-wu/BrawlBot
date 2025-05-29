import argparse
import time
from pathlib import Path
import cv2
from scrcpy_video import VideoDecoder


def main() -> None:
    parser = argparse.ArgumentParser(description="测试 VideoDecoder 延迟与 FPS")
    parser.add_argument("--host", default="localhost", help="scrcpy 视频地址")
    parser.add_argument("--port", type=int, default=27183, help="视频套接字端口")
    parser.add_argument("--width", type=int, default=1220, help="缩放后宽度")
    parser.add_argument("--height", type=int, default=2712, help="缩放后高度")
    parser.add_argument("--duration", type=int, default=10, help="测试时长(秒)")
    parser.add_argument("--outdir", default="frames", help="保存帧的目录")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    decoder = VideoDecoder(args.host, args.port, resize=(args.width, args.height))
    start_time = last_time = time.time()
    total_frames = sec_frames = 0

    print("[INFO] 正在采集… 按 Ctrl+C 可提前终止")
    try:
        while True:
            frame = decoder.read()
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

            # # 达到设定时长退出
            # if now - start_time >= args.duration:
            #     break
    except KeyboardInterrupt:
        print("[INFO] 用户终止。")

    elapsed = time.time() - start_time
    avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
    print(f"[SUMMARY] 总帧数 {total_frames}, 用时 {elapsed:.2f}s, 平均 FPS ≈ {avg_fps:.2f}")

    decoder.close()


if __name__ == "__main__":
    main()