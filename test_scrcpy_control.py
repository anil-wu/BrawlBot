import argparse
import subprocess
import re
import time
from typing import Tuple, Optional
from scrcpy_control import ControlSocket

# 常用按键名 -> 键码映射
KEYMAP = {
    "HOME": 3,
    "BACK": 4,
    "MENU": 82,
    "POWER": 26,
    "VOLUME_UP": 24,
    "VOLUME_DOWN": 25,
}

def get_device_resolution(serial: Optional[str] = None) -> Tuple[int, int]:
    """通过 `adb shell wm size` 获取物理分辨率。"""
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["shell", "wm", "size"]
    out = subprocess.check_output(cmd, encoding="utf-8", errors="ignore")
    m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
    if not m:
        raise RuntimeError(f"无法解析分辨率：{out.strip()}")
    return int(m.group(1)), int(m.group(2))

def main() -> None:
    parser = argparse.ArgumentParser(description="测试 ControlSocket 注入")
    parser.add_argument("--host", default="127.0.0.1", help="control 套接字地址")
    parser.add_argument("--port", type=int, default=27183, help="control 套接字端口")

    action_grp = parser.add_mutually_exclusive_group(required=True)
    action_grp.add_argument("--tap", nargs=2, metavar=("X", "Y"), type=int,
                            help="发送一次触控到 (X, Y)")
    action_grp.add_argument("--key", choices=KEYMAP.keys(),
                            help="发送指定按键")

    parser.add_argument("--repeat", type=int, default=1, help="重复次数")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="两次动作间隔秒数")

    args = parser.parse_args()

    cs = ControlSocket(args.host, args.port)
    print(f"[INFO] 已连接 control 套接字 {args.host}:{args.port}")

    w, h = get_device_resolution("N7AIPVDEIFPRRSL7")
    print(f"[RES] 设备分辨率: {w}×{h}") # 测试分辨率1220×2712

    try:
        for i in range(args.repeat):
            if args.tap:
                x, y = args.tap
                cs.tap(x, y)
                print(f"[TAP] ({x}, {y}) #{i + 1}")
            else:
                keycode = KEYMAP[args.key]
                cs.key(keycode, action=0)  # key down
                cs.key(keycode, action=1)  # key up
                print(f"[KEY] {args.key} ({keycode}) #{i + 1}")
            if i < args.repeat - 1:
                time.sleep(args.interval)
    finally:
        cs.close()
        print("[INFO] 测试完成，套接字已关闭。")


if __name__ == "__main__":
    main()