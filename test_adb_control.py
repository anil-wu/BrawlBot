from adb_control import AdbControl

ctl = AdbControl(serial="N7AIPVDEIFPRRSL7")  # 也可以不传 serial
# ctl.tap(540, 1200)                           # 点击
ctl.swipe(540, 1600, 540, 400, 100)          # 0.5 秒上滑
