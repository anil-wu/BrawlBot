from __future__ import annotations
import cv2
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

__all__ = ["GameState", "get_game_state", "load_templates"]

# ───────────────────── 枚举定义 ─────────────────────
class GameState(str, Enum):
    # VICTORY = "victory"
    DEFEAT  = "defeat"
    SETTLE  = "settle"
    BATTLE  = "battle"
    GO_ON   = "go_on"
    LOBBY = "lobby"


# ───────────────────── 模板资源 ─────────────────────
_ASSET_DIR = Path(__file__).parent / "assets"
_DEFAULT_TEMPLATES = {
    # GameState.VICTORY: _ASSET_DIR / "victory_tpl.png",
    GameState.DEFEAT:  _ASSET_DIR / "defeat_tpl.png",
    GameState.SETTLE:  _ASSET_DIR / "settle_tpl.png",
    GameState.GO_ON:  _ASSET_DIR / "go_on.png",
    GameState.LOBBY:  _ASSET_DIR / "loby.png",
}

_ROI = {
    # GameState.VICTORY: (slice(0, 600), slice(1400, 2712)),   # (y_slice, x_slice)
    GameState.DEFEAT:  (slice(765, 835), slice(930, 1045)),
    GameState.SETTLE:  (slice(752, 844), slice(1260, 1560)),
    GameState.GO_ON:  (slice(752, 844), slice(1590, 1890)),
    GameState.LOBBY:  (slice(720, 844), slice(1516, 1905)),
}

_TPL_GRAY: Dict[GameState, np.ndarray] = {}

# _THRESH:   Dict[GameState, float] = {
#     GameState.VICTORY: 0.80,
#     GameState.DEFEAT:  0.80,
#     GameState.SETTLE:  0.80,
# }


def load_templates(custom_map: Dict[GameState, Path | str] | None = None) -> None:
    """
    初始化/替换模板。
    Parameters
    ----------
    custom_map : dict
        可传入 {GameState: "path/to/file.png"} 覆盖默认路径。
    """
    path_map = _DEFAULT_TEMPLATES.copy()
    if custom_map:
        path_map.update({k: Path(v) for k, v in custom_map.items()})

    for state, p in path_map.items():
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[game_state] 模板不存在或无法读取: {p}")
        _TPL_GRAY[state] = img
        print(f"[game_state] 模板尺寸: {state} {img.shape}")
    print(f"[game_state] 模板加载完毕: {', '.join(s.name for s in _TPL_GRAY)}")


# 在 import 时就尝试加载默认模板；找不到会在首次调用时抛错误
try:
    load_templates()
except FileNotFoundError as e:
    print(e)
    print("[game_state] ⚠️ 未找到默认模板，请调用 load_templates() 指定路径后再用 get_game_state().")


# ───────────────────── 主判别函数 ─────────────────────
def _match(gray: np.ndarray, tpl: np.ndarray) -> float:
    """返回模板匹配最大相似度（0~1）。"""
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    print(f"[game_state] _match value :  {max_val:.3f}")
    return max_val

def _roi(gray: np.ndarray, state: GameState) -> np.ndarray:
    print(f"[game_state] 模板匹配区域: {state}")
    ys, xs = _ROI[state]
    return gray[ys, xs]

def get_game_state(frame_rgb: np.ndarray) -> GameState:
    """
    根据 RGB ndarray 判别当前状态。
    返回 GameState 枚举值。
    """
    if not _TPL_GRAY:
        raise RuntimeError("模板未加载，请先调用 load_templates().")

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)   # ← 正确的颜色码

    # 1) 失败检测
    roi = _roi(gray, GameState.LOBBY)
    if _match(roi, _TPL_GRAY[GameState.LOBBY]) > 0.95:
        return GameState.LOBBY

    # 1) 失败检测
    roi = _roi(gray, GameState.DEFEAT)
    if _match(roi, _TPL_GRAY[GameState.DEFEAT]) > 0.95:
        return GameState.DEFEAT

    # 2) 继续按钮
    roi = _roi(gray, GameState.GO_ON)
    if _match(roi, _TPL_GRAY[GameState.GO_ON]) > 0.95:
        return GameState.GO_ON

    # 3) 结算检测
    roi = _roi(gray, GameState.SETTLE)
    if _match(roi, _TPL_GRAY[GameState.SETTLE]) > 0.95:
        return GameState.SETTLE



    return GameState.BATTLE
