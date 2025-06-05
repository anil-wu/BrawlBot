#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量压缩图片到最长边 ≤ 1080 像素，并以 m_ 前缀保存到原目录
Usage:
    python resize_images.py /path/to/folder
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ExifTags

# --------- 处理 EXIF 方向标记，保证旋转正确 ---------- #
_EXIF_ORIENTATION_TAG = None
for tag, name in ExifTags.TAGS.items():
    if name == "Orientation":
        _EXIF_ORIENTATION_TAG = tag
        break


def _auto_orient(img: Image.Image) -> Image.Image:
    """根据 EXIF Orientation 自动旋转图片."""
    if not _EXIF_ORIENTATION_TAG or not hasattr(img, "_getexif"):
        return img
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation = exif.get(_EXIF_ORIENTATION_TAG, 1)
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: 180,
            4: Image.FLIP_TOP_BOTTOM,
            5: (Image.FLIP_LEFT_RIGHT, 90),
            6: 270,
            7: (Image.FLIP_LEFT_RIGHT, 270),
            8: 90,
        }.get(orientation)
        if method is None:
            return img

        if isinstance(method, tuple):
            # 5、7 需要先镜像再旋转
            img = img.transpose(method[0])
            img = img.rotate(method[1], expand=True)
        elif isinstance(method, int):
            img = img.rotate(method, expand=True)
        else:
            img = img.transpose(method)
        return img
    except Exception:
        return img


# --------- 主逻辑 ---------- #
def resize_images(folder: Path, max_size: int = 1080) -> None:
    """遍历并压缩 folder 中的图片."""
    if not folder.is_dir():
        print(f"❌ 路径不存在或不是文件夹: {folder}")
        sys.exit(1)

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    processed = 0
    skipped  = 0

    for p in folder.rglob("*"):
        if p.suffix.lower() not in img_exts:
            continue
        if p.name.startswith("m_"):
            skipped += 1
            continue

        try:
            with Image.open(p) as im:
                im = _auto_orient(im)

                # 已满足尺寸要求时直接拷贝
                if max(im.size) <= max_size:
                    out_path = p.with_name("m_" + p.name)
                    im.save(out_path, optimize=True, quality=90)
                    processed += 1
                    continue

                # 根据较长边等比缩放
                im.thumbnail((max_size, max_size), Image.LANCZOS)

                out_path = p.with_name("m_" + p.name)
                im.save(out_path, optimize=True, quality=90)
                processed += 1
                print(f"✅ {p.name} → {out_path.name}   {im.size}")

        except Exception as e:
            print(f"⚠️ 处理失败 {p}: {e}")

    print(f"\n完成！处理 {processed} 张，跳过 {skipped} 张。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量压缩图片到最长边 ≤ 1080 像素")
    parser.add_argument(
        "folder",
        type=Path,
        help="目标文件夹路径",
    )
    parser.add_argument(
        "--max", type=int, default=1080, help="最长边限制 (默认 1080)"
    )
    args = parser.parse_args()
    resize_images(args.folder, max_size=args.max)
