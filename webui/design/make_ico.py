#!/usr/bin/env python3
"""
生成最终图标 TTComet.ico
========================
- 大尺寸（48+）用完整版（细节彗尾）
- 小尺寸（16/24/32）用简化版（球更大、尾更短更实），避免缩小时糊成一团
- ICO 直接手工拼装 PNG 帧（Vista+ 支持，Win10/11 完全没问题），
  这样可以给每个尺寸用不同的源图

用法：
    webui\.venv\Scripts\python.exe webui\design\make_ico.py
输出：
    webui\packaging\TTComet.ico         （exe / 安装包用）
    webui\app\static\favicon.ico        （浏览器标签页用）
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
S = 512                      # 每个尺寸都从这个分辨率的母版缩下来

BG_TOP = (26, 41, 66)
BG_BOTTOM = (38, 62, 90)
ACCENT_BLUE = (114, 135, 254)
ACCENT_CYAN = (69, 243, 255)
BALL_WHITE = (247, 250, 255)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))


def rounded_bg(size, radius_ratio=0.225):
    grad = Image.new("RGB", (1, size))
    px = grad.load()
    for y in range(size):
        px[0, y] = lerp(BG_TOP, BG_BOTTOM, y / max(1, size - 1))
    grad = grad.resize((size, size))

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=255)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)

    shine = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shine).ellipse(
        [-size * 0.4, -size * 0.9, size * 1.4, size * 0.42], fill=(255, 255, 255, 18))
    shine.putalpha(Image.composite(shine.getchannel("A"), Image.new("L", (size, size), 0), mask))
    return Image.alpha_composite(img, shine)


def sphere(size, base=BALL_WHITE, light=(-0.42, -0.52, 0.74)):
    r = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xx - r + 0.5) / r
    ny = (yy - r + 0.5) / r
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    L = np.array(light, dtype=np.float32)
    L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
    shade = 0.52 + 0.62 * diff
    rim = np.clip((-nx * 0.5 + ny * 0.5), 0, 1) * (1 - nz) ** 0.6

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(3):
        rgb[..., i] = base[i] * shade + 70 * rim
    rgb = np.clip(rgb, 0, 255)
    alpha = np.clip(np.clip((1.0 - r2) * r * 1.5, 0.0, 1.0) * inside * 255, 0, 255)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def radial_glow(size, color, peak_alpha=90, power=2.2):
    r = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt(((xx - r + 0.5) / r) ** 2 + ((yy - r + 0.5) / r) ** 2)
    a = (np.clip(1.0 - d, 0.0, 1.0) ** power) * peak_alpha
    buf = np.zeros((size, size, 4), dtype=np.uint8)
    buf[..., 0], buf[..., 1], buf[..., 2] = color
    buf[..., 3] = a.astype(np.uint8)
    return Image.fromarray(buf, "RGBA")


def dot(img, center, radius, color, alpha, blur=0):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x, y = center
    ImageDraw.Draw(layer).ellipse(
        [x - radius, y - radius, x + radius, y + radius], fill=(*color, int(alpha)))
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(img, layer)


def paste_glow(img, sprite, center, color, scale=2.2, alpha=90):
    w = sprite.size[0]
    gs = int(w * scale)
    img.alpha_composite(radial_glow(gs, color, alpha), (int(center[0] - gs / 2), int(center[1] - gs / 2)))
    img.alpha_composite(sprite, (int(center[0] - w / 2), int(center[1] - w / 2)))
    return img


def build(compact: bool) -> Image.Image:
    """compact=True 用于 16/24/32px：球更大、尾巴更短更实。"""
    img = rounded_bg(S)
    if compact:
        head = (S * 0.545, S * 0.520)
        tail = (S * 0.26, S * 0.20)
        count, r0, span = 7, S * 0.090, 0.60
        ball_d = int(S * 0.500)
    else:
        head = (S * 0.545, S * 0.500)      # 球更居中
        tail = (S * 0.470, S * 0.345)      # 彗尾更长，端点在圆角安全区内
        count, r0, span = 15, S * 0.074, 1.0
        ball_d = int(S * 0.245)

    for i in range(count, 0, -1):
        t = i / (count + 1)
        cx = head[0] - tail[0] * span * t
        cy = head[1] - tail[1] * span * t
        r = r0 * (1 - (0.72 if not compact else 0.45) * t)
        col = lerp(ACCENT_CYAN, ACCENT_BLUE, t ** 0.75)
        a = int(235 * (1 - t) ** (1.35 if not compact else 0.9))
        img = dot(img, (cx, cy), r, col, a, r * 0.5)

    img = paste_glow(img, sphere(ball_d), head, ACCENT_CYAN,
                     scale=2.0 if compact else 2.2,
                     alpha=110 if compact else 90)
    return img


def make_ico(frames: dict[int, Image.Image], out_path: str) -> None:
    """手工拼装 ICO（每帧 PNG 压缩，Win10/11 支持所有尺寸）。"""
    sizes = sorted(frames)
    blobs = []
    for s in sizes:
        buf = io.BytesIO()
        frames[s].save(buf, format="PNG", optimize=True)
        blobs.append(buf.getvalue())

    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for s, blob in zip(sizes, blobs):
        w = 0 if s >= 256 else s
        entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    with open(out_path, "wb") as f:
        f.write(header + entries + b"".join(blobs))


def main():
    big = build(compact=False)
    small = build(compact=True)

    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = {}
    for s in sizes:
        src = small if s <= 32 else big
        frames[s] = src.resize((s, s), Image.LANCZOS).convert("RGBA")

    ico_packaging = os.path.join(ROOT, "packaging", "TTComet.ico")
    ico_static = os.path.join(ROOT, "app", "static", "favicon.ico")
    make_ico(frames, ico_packaging)
    make_ico(frames, ico_static)

    # 256 的 PNG 也留一份，方便做宣传图/商店素材
    big.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "TTComet-256.png"))

    # 预览图：各尺寸平铺在深色底上
    pad, cell = 24, 150
    sheet = Image.new("RGBA", (cell * len(sizes) + pad, 200), (14, 18, 26, 255))
    x = pad // 2
    for s in sizes:
        sheet.alpha_composite(frames[s], (x, 24))
        x += cell
    sheet.save(os.path.join(HERE, "ico_preview.png"))

    print("生成完成:")
    for p in (ico_packaging, ico_static,
              os.path.join(HERE, "TTComet-256.png"),
              os.path.join(HERE, "ico_preview.png")):
        print(f"  {p}  ({os.path.getsize(p) // 1024} KB)")


if __name__ == "__main__":
    main()
