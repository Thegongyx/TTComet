#!/usr/bin/env python3
"""
图标设计草图 v2
===============
改进点（对照 v1）：
  - 球改成真正的球面着色（numpy 逐像素），高光在左上、右下是冷色环境光，不再像"凹陷"
  - 彗尾加亮、加长，并叠一层整体柔光，小尺寸下也能看出"拖尾"
  - 增加速度细线，强化运动感

用法：
    webui\.venv\Scripts\python.exe webui\design\make_icons.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
S = 1024
FINAL = 256

BG_TOP = (26, 41, 66)
BG_BOTTOM = (38, 62, 90)
ACCENT_BLUE = (114, 135, 254)
ACCENT_CYAN = (69, 243, 255)
BALL_WHITE = (247, 250, 255)
PADDLE_RED = (219, 72, 64)
PADDLE_DARK = (24, 28, 38)
FELT_GREEN = (26, 130, 84)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))


def rounded_bg(size: int, radius_ratio: float = 0.225) -> Image.Image:
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

    # 顶部柔光，避免死板
    shine = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shine).ellipse(
        [-size * 0.4, -size * 0.9, size * 1.4, size * 0.42],
        fill=(255, 255, 255, 18))
    shine.putalpha(Image.composite(shine.getchannel("A"), Image.new("L", (size, size), 0), mask))
    return Image.alpha_composite(img, shine)


def sphere(size: int, base=BALL_WHITE, light=(-0.42, -0.52, 0.74)) -> Image.Image:
    """逐像素球面着色，返回带 alpha 的球贴图（边长 size）。"""
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

    # 右下边缘冷色环境光
    rim = np.clip((-nx * 0.5 + ny * 0.5), 0, 1) * (1 - nz) ** 0.6
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(3):
        rgb[..., i] = base[i] * shade + (70 * rim)
    rgb = np.clip(rgb, 0, 255)

    # 边缘抗锯齿
    alpha = np.clip((1.0 - r2) * r * 1.5, 0.0, 1.0) * inside
    alpha = np.clip(alpha * 255, 0, 255)

    out = np.dstack([rgb, alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def radial_glow(size: int, color, peak_alpha: int = 80, power: float = 2.2) -> Image.Image:
    """径向渐变光晕（逐像素），避免"椭圆高斯模糊"在贴图边界留下方形接缝。"""
    r = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt(((xx - r + 0.5) / r) ** 2 + ((yy - r + 0.5) / r) ** 2)
    a = (np.clip(1.0 - d, 0.0, 1.0) ** power) * peak_alpha
    buf = np.zeros((size, size, 4), dtype=np.uint8)
    buf[..., 0], buf[..., 1], buf[..., 2] = color
    buf[..., 3] = a.astype(np.uint8)
    return Image.fromarray(buf, "RGBA")


def paste_glow(img: Image.Image, sprite: Image.Image, center, glow_color, glow_scale=1.7,
               glow_alpha=70, blur_ratio=0.5):
    """先贴柔光，再贴实体。"""
    w, h = sprite.size
    gs = int(max(w, h) * glow_scale)
    glow = radial_glow(gs, glow_color, glow_alpha, 2.2)
    img.alpha_composite(glow, (int(center[0] - gs / 2), int(center[1] - gs / 2)))
    img.alpha_composite(sprite, (int(center[0] - w / 2), int(center[1] - h / 2)))
    return img


def dot(img: Image.Image, center, radius, color, alpha, blur=0):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x, y = center
    ImageDraw.Draw(layer).ellipse(
        [x - radius, y - radius, x + radius, y + radius], fill=(*color, int(alpha)))
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(img, layer)


def comet_trail(img, head, tail_vec, count, r0, color_a, color_b):
    """渐隐彗尾：一串由大到小、由亮到暗的圆（先画远端的，保证近端压在上面）。"""
    for i in range(count, 0, -1):
        t = i / (count + 1)
        cx = head[0] - tail_vec[0] * t
        cy = head[1] - tail_vec[1] * t
        r = r0 * (1 - 0.66 * t)
        col = lerp(color_a, color_b, t ** 0.8)
        alpha = int(235 * (1 - t) ** 0.85)
        img = dot(img, (cx, cy), r, col, alpha, r * 0.35)
    return img


def ball_sprite(diameter):
    return sphere(diameter)


# ---------------------------------------------------------------- A：彗星球
def make_a():
    img = rounded_bg(S)
    head = (S * 0.585, S * 0.425)          # 球偏中心，视觉重心稳
    tail = (S * 0.425, S * 0.290)          # 彗尾朝左上
    # 彗尾：细长渐隐，末端更淡，避免糊成一团
    for i in range(12, 0, -1):
        t = i / 13
        cx = head[0] - tail[0] * t
        cy = head[1] - tail[1] * t
        r = S * (0.072 * (1 - 0.72 * t))
        col = lerp(ACCENT_CYAN, ACCENT_BLUE, t ** 0.75)
        alpha = int(225 * (1 - t) ** 1.35)
        img = dot(img, (cx, cy), r, col, alpha, r * 0.55)
    d = int(S * 0.245)
    img = paste_glow(img, ball_sprite(d), head, ACCENT_CYAN, 2.2, 90, 0.5)
    return img


# ---------------------------------------------------------------- A2：彗星 + 速度线
def make_a2():
    img = rounded_bg(S)
    head = (S * 0.70, S * 0.33)
    # 两条速度细线
    lines = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for k, (dy, ln, al) in enumerate(((0.075, 0.30, 90), (0.135, 0.22, 60))):
        y = head[1] + S * dy
        ld.line([(S * (0.20 + k * 0.06), y), (S * (0.20 + k * 0.06 + ln), y)],
                fill=(*ACCENT_CYAN, al), width=int(S * 0.012))
    lines = lines.filter(ImageFilter.GaussianBlur(S * 0.004))
    img = Image.alpha_composite(img, lines)

    img = comet_trail(img, head, (S * 0.46, S * 0.26), 8, S * 0.078, ACCENT_CYAN, ACCENT_BLUE)
    d = int(S * 0.225)
    img = paste_glow(img, ball_sprite(d), head, ACCENT_CYAN, 2.0, 80, 0.5)
    return img


# ---------------------------------------------------------------- B：球拍 + 彗星
def make_b():
    img = rounded_bg(S)
    # 拍柄
    pd = ImageDraw.Draw(img)
    hx0, hy0 = S * 0.40, S * 0.62
    hx1, hy1 = S * 0.28, S * 0.83
    pd.line([(hx0, hy0), (hx1, hy1)], fill=(*PADDLE_DARK, 255), width=int(S * 0.085))
    # 拍面
    cx, cy = S * 0.47, S * 0.47
    rx, ry = S * 0.215, S * 0.245
    pd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(*PADDLE_RED, 255))
    pd.ellipse([cx - rx * 0.74, cy - ry * 0.74, cx + rx * 0.74, cy + ry * 0.74],
               fill=(196, 54, 48, 255))
    # 拍面高光
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse(
        [cx - rx * 0.55, cy - ry * 0.62, cx + rx * 0.15, cy - ry * 0.05],
        fill=(255, 255, 255, 26))
    img = Image.alpha_composite(img, hl)

    head = (S * 0.70, S * 0.30)
    img = comet_trail(img, head, (S * 0.34, S * 0.26), 7, S * 0.072, ACCENT_CYAN, ACCENT_BLUE)
    img = paste_glow(img, ball_sprite(int(S * 0.205)), head, ACCENT_CYAN, 2.0, 80, 0.5)
    return img


# ---------------------------------------------------------------- C：球台 + 彗星
def make_c():
    img = rounded_bg(S)
    td = ImageDraw.Draw(img)
    far = [(S * 0.25, S * 0.50), (S * 0.75, S * 0.50)]
    near = [(S * 0.12, S * 0.79), (S * 0.88, S * 0.79)]
    td.polygon([far[0], far[1], near[1], near[0]], fill=(*FELT_GREEN, 255))
    td.line([far[0], far[1], near[1], near[0], far[0]],
            fill=(255, 255, 255, 240), width=int(S * 0.017))
    td.line([(S * 0.5, S * 0.50), (S * 0.5, S * 0.79)],
            fill=(255, 255, 255, 110), width=int(S * 0.011))
    td.line([(S * 0.5, S * 0.487), (S * 0.5, S * 0.66)],
            fill=(228, 238, 250, 235), width=int(S * 0.017))

    head = (S * 0.665, S * 0.265)
    img = comet_trail(img, head, (S * 0.30, S * 0.18), 7, S * 0.064, ACCENT_CYAN, ACCENT_BLUE)
    img = paste_glow(img, ball_sprite(int(S * 0.185)), head, ACCENT_CYAN, 2.0, 80, 0.5)
    return img


def save(img: Image.Image, name: str) -> Image.Image:
    small = img.resize((FINAL, FINAL), Image.LANCZOS)
    small.save(os.path.join(OUT_DIR, name))
    return small


def main():
    variants = [
        ("A  彗星球", make_a()),
        ("A2 彗星 + 速度线", make_a2()),
        ("B  球拍 + 彗星", make_b()),
        ("C  球台 + 彗星", make_c()),
    ]
    smalls = [save(img, f"icon_{i}.png") for i, (_, img) in enumerate(variants)]

    pad, cell = 28, 300
    sheet = Image.new("RGBA", (cell * len(smalls) + pad, 470), (14, 18, 26, 255))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        font_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        font = font_s = ImageFont.load_default()

    for i, (label, small) in enumerate(zip([v[0] for v in variants], smalls)):
        x = pad // 2 + i * cell
        sheet.alpha_composite(small, (x, 20))
        d.text((x + 6, 286), label, font=font, fill=(230, 238, 250, 255))
        for j, sz in enumerate((48, 32, 16)):
            s = small.resize((sz, sz), Image.LANCZOS)
            sheet.alpha_composite(s, (x + 6 + j * 56, 320))
        d.text((x + 6, 390), "16 / 32 / 48 px 可读性", font=font_s, fill=(140, 155, 180, 255))

    sheet.save(os.path.join(OUT_DIR, "candidates.png"))
    print("已生成:")
    for n in ("candidates.png",) + tuple(f"icon_{i}.png" for i in range(len(variants))):
        print(f"  {os.path.join(OUT_DIR, n)}")


if __name__ == "__main__":
    main()
