"""Generate a synthetic low-dynamic-range 'concert' clip for demos and screenshots (no real footage).

Scene is rendered with HDR values, then exposed so shadows crush and lights clip, plus sensor noise and a
low-bitrate H.264 encode -- the same failure modes the tool targets.
Usage: python3.11 tools/make_demo.py [samples/demo_stage.mp4] [--seconds 6]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 576, 1024, 30
rng = np.random.default_rng(7)


def noise_field(scale, size=(H, W)):
    """Smooth value noise in [0,1]: random grid upsampled bicubically."""
    small = rng.random((max(2, size[0] // scale), max(2, size[1] // scale))).astype(np.float32)
    return cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# ---- static layers (linear light, BGR) ----
folds = 0.5 + 0.5 * np.sin(xx / 9.0 + 3 * noise_field(64)) * noise_field(24)
wall = (0.012 + 0.02 * folds + 0.008 * noise_field(4))[..., None] * np.array([1.0, 1.0, 0.95], np.float32)
symbol = np.zeros((H, W), np.float32)
for (x0, y0, x1, y1) in [(90, 180, 200, 520), (200, 180, 90, 520), (60, 350, 230, 350), (145, 150, 145, 560)]:
    cv2.line(symbol, (x0, y0), (x1, y1), 1.0, 4, cv2.LINE_AA)
cv2.ellipse(symbol, (300, 330), (230, 200), 0, 200, 320, 1.0, 3, cv2.LINE_AA)
wall += (0.12 * cv2.GaussianBlur(symbol, (0, 0), 1.2))[..., None]

# performer silhouette with faint dress texture (the detail we want recovered)
body = np.zeros((H, W), np.float32)
cv2.ellipse(body, (300, 452), (25, 31), 0, 0, 360, 1.0, -1, cv2.LINE_AA)               # head
cv2.fillPoly(body, [np.array([                                                         # long hair
    [272, 440], [328, 440], [340, 500], [336, 560], [264, 560], [260, 500]])], 1.0, cv2.LINE_AA)
cv2.fillPoly(body, [np.array([                                                         # shoulders, arms, dress
    [288, 480], [312, 480], [346, 500], [356, 520], [372, 640], [362, 700], [352, 640], [344, 560],
    [350, 640], [384, 780], [414, 920], [186, 920], [216, 780], [250, 640], [256, 560], [248, 640],
    [238, 700], [228, 640], [244, 520], [254, 500]])], 1.0, cv2.LINE_AA)
body = cv2.GaussianBlur(body, (0, 0), 1.2)
dress = 0.012 + 0.010 * (0.5 + 0.5 * np.sin(yy / 5.0 + 4 * noise_field(40))) * noise_field(6)
rim = np.clip(cv2.GaussianBlur(body, (0, 0), 3) - cv2.GaussianBlur(body, (0, 0), 1.0), 0, 1) * 6

# drum kit + mic stand (dark metal with small glints)
kit = np.zeros((H, W), np.float32)
for (cx, cy, ax, ay) in [(110, 780, 70, 14), (470, 760, 60, 12), (130, 860, 50, 40)]:
    cv2.ellipse(kit, (cx, cy), (ax, ay), 0, 0, 360, 1.0, 2, cv2.LINE_AA)
cv2.line(kit, (110, 780), (110, 960), 1.0, 2, cv2.LINE_AA)
cv2.line(kit, (470, 760), (470, 960), 1.0, 2, cv2.LINE_AA)
cv2.line(kit, (430, 560), (440, 960), 1.0, 2, cv2.LINE_AA)
cv2.line(kit, (430, 560), (360, 520), 1.0, 2, cv2.LINE_AA)

# crowd heads in the foreground (near black, a little texture)
crowd = np.zeros((H, W), np.float32)
for cx, cy, r in [(60, 1000, 120), (230, 1040, 110), (420, 1010, 130), (560, 980, 90)]:
    cv2.circle(crowd, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
crowd = cv2.GaussianBlur(crowd, (0, 0), 6)
hair = 0.0015 + 0.002 * noise_field(3)
crowd_rim = np.clip(cv2.GaussianBlur(crowd, (0, 0), 4) - crowd, 0, 1) * 3  # backlight on head edges

haze_a, haze_b = noise_field(90, (H * 2, W)), noise_field(40, (H * 2, W))


def beam(t, x_top, angle, width, color, power):
    """Soft cone from the top edge, swinging over time; returns BGR light."""
    a = angle + 0.18 * np.sin(t * 0.7 + x_top)
    dx = xx - (x_top + np.tan(a) * yy)
    w = width * (0.25 + yy / H)
    cone = np.exp(-(dx / w) ** 2) * (1.2 - 0.6 * yy / H)
    return cone[..., None] * np.array(color, np.float32) * power


def frame(t):
    off = int(t * 40) % H
    haze = (0.6 * haze_a[off:off + H] + 0.4 * haze_b[H - off:2 * H - off])[..., None]
    light = (beam(t, 380, -0.25, 60, (1.0, 0.95, 0.9), 2.4)           # white spot behind performer
             + beam(t, 80, 0.30, 70, (1.0, 0.75, 0.2), 1.6)           # cyan/blue wash (BGR)
             + beam(t, 520, -0.45, 45, (0.9, 0.35, 0.15), 1.3))
    img = wall * (0.4 + 0.5 * light) + light * haze * 0.9
    hot = np.exp(-(((xx - 380) / 26) ** 2 + ((yy - 40) / 22) ** 2))[..., None] * 8  # lamp, far over white
    img += hot
    sway = int(6 * np.sin(t * 1.3))
    b = np.roll(body, sway, axis=1)[..., None]
    img = img * (1 - b) + b * (dress[..., None] + np.roll(rim, sway, axis=1)[..., None] * light * 0.6)
    img += kit[..., None] * (0.03 + 0.25 * light)
    c = crowd[..., None]
    img = img * (1 - c) + c * hair[..., None] + crowd_rim[..., None] * light * 0.08
    # Camera: underexposed, hard clip, sensor noise, sRGB-ish gamma.
    img = img * 0.55 + rng.normal(0, 0.004, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="samples/demo_stage.mp4")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "30", "-preset", "fast",
                            "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    try:
        for i in range(int(a.seconds * FPS)):
            enc.stdin.write(frame(i / FPS).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode:
        sys.exit(f"ffmpeg failed ({enc.returncode})")
    print(a.out)


if __name__ == "__main__":
    main()
