"""Shadow/Highlight for video, audio copied.

Pipeline: ffmpeg strong+weak cleanup (16-bit, blended by brightness) -> motion-compensated multi-frame
merge -> tone engine (llf default: fast local Laplacian on max-RGB; dual: DUAL + Mertens) -> sharpen -> x264.
Usage: unshadow.py IN OUT [--detail 0.4 --range 0.5 --lift 0.4 --black 0.05 --merge 3 --sharpen 0.7]
IN may be a video or an image (.png/.jpg; no merge or cleanup for images).
"""
import argparse
import functools
import multiprocessing as mp
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dual import dual  # noqa: E402
from llf import enhance  # noqa: E402
from temporal import merge  # noqa: E402

# Two cleanups decoded side by side (hstack), all at 16 bit: the strong one hides codec blocks in near-black,
# the weak one keeps midtone detail the strong one softens. Blended per pixel by brightness (split_blend).
# Temporal denoise is left to merge().
FILTERGRAPH = ("[0:v]format=yuv420p16le,split[a][b];"
               "[a]deblock=filter=strong:block=8,hqdn3d=2:1.5:0:0,"
               "deband=1thr=0.02:2thr=0.04:3thr=0.04:range=16:blur=1[c];"
               "[b]deblock=filter=weak:block=8[d];[c][d]hstack[v]")
DARK_LO, DARK_HI = 0.08, 0.20  # strong-clean below LO, weak-clean above HI (max-RGB, gamma-encoded)
CHUNK = 12  # frames per worker task


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    w, h, fps = r.stdout.strip().split(",")[:3]
    return int(w), int(h), fps


def split_blend(both, w):
    """both: float32 (h, 2w, 3) = [strong-clean | weak-clean] -> one frame."""
    strong, weak = both[:, :w], both[:, w:]
    t = cv2.GaussianBlur(strong.max(axis=2), (0, 0), 2)
    m = np.clip((t - DARK_LO) / (DARK_HI - DARK_LO), 0, 1)
    m = (m * m * (3 - 2 * m))[..., None]
    return strong * (1 - m) + weak * m


def sharpen(img_u8, amount, radius=1.0):
    """RGB unsharp for the dual engine; llf sharpens its scalar intensity instead (colour-safe)."""
    if not amount:
        return img_u8
    f = img_u8.astype(np.float32)
    return np.clip(f + amount * (f - cv2.GaussianBlur(f, (0, 0), radius)) + 0.5, 0, 255).astype(np.uint8)


def dual_f(frame, sharpen_amount=0.0, **kw):
    return sharpen(dual((frame * 255 + 0.5).astype(np.uint8), **kw), sharpen_amount)


def process_one(frames, center, tone, radius):
    """frames: float32 BGR window; returns uint8 BGR for frames[center]."""
    f = merge(frames, center) if radius and len(frames) > 1 else frames[center]
    return tone(f)


_CFG = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg
    cv2.setNumThreads(1)  # one task per process; avoid oversubscribing cores


def _work(task):
    """task: (window buffers, first output index in window, count). Window carries `radius` margin."""
    bufs, first, count = task
    tone, radius, h, w = _CFG
    fr = [split_blend(np.frombuffer(b, np.uint16).reshape(h, 2 * w, 3).astype(np.float32) / 65535.0, w)
          for b in bufs]
    out = []
    for c in range(first, first + count):
        lo, hi = max(0, c - radius), min(len(fr), c + radius + 1)
        out.append(process_one(fr[lo:hi], c - lo, tone, radius).tobytes())
    return out


def run_video(src, dst, tone, radius=3, start=None, duration=None, workers=None, progress=None):
    """progress: optional callable(frames_done), called after every written frame."""
    w, h, fps = probe(src)
    seek = (["-ss", str(start)] if start else []) + (["-t", str(duration)] if duration else [])
    dec = subprocess.Popen(["ffmpeg", "-v", "error", *seek, "-i", src, "-filter_complex", FILTERGRAPH,
                            "-map", "[v]", "-f", "rawvideo", "-pix_fmt", "bgr48le", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", fps, "-i", "-", *seek, "-i", src,
                            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "17",
                            "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", dst],
                           stdin=subprocess.PIPE)
    size = 2 * w * h * 6

    def read_all():
        while True:
            b = dec.stdout.read(size)
            if len(b) < size:
                return
            yield b

    def tasks():
        # Sliding chunks of CHUNK frames, each with `radius` frames of context on both sides.
        buf, base = [], 0  # base = absolute index of buf[0]
        nxt = 0            # next absolute frame index to emit
        for b in read_all():
            buf.append(b)
            if base + len(buf) - nxt >= CHUNK + radius:
                lo = max(base, nxt - radius)
                yield buf[lo - base:], nxt - lo, CHUNK
                nxt += CHUNK
                drop = max(0, nxt - radius - base)
                buf, base = buf[drop:], base + drop
        rem = base + len(buf) - nxt
        if rem > 0:
            lo = max(base, nxt - radius)
            yield buf[lo - base:], nxt - lo, rem

    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    n = 0
    try:
        with mp.Pool(workers, initializer=_init_worker, initargs=((tone, radius, h, w),)) as pool:
            for outs in pool.imap(_work, tasks()):  # imap keeps chunk order
                for o in outs:
                    enc.stdin.write(o)
                    n += 1
                    if progress:
                        progress(n)
                    if n % 100 == 0:
                        print(f"{n} frames", file=sys.stderr)
    finally:
        enc.stdin.close()
        dec.wait()
        enc.wait()
    if dec.returncode or enc.returncode:
        sys.exit(f"ffmpeg failed: dec={dec.returncode} enc={enc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--engine", default="llf", choices=["llf", "dual"])
    ap.add_argument("--detail", type=float, default=0.4, help="llf alpha, lower = more local detail")
    ap.add_argument("--range", type=float, default=0.5, help="llf beta, lower = stronger shadow/highlight compression")
    ap.add_argument("--lift", type=float, default=0.4, help="llf global shadow lift 0..1")
    ap.add_argument("--gamma", type=float, default=0.6, help="dual lift strength, lower = stronger")
    ap.add_argument("--clarity", type=float, default=0.4, help="dual local contrast")
    ap.add_argument("--black", type=float, default=0.05, help="below this stays black (hides codec blocks)")
    ap.add_argument("--sat", type=float, default=1.0)
    ap.add_argument("--merge", type=int, default=3, help="frames merged on each side (0 = off)")
    ap.add_argument("--sharpen", type=float, default=0.7, help="fine sharpening after merge")
    ap.add_argument("--start", type=float)
    ap.add_argument("--duration", type=float)
    a = ap.parse_args()
    if a.engine == "llf":
        tone = functools.partial(enhance, alpha=a.detail, beta=a.range, lift=a.lift, black=a.black, sat=a.sat,
                                 sharpen=a.sharpen)
    else:
        tone = functools.partial(dual_f, gamma=a.gamma, black=a.black, clarity=a.clarity, sat=a.sat, mode="mix",
                                 sharpen_amount=a.sharpen)
    if a.src.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(a.src, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
        cv2.imwrite(a.dst, tone(img))
    else:
        run_video(a.src, a.dst, tone, radius=a.merge, start=a.start, duration=a.duration)


if __name__ == "__main__":
    main()
