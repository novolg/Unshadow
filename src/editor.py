"""Local Shadow/Highlight video editor: browser UI with sliders, before/after preview, clip player, export.

Run:  python3.11 src/editor.py  [--port 8765]  then open http://127.0.0.1:8765  (or double-click Unshadow.command)
Videos: test_videos/ plus any file opened via the macOS file dialog. Exports go to output/.
"""
import argparse
import base64
import functools
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llf import enhance  # noqa: E402
from temporal import merge  # noqa: E402
from unshadow import FILTERGRAPH, probe, run_video, split_blend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = Path(os.environ.get("UNSHADOW_VIDEO_DIR", ROOT / "test_videos"))
OUT_DIR = ROOT / "output"
CLIP_DIR = OUT_DIR / ".preview"
PAGE = Path(__file__).with_name("editor.html")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm")
RADIUS = 3  # merged frames on each side, same as export default

log = logging.getLogger("editor")

_lock = threading.Lock()
_cache = {}  # (path, t, merge) -> (before_u8, merged_float)
_opened = []  # files picked via the dialog this session
# One background job at a time: a full export or a short preview clip.
_job = {"state": "idle", "kind": "", "done": 0, "total": 0, "out": "", "error": "", "stamp": 0}


def list_videos():
    found = sorted(str(p) for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXT) if VIDEO_DIR.is_dir() else []
    return _opened + [v for v in found if v not in _opened]


def pick_file():
    """Native macOS open dialog, shown over the frontmost app (the browser). Returns path or None if cancelled."""
    script = ('tell application (path to frontmost application as text) to POSIX path of '
              '(choose file with prompt "Open video" of type {"public.movie"})')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode:
        if "-128" in r.stderr:  # user cancelled
            return None
        raise ValueError(f"file dialog failed: {r.stderr.strip()}")
    path = r.stdout.strip()
    if path not in _opened:
        _opened.insert(0, path)
    return path


def video_info(path):
    w, h, fps = probe(path)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    num, den = (fps.split("/") + ["1"])[:2]
    return {"width": w, "height": h, "fps": float(num) / float(den), "duration": float(r.stdout.strip())}


def decode_window(path, t, fps, w, h, radius):
    """Decode 2*radius+1 frames centred on t through the export filtergraph -> list of float frames."""
    n = 2 * radius + 1
    start = max(0.0, t - radius / fps)
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path,
                        "-filter_complex", FILTERGRAPH, "-map", "[v]", "-frames:v", str(n),
                        "-f", "rawvideo", "-pix_fmt", "bgr48le", "-"], capture_output=True, check=True)
    size = 2 * w * h * 6
    k = len(r.stdout) // size
    if k == 0:
        raise ValueError(f"no frames decoded at t={t:.2f}s")
    raw = [np.frombuffer(r.stdout[i * size:(i + 1) * size], np.uint16).reshape(h, 2 * w, 3) for i in range(k)]
    center = min(k - 1, int(round((t - start) * fps)))
    # "Before" = weak-clean half (~ the original decode) of the centre frame.
    before = (raw[center][:, w:].astype(np.float32) / 257 + 0.5).astype(np.uint8)
    frames = [split_blend(x.astype(np.float32) / 65535.0, w) for x in raw]
    return before, frames, center


def preview(path, t, p):
    info = video_info(path)
    key = (path, round(t, 2), bool(p["merge"]))
    with _lock:
        hit = _cache.get(key)
    if hit is None:
        before, frames, c = decode_window(path, t, info["fps"], info["width"], info["height"], RADIUS)
        merged = merge(frames, c) if p["merge"] else frames[c]
        hit = (before, merged)
        with _lock:
            if len(_cache) > 8:
                _cache.clear()
            _cache[key] = hit
    before, merged = hit
    return before, enhance(merged, **tone_kwargs(p))


def tone_kwargs(p):
    return dict(alpha=p["detail"], beta=p["range"], lift=p["lift"], black=p["black"], sat=p["sat"],
                sharpen=p["sharpen"])


def params_from(q):
    def f(name, default):
        try:
            return float(q.get(name, [default])[0])
        except ValueError:
            return default
    return {"lift": f("lift", 0.4), "detail": f("detail", 0.4), "range": f("range", 0.5),
            "black": f("black", 0.05), "sat": f("sat", 1.0), "sharpen": f("sharpen", 0.7),
            "merge": f("merge", 1) > 0, "t": f("t", 0.0), "seconds": f("seconds", 5.0)}


def jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _run_job(kind, path, p):
    try:
        info = video_info(path)
        tone = functools.partial(enhance, **tone_kwargs(p))
        radius = RADIUS if p["merge"] else 0

        def prog(n):
            _job["done"] = n

        if kind == "export":
            OUT_DIR.mkdir(exist_ok=True)
            dst = OUT_DIR / f"{Path(path).stem}_unshadow.mp4"
            _job.update(total=int(info["duration"] * info["fps"]), out=str(dst))
            run_video(path, str(dst), tone, radius=radius, progress=prog)
        else:
            CLIP_DIR.mkdir(parents=True, exist_ok=True)
            t, secs = p["t"], min(p["seconds"], max(0.5, info["duration"] - p["t"]))
            _job.update(total=int(secs * info["fps"]), out="clip")
            # Before: same window, re-encoded so both players start on the same frame.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{secs:.3f}", "-i", path,
                            "-an", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                            str(CLIP_DIR / "before.mp4")], check=True, capture_output=True)
            run_video(path, str(CLIP_DIR / "after.mp4"), tone, radius=radius, start=t, duration=secs, progress=prog)
        _job.update(state="done", stamp=int(time.time() * 1000))
        log.info("%s done: %s", kind, _job["out"])
    except (Exception, SystemExit) as e:  # SystemExit: run_video reports ffmpeg failure that way
        log.exception("%s failed", kind)
        _job.update(state="error", error=str(e))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _video(self, q):
        path = q.get("video", [""])[0]
        if not Path(path).is_file():
            raise ValueError(f"video not found: {path}")
        return path

    def _media(self, name):
        """Serve a preview clip with HTTP Range support (Safari's <video> requires it)."""
        if name not in ("before", "after"):
            self._send(404, {"error": "not found"})
            return
        f = CLIP_DIR / f"{name}.mp4"
        size = f.stat().st_size
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
        end = min(end, size - 1)
        with open(f, "rb") as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self.wfile.write(data)

    def _start_job(self, kind, q):
        if _job["state"] == "running":
            self._send(409, {"error": f"{_job['kind']} already running"})
            return
        path, p = self._video(q), params_from(q)
        # Mark running before the thread starts so an immediate /api/status poll never sees "idle".
        _job.update(state="running", kind=kind, done=0, total=0, out="", error="")
        threading.Thread(target=_run_job, args=(kind, path, p), daemon=True).start()
        self._send(200, {"ok": True})

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/videos":
                self._send(200, {"videos": list_videos()})
            elif u.path == "/api/open":
                self._send(200, {"path": pick_file()})
            elif u.path == "/api/info":
                self._send(200, video_info(self._video(q)))
            elif u.path == "/api/preview":
                t0 = time.perf_counter()
                p = params_from(q)
                before, after = preview(self._video(q), p["t"], p)
                self._send(200, {"before": jpeg_b64(before), "after": jpeg_b64(after),
                                 "ms": int((time.perf_counter() - t0) * 1000)})
            elif u.path == "/api/export":
                self._start_job("export", q)
            elif u.path == "/api/clip":
                self._start_job("clip", q)
            elif u.path == "/api/status":
                self._send(200, _job)
            elif u.path.startswith("/media/"):
                self._media(u.path.rsplit("/", 1)[-1].removesuffix(".mp4"))
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning("request %s failed: %s", u.path, e)
            self._send(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser dropped a media range request mid-transfer; normal during seeking
        except Exception as e:
            log.exception("request %s crashed", u.path)
            self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("UNSHADOW_PORT", 8765)))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    log.info("open http://127.0.0.1:%d", a.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
