"""Fast DUAL exposure correction (Zhang et al. 2019) with guided-filter illumination + Mertens fusion."""
import cv2
import numpy as np

_MERTENS = cv2.createMergeMertens(1.0, 1.0, 1.0)


LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR weights


def illumination(img, radius, mode, eps=1e-3):
    """Smooth, edge-aware illumination map (Retinex). mode: 'max' = max-RGB, 'luma' = Rec.601 luma,
    'mix' = mean of both (luma sees dark detail under saturated coloured light)."""
    if mode == "max":
        lum = img.max(axis=2)
    elif mode == "luma":
        lum = img @ LUMA
    else:
        lum = 0.5 * (img.max(axis=2) + img @ LUMA)
    t = cv2.ximgproc.guidedFilter(lum, lum, radius, eps)
    return np.clip(t, 1e-3, 1.0)


def under_fix(img, radius, gamma, mode, denoise=0.0, black=0.0):
    t = illumination(img, radius, mode)
    if denoise:
        # Only near-black zones: lifting them exposes codec blocks; mid-shadows keep their detail.
        smooth = cv2.ximgproc.guidedFilter(img, img, 6, (0.02 * denoise) ** 2)
        w = (np.clip(1 - t / 0.15, 0, 1) ** 2)[..., None]
        img = img * (1 - w) + smooth * w
    gain = t ** -gamma
    if black:
        # Black point: illumination below `black` holds no real detail, so fade the lift out there.
        p = np.clip((t - black) / black, 0, 1)  # no lift below black, full lift above 2*black
        gain = 1 + (gain - 1) * (p * p * (3 - 2 * p))
    return np.clip(img * gain[..., None], 0, 1)


def dual(frame_bgr, gamma=0.6, radius=0.02, clarity=0.0, sat=1.0, fuse_orig=True, mode="max", black=0.0,
         denoise=0.0):
    """uint8 BGR -> uint8 BGR."""
    img = frame_bgr.astype(np.float32) / 255.0
    r = max(2, int(radius * frame_bgr.shape[0]))
    up = under_fix(img, r, gamma, mode, denoise, black)     # shadows recovered
    down = 1 - under_fix(1 - img, r, gamma, "max")    # highlights recovered
    stack = [img, up, down] if fuse_orig else [up, down]
    # OpenCV's Mertens expects 0..255 range even for float input.
    out = _MERTENS.process([s * 255.0 for s in stack])

    if clarity:
        # Local contrast on luminance, large radius: adds "punch" without halos at this strength.
        lab = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2Lab)
        L = lab[..., 0]
        blur = cv2.GaussianBlur(L, (0, 0), max(2, frame_bgr.shape[0] * 0.015))
        lab[..., 0] = np.clip(L + clarity * (L - blur), 0, 100)
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    if sat != 1.0:
        hsv = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def clahe(frame_bgr, clip=2.5, tiles=8):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    lab[..., 0] = cv2.createCLAHE(clip, (tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
