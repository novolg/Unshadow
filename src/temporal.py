"""Motion-compensated multi-frame merge (phone 'night mode' style) before shadow lifting.

Neighbours are aligned to the centre frame with DIS optical flow and averaged with per-pixel
robust weights, so noise drops and sub-8-bit shadow detail emerges; misaligned pixels are rejected.
"""
import cv2
import numpy as np

_DIS = None


def _dis():
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS


def _flow_gray(img):
    # Lift shadows for flow only, so the matcher sees texture in dark regions.
    g = img.max(axis=2) ** 0.5
    return (g * 255).astype(np.uint8)


def merge(frames, center, h=0.03):
    """frames: list of float32 BGR [0,1]; center: index of the reference frame.
    h: rejection threshold (difference after alignment beyond ~2h is treated as motion/occlusion)."""
    ref = frames[center]
    gref = _flow_gray(ref)
    H, W = gref.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ref_s = cv2.GaussianBlur(ref, (0, 0), 1.5)
    acc = ref.copy()
    wsum = np.ones((H, W, 1), np.float32)
    for i, f in enumerate(frames):
        if i == center:
            continue
        flow = _dis().calc(gref, _flow_gray(f), None)
        warped = cv2.remap(f, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(cv2.GaussianBlur(warped, (0, 0), 1.5) - ref_s).max(axis=2, keepdims=True)
        w = np.exp(-(d / h) ** 2)
        acc += w * warped
        wsum += w
    return acc / wsum
