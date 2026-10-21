"""Fast Local Laplacian Filter (Aubry et al. 2014) on max-RGB intensity, re-applied as shared RGB gain.

max-RGB as intensity: under saturated blue light it follows the blue channel (luma would see ~7 %),
and since every channel <= max, scaling RGB by I'/I with I' <= 1 can never clip a channel (no hue burn).
"""
import cv2
import numpy as np


def _gauss_pyr(img, levels):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyr(img, levels):
    g = _gauss_pyr(img, levels)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(levels - 1)]
    lap.append(g[-1])
    return lap


def _remap(i, g, sigma, alpha, beta, noise):
    """Aubry/Paris remapping: |d|<=sigma is detail (power alpha, <1 boosts), beyond is edge (scaled by beta).
    Differences below `noise` stay linear (Paris 2011): x^alpha has infinite slope at 0 and would turn
    8-bit steps and codec noise into banding."""
    d = i - g
    a = np.abs(d)
    ac = np.minimum(a, sigma)
    detail = sigma * (ac / sigma) ** alpha
    t = np.clip((ac - noise) / noise, 0, 1)
    detail = ac + (detail - ac) * (t * t * (3 - 2 * t))
    edge = beta * np.maximum(a - sigma, 0)
    return g + np.sign(d) * (detail + edge)


def llf(I, sigma=0.2, alpha=0.6, beta=0.6, levels=None, samples=40, noise=0.02, pivot=0.4):
    """I: float32 HxW in [0,1]. Returns filtered intensity."""
    if levels is None:
        levels = max(3, int(np.log2(min(I.shape))) - 3)
    gI = _gauss_pyr(I, levels)
    out = [np.zeros_like(x) for x in gI]
    # Fast-LLF interpolates between remapped pyramids; <30 samples shows as contour banding in smooth glow.
    grid = np.linspace(0, 1, samples).astype(np.float32)
    step = grid[1] - grid[0]
    for j, gj in enumerate(grid):
        lp = _lap_pyr(_remap(I, gj, sigma, alpha, beta, noise).astype(np.float32), levels)
        for l in range(levels - 1):
            # Hat-function weight: linear interpolation between neighbouring intensity samples.
            w = np.clip(1 - np.abs(gI[l] - gj) / step, 0, 1)
            out[l] += w * lp[l]
    # Coarsest residual: compress around a fixed mid-grey pivot (a per-frame mean would pulse in video).
    out[-1] = pivot + (gI[-1] - pivot) * beta
    res = out[-1]
    for l in range(levels - 2, -1, -1):
        res = cv2.pyrUp(res, dstsize=out[l].shape[1::-1]) + out[l]
    return res


def enhance(frame_bgr, sigma=0.2, alpha=0.5, beta=0.5, lift=0.3, black=0.05, sat=1.0, noise=0.02,
            sharpen=0.0, sharpen_radius=1.0):
    """BGR (uint8, or float32 in [0,1]) -> uint8 BGR.
    alpha: detail (<1 boosts), beta: range compression (<1 compresses), lift: global shadow gamma (0..1),
    black: below this no change (hides codec floor), sat: >1 more saturated,
    sharpen: unsharp amount on the scalar intensity (keeps the shared-gain colour guarantee)."""
    img = frame_bgr.astype(np.float32) / 255.0 if frame_bgr.dtype == np.uint8 else frame_bgr
    I = np.maximum(img.max(axis=2), 1e-4)
    Iin = I ** (1 - 0.5 * lift) if lift else I
    Io = np.clip(llf(Iin, sigma, alpha, beta, noise=noise), 0, 1)
    if black:
        p = np.clip((I - black) / black, 0, 1)
        p = p * p * (3 - 2 * p)
        Io = I + (Io - I) * p
    if sharpen:
        Io = np.clip(Io + sharpen * (Io - cv2.GaussianBlur(Io, (0, 0), sharpen_radius)), 0, 1)
    ratio = img / I[..., None]                      # chroma, all channels <= 1
    if sat != 1.0:
        ratio = ratio ** sat                        # Mantiuk-style exponent on colour ratios
    out = ratio * Io[..., None]
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
