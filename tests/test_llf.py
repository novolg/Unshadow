"""Run: python3.11 -m pytest tests/  (or python3.11 tests/test_llf.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llf import enhance  # noqa: E402


def _ratios(img):
    f = img.astype(np.float32) + 1
    return f / f.max(axis=2, keepdims=True)


def test_hue_kept_on_saturated_blue():
    # Deep blue gradient (R,G ~ 0): shared gain must keep channel ratios, sharpening included.
    x = np.linspace(20, 250, 128, dtype=np.float32)
    img = np.zeros((128, 128, 3), np.uint8)
    img[..., 0] = x[None, :]                     # B
    img[..., 1] = (x[None, :] * 0.08).astype(np.uint8)
    out = enhance(img, sharpen=1.0)
    assert np.abs(_ratios(out)[..., 1] - _ratios(img)[..., 1]).max() < 0.06


def test_black_gate_keeps_near_black():
    img = np.full((64, 64, 3), 6, np.uint8)     # ~0.024, below the 0.05 gate
    out = enhance(img, lift=1.0)
    assert np.abs(out.astype(int) - img).max() <= 1


if __name__ == "__main__":
    test_hue_kept_on_saturated_blue()
    test_black_gate_keeps_near_black()
    print("ok")
