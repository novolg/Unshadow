"""Build a side-by-side grid of methods: compare.py out.png frame1.png frame2.png ..."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dual import clahe, dual  # noqa: E402

VARIANTS = [
    ("orig", lambda f: f),
    ("max", lambda f: dual(f, clarity=0.5)),
    ("luma", lambda f: dual(f, mode="luma", clarity=0.5)),
    ("mix", lambda f: dual(f, mode="mix", clarity=0.5)),
    ("luma g0.45", lambda f: dual(f, mode="luma", gamma=0.45, clarity=0.7, sat=1.1)),
]

rows = []
for p in sys.argv[2:]:
    f = cv2.imread(p)
    f = cv2.resize(f, (int(f.shape[1] * 480 / f.shape[0]), 480))
    cells = []
    for name, fn in VARIANTS:
        o = fn(f).copy()
        cv2.putText(o, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cells.append(o)
    rows.append(np.hstack(cells))
w = min(r.shape[1] for r in rows)
cv2.imwrite(sys.argv[1], np.vstack([r[:, :w] for r in rows]))
