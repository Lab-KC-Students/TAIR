"""Plate geometry shared by TAIR inference (infer_plates.py) and the plate training set (../scripts/build_tair_set.py),
so restoration is trained on exactly the letterbox it is run on. PIL + numpy only (importable from the main venv)."""
import numpy as np
from PIL import Image


def letterbox(img, s=512):
    """Keep aspect: bicubic-resize the long side to s, edge-pad to s x s (val.py squashes to 512x512, which stretches
    ~3:1 plates 3x vertically). Returns (PIL RGB s x s, (left, top, new_w, new_h)); crop that box back out after
    restoring. Edge padding = np.pad(mode='edge'), the same op torchvision's TF.pad(padding_mode='edge') runs on PIL."""
    w, h = img.size
    nw, nh = max(1, round(w * s / max(w, h))), max(1, round(h * s / max(w, h)))
    l, t = (s - nw) // 2, (s - nh) // 2
    a = np.asarray(img.convert('RGB').resize((nw, nh), Image.BICUBIC))
    a = np.pad(a, ((t, s - nh - t), (l, s - nw - l), (0, 0)), mode='edge')
    return Image.fromarray(a), (l, t, nw, nh)
