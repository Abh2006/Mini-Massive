from __future__ import annotations
import random
import numpy as np

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))