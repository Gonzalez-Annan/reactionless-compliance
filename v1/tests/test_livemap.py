"""The pooled cage gives the same tips as the cached one-thread cage, from rest, on six of the probe lines."""
import sys; sys.path.insert(0, "src")
import numpy as np
from multiprocessing import Pool
import teleop as T, livemap


def test_same_tips_as_cache():
    m, d, sid, Rb0 = T.setup("medium")
    old = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    idx = [0, 20, 40, 60, 80, 97]
    with Pool(3, livemap.init) as pl:
        job = livemap.ask(pl, d, [T.DIRS[i] for i in idx])
        tips = np.array(job.get(120))
    assert np.abs(tips - old[idx]).max() < 1e-6
