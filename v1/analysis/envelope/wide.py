"""The guaranteed edge over many starts. Run from the repo root.
40 random starts: one straight move from rest to a point 0.3 to 0.8 of the way to the cage, random direction.
From each, the edge along the same 20 directions as guar.py, tilt counted from rest as in use.
Guaranteed edge of a direction = worst over the starts that arrived. Printed against the number of starts used,
to see whether it has stopped shrinking.
ponytail: one-leg starts only; a hand that wandered through several legs is not covered.
--98: along the 98 lines of the cage instead (edge 1 = the cage tip), for the map the study draws.
-> data/wide_medium.npz (--98: wide98_medium.npz) and a table."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/envelope")
import time, numpy as np, mujoco
from multiprocessing import Pool
import teleop as T
from guar import edge

N = 40


def dirs(seed, n, tips, p0):
    U = np.random.default_rng(seed).normal(size=(n, 3)); U /= np.linalg.norm(U, axis=1)[:, None]
    return np.array([u / T.depth(tips, p0, p0 + u) for u in U])


def one(a):
    start, E = a
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    for k in range(int(12 / m.opt.timestep)):                 # 12 s: the longest start is under 0.6 m away
        if k % T.DECIM == 0:
            p, base = T.ctrl(m, d, sid, Rb0, p0 + start, 5, "3d")
        mujoco.mj_step(m, d)
    if np.linalg.norm(p - p0 - start) > T.TOL:
        return [base] + [np.nan] * len(E)
    return [base] + [edge(m, d, sid, Rb0, p0, e) for e in E]


if __name__ == "__main__":
    t = time.time()
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1])
    W = "--98" in sys.argv
    E = np.array(tips) - p0 if W else dirs(0, 20, tips, p0)   # the same 20 as guar.py
    S = dirs(1, N, tips, p0) * np.random.default_rng(2).uniform(0.3, 0.8, N)[:, None]
    with Pool(10) as pl:
        R = np.array(pl.map(one, [(s, E) for s in S], chunksize=1), float)     # (start, 1 + direction)
    rest = np.ones(len(E)) if W else np.load(T.OUT.parent / "guar_medium.npz")["R"][:, 0, 2]   # ponytail: --98 takes the cage tip as the rest edge
    np.savez(T.OUT.parent / ("wide98_medium.npz" if W else "wide_medium.npz"), R=R, E=E, S=S, rest=rest)
    K = R[~np.isnan(R[:, 1]), 1:]
    print("%d of %d starts arrived, tilt left there: median %.2f, most %.2f deg, %.0f min" % (
        len(K), N, np.nanmedian(R[:, 0]), np.nanmax(R[:, 0]), (time.time() - t) / 60))
    print("pairs within 0.05 of the rest edge: %.0f%%, below 0.8 of it: %.0f%%" % (
        100 * np.nanmean(abs(K - rest) <= 0.05), 100 * np.nanmean(K < 0.8 * rest)))
    print("starts used   guaranteed edge (median, worst)   reach kept as volume")
    for n in (5, 10, 20, len(K)):
        g = np.nanmin(K[:n], 0)
        print("%6d          %.2f  %.2f                       %.0f%%" % (n, np.median(g), g.min(), 100 * np.mean(g ** 3) / np.mean(rest ** 3)))
    print("directions with an unreachable near end (0.3) from some start: %d" % np.isnan(K).any(0).sum())
