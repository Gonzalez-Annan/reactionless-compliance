"""Does the guaranteed cage hold from starts it was not built on? Run from the repo root.
The cage was measured from 40 one-leg starts (wide.py --98). Here: 40 two-leg starts, rest -> a -> b, a and b drawn
like the starts of wide.py with other seeds. From each, one rehearsal to the tip of each of the 98 guaranteed lines.
A tip that fails is a place the small map shows as safe and is not.
ponytail: one check per tip, not a search, so it says that a line fails, not by how much.
-> data/twoleg_medium.npz and a few lines."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/envelope")
import time, numpy as np, mujoco
from multiprocessing import Pool
import teleop as T
from wide import dirs, N


def one(a):
    legs, G = a
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    for s in legs:
        for k in range(int(T.T_MOVE / m.opt.timestep)):
            if k % T.DECIM == 0:
                p, base = T.ctrl(m, d, sid, Rb0, p0 + s, 5, "3d")
            mujoco.mj_step(m, d)
    ok = np.linalg.norm(p - p0 - legs[-1]) <= T.TOL
    return [base] + [float(T.check(m, d, sid, Rb0, g, 5, "3d")[0]) if ok else np.nan for g in G]


if __name__ == "__main__":
    t = time.time()
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    rays = np.load(T.OUT.parent / "decide4_medium.npz")["rays"]
    tips = list(rays[:, -1])
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0      # twoleg.py 0.8: the same tips pulled in to 0.8
    G = p0 + k * (np.array([r[-1] for r in T.guaranteed(rays, p0, "medium")]) - p0)
    S = [dirs(s, N, tips, p0) * np.random.default_rng(s + 1).uniform(0.3, 0.8, N)[:, None] for s in (11, 13)]
    with Pool(10) as pl:
        R = np.array(pl.map(one, [((a, b), G) for a, b in zip(*S)], chunksize=1), float)
    np.savez(T.OUT.parent / ("twoleg_medium.npz" if k == 1 else "twoleg%g_medium.npz" % k), R=R, S=np.array(S), G=G)
    K = R[~np.isnan(R[:, 1]), 1:]
    keep = np.linalg.norm(G - p0, axis=1) / np.linalg.norm(np.array(tips) - p0, axis=1)
    print("%d of %d two-leg starts arrived, tilt left there: median %.2f, most %.2f deg, %.0f min" % (
        len(K), N, np.nanmedian(R[:, 0]), np.nanmax(R[:, 0]), (time.time() - t) / 60))
    print("guaranteed tips that failed: %d of %d pairs (%.1f%%), on %d of 98 lines, from %d of %d starts" % (
        (K == 0).sum(), K.size, 100 * (K == 0).mean(), (K == 0).any(0).sum(), (K == 0).any(1).sum(), len(K)))
    print("lines that failed keep %s of the rest line" % np.round(np.sort(keep[(K == 0).any(0)]), 2))
