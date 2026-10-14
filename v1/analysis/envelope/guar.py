"""How far does the 2 deg edge move with where the hand starts, and why? Run from the repo root.
20 random directions between the probe lines. For each, the edge (furthest reachable goal on that line from the rest
hand position, as a fraction of the map's edge) is found by bisection from rest and after five 0.3 m detours.
Two versions after a detour: "kept" counts tilt from the attitude at rest, as in use; "forgiven" counts it from the
attitude the detour left, so only the arm's new pose can move the edge.
The guaranteed edge of a direction is the worst over the six starts.
ponytail: bisection takes reachability as one stretch along the line (in then out); a hole would be missed.
-> data/guar_medium.npz and a table."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
import teleop as T

DETS = ((0, 0, 0.3), (0, 0, -0.3), (-0.3, 0, 0), (0.3, 0, 0), (0, 0.3, 0))
LO, HI, STEPS = 0.3, 1.3, 6                                   # edge known to (HI - LO) / 2^STEPS = 0.016


def edge(m, d, sid, Rb, p0, e):
    if not T.check(m, d, sid, Rb, p0 + LO * e, 5, "3d")[0]:
        return np.nan                                         # not even the near end is reachable
    lo, hi = LO, HI
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if T.check(m, d, sid, Rb, p0 + mid * e, 5, "3d")[0] else (lo, mid)
    return lo


def one(e):
    out = []                                                  # per start: tilt left (deg), arrived, edge kept, edge forgiven
    for det in (None,) + DETS:
        m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy(); base, ok = 0.0, True
        if det is not None:
            for k in range(int(6 / m.opt.timestep)):          # the same detour as start.py: 6 s, travel then settle
                if k % T.DECIM == 0:
                    p, base = T.ctrl(m, d, sid, Rb0, p0 + np.array(det), 5, "3d")
                mujoco.mj_step(m, d)
            ok = np.linalg.norm(p - p0 - det) < T.TOL
        kept = edge(m, d, sid, Rb0, p0, e)
        out.append([base, ok, kept, kept if det is None else edge(m, d, sid, Rot.from_quat(d.qpos[[4, 5, 6, 3]]), p0, e)])
    return out


if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1])
    rng = np.random.default_rng(0)
    U = rng.normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]      # the same 20 as start.py
    E = np.array([u / T.depth(tips, p0, p0 + u) for u in U])
    with Pool(10) as pl:
        R = np.array(pl.map(one, E), float)                   # (direction, start, 4)
    np.savez(T.OUT.parent / "guar_medium.npz", R=R, E=E, dets=np.array(DETS))
    kept, forg = R[:, :, 2], R[:, :, 3]
    print("start        tilt left  arrived  edge kept (med, min)  edge forgiven (med, min)  |kept - rest| med")
    for s, name in enumerate(("rest",) + DETS):
        print("%-12s %6.2f deg  %5d    %5.2f %5.2f           %5.2f %5.2f              %5.2f" % (
            name, R[0, s, 0], R[0, s, 1], np.nanmedian(kept[:, s]), np.nanmin(kept[:, s]),
            np.nanmedian(forg[:, s]), np.nanmin(forg[:, s]), np.nanmedian(abs(kept[:, s] - kept[:, 0]))))
    g = np.nanmin(kept, 1)
    print("guaranteed edge over the six starts: median %.2f, worst %.2f of the map's edge; from rest: median %.2f"
          % (np.median(g), g.min(), np.median(kept[:, 0])))
    print("reach kept by the guarantee, as volume (edge cubed, mean over directions): %.0f%%" % (100 * np.mean(g ** 3) / np.mean(kept[:, 0] ** 3)))
    print("directions with an unreachable near end from some start: %d" % np.isnan(kept).any(1).sum())
    print("edge moved back by forgiving the tilt: median |forgiven - rest| %.2f against |kept - rest| %.2f"
          % (np.nanmedian(abs(forg[:, 1:] - kept[:, :1])), np.nanmedian(abs(kept[:, 1:] - kept[:, :1]))))
