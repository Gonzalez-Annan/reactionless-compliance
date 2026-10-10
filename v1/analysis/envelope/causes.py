"""What the envelope rests on: why the base starts to tilt, direction by direction. Run from the repo root.

The controller (scheme 5, 3-D) solves hand velocity (3 rows of J) and zero base rotation (3 rows of W) together:
6 equations, 7 joints. While the 6 x 7 matrix S = [J; W] over the joints not at a limit has rank 6, both are met
exactly and the base does not tilt at all. So the tilt-free region should end where either
  (a) a second joint reaches its limit (5 free joints < 6 equations), or
  (b) S loses rank on its own (a dynamic singularity: smallest singular value -> 0).
This script tests that on every probe direction: at tilt onset (base first past ONSET deg) it records the locked
joints and the 6th singular value of S over the free joints, and at the end of the probe what stopped it.
-> analysis/envelope/causes.npz and the counts. "neither" is the part the explanation does not cover."""
import sys; sys.path.insert(0, "src")
import copy
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T
from dynamics import kin
from schemes import MARGIN

ONSET, SING = 0.05, 0.1        # deg of base tilt that counts as "started"; singular = sigma6 under SING x its value at rest


def sig6(m, d, free):
    k = kin(m, d)
    S = np.vstack([k.Jstar[:3], k.W])[:, free]
    return np.linalg.svd(S, compute_uv=False)[5] if free.sum() >= 6 else 0.0


def probe(u):
    m, d, sid, Rb0 = T.setup("medium")
    d2, p0 = copy.copy(d), d.site_xpos[sid].copy()
    s0 = sig6(m, d, np.ones(7, bool))
    on, last, end, best, mk, lo = None, 0.0, "arm stalled", p0, [], 1.0
    for i in range(int((2 * T.REACH / T.V_GO + 1) / m.opt.timestep)):
        if i % T.DECIM == 0:
            p, base = T.ctrl(m, d2, sid, Rb0, p0 + T.REACH * u, 5, "3d")
            q = d2.qpos[7:]
            free = np.minimum(q - T.Q_LIM[:, 0], T.Q_LIM[:, 1] - q) > MARGIN + 1e-3
            if base < T.LIMITS[0]:
                lo = min(lo, sig6(m, d2, free) / s0)     # lowest sigma6 seen while the base was still inside 0.5 deg
            while len(mk) < 3 and base > T.LIMITS[len(mk)]:
                mk.append((7 - free.sum(), sig6(m, d2, free) / s0))
            if on is None and base > ONSET:
                on = ((p - p0) @ u, 7 - free.sum(), sig6(m, d2, free) / s0, sig6(m, d2, np.ones(7, bool)) / s0)
            if base > T.BASE_MAX:
                end = "base at 2 deg"; break
            if T.clear(m, d2) < T.CLEAR:
                end = "bus"; break
            best = p
            if i and i % (50 * T.DECIM) == 0:
                if (p - p0) @ u - last < 0.002:
                    break
                last = (p - p0) @ u
        mujoco.mj_step(m, d2)
    return (np.linalg.norm(best - p0), end) + (on or (np.nan, -1, np.nan, np.nan)) + (np.array(mk + [(-1, np.nan)] * 3)[:3], s0, lo)


if __name__ == "__main__":
    with Pool(10) as pl:
        R = pl.map(probe, T.DIRS)
    reach, end = np.array([r[0] for r in R]), np.array([r[1] for r in R])
    on, nl, sf, sa = (np.array([r[k] for r in R], float) for k in (2, 3, 4, 5))
    print("what ends a probe: " + ", ".join(f"{e} {int((end == e).sum())}" for e in sorted(set(end))) + f" (of {len(R)})")
    t = nl >= 0                                                # directions where the base did start to tilt
    lim, sing = t & (nl >= 2), t & (nl < 2) & (sf < SING)
    print(f"base starts to tilt on {t.sum()} directions: {lim.sum()} with two or more joints at a limit, "
          f"{sing.sum()} at a dynamic singularity (sigma6 under {SING:g} of rest), {(t & ~lim & ~sing).sum()} neither")
    print(f"joints at a limit at onset: " + ", ".join(f"{k}: {int((nl[t] == k).sum())}" for k in range(8) if (nl[t] == k).any()))
    print(f"sigma6 of S over all joints at onset, relative to rest: median {np.nanmedian(sa[t]):.2f}, "
          f"10th-90th pct {np.nanpercentile(sa[t], 10):.2f}-{np.nanpercentile(sa[t], 90):.2f}")
    w = (end == "base at 2 deg") & t
    print(f"tilt-free reach (to onset): median {np.nanmedian(on[t]) * 100:.0f} cm; where the probe ends at 2 deg, onset comes at "
          f"{np.nanmedian(on[w] / reach[w]) * 100:.0f}% of the 2 deg reach (median): the wall is that thin")
    M = np.array([r[6] for r in R])                            # (98, 3 shells, [joints at a limit, sigma6 over free joints / rest])
    print(f"sigma6 at rest: {R[0][7]:.3f}")
    for k, lim_ in enumerate(T.LIMITS):
        c = M[:, k, 0] >= 0
        a, b = c & (M[:, k, 0] >= 2), c & (M[:, k, 0] < 2) & (M[:, k, 1] < SING)
        print(f"{lim_:g} deg crossed on {c.sum()} directions: {a.sum()} with two or more joints at a limit, {b.sum()} singular, "
              f"{(c & ~a & ~b).sum()} neither; joints at a limit {np.bincount(M[c, k, 0].astype(int)).tolist()}; "
              f"sigma6 ratio median {np.nanmedian(M[c, k, 1]):.2f}, 90th pct {np.nanpercentile(M[c, k, 1], 90):.2f}")
    lo, c = np.array([r[8] for r in R]), M[:, 0, 0] >= 0
    print(f"lowest sigma6 ratio while inside 0.5 deg: directions that never cross it ({(~c).sum()}): median {np.median(lo[~c]):.2f}, "
          f"min {lo[~c].min():.3f}, under {SING:g} on {(lo[~c] < SING).sum()}; directions that do cross ({c.sum()}): median {np.median(lo[c]):.3f}")
    np.savez("analysis/envelope/causes.npz", reach=reach, end=end, onset=on, nlock=nl, sig_free=sf, sig_all=sa, marks=M)
