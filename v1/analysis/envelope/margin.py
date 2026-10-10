"""Does the live singularity margin (teleop.sig6 / its value at the ready pose) warn of base tilt from poses other
than the ready pose? The cage cannot: it is drawn from rest. Run from the repo root.

Starts: the ready pose and the hand moved 15 cm along each axis direction (6 more, where that move is possible).
From each, 14 straight probes. Per probe that tilts the base 0.5 deg further than it started:
  margin at that moment (should be red, under MARGIN_R), and how far ahead yellow and red came on (cm of hand travel).
Also the false alarms: probes that went red and never tilted 0.5 deg further."""
import sys; sys.path.insert(0, "src")
import copy
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T

AX = [np.array(u, float) / np.linalg.norm(u) for u in
      [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]


def probe(a):
    k, u = a
    m, d, sid, Rb0 = T.setup("medium")
    s0 = T.sig6(m, d)
    if k:                                                       # start k: 15 cm along axis k - 1 first
        d = T.reachable(m, d, sid, Rb0, d.site_xpos[sid] + 0.15 * AX[k - 1], 5, "3d")
        if d is None:
            return None
    d2, p0 = copy.copy(d), d.site_xpos[sid].copy()
    b0, y, r, hit, last = None, None, None, None, 0.0
    for i in range(int((2 * T.REACH / T.V_GO + 1) / m.opt.timestep)):
        if i % T.DECIM == 0:
            p, base = T.ctrl(m, d2, sid, Rb0, p0 + T.REACH * u, 5, "3d")
            b0 = base if b0 is None else b0
            mg, x = T.sig6(m, d2) / s0, (p - p0) @ u
            y = x if y is None and mg < T.MARGIN_Y else y
            r = x if r is None and mg < T.MARGIN_R else r
            if base > b0 + 0.5:
                hit = (x, mg)
                break
            if base > T.BASE_MAX or T.clear(m, d2) < T.CLEAR or (i and i % (50 * T.DECIM) == 0 and x - last < 0.002):
                break
            if i % (50 * T.DECIM) == 0:
                last = x
        mujoco.mj_step(m, d2)
    return k, b0, y, r, hit


if __name__ == "__main__":
    with Pool(10) as pl:
        R = [x for x in pl.map(probe, [(k, u) for k in range(7) for u in AX]) if x]
    for name, sel in (("ready pose", lambda k: k == 0), ("other poses", lambda k: k > 0)):
        P = [x for x in R if sel(x[0])]
        H = [x for x in P if x[4]]
        red = sum(x[4][1] < T.MARGIN_R for x in H)
        yel = sum(x[4][1] < T.MARGIN_Y for x in H)
        ly = [100 * (x[4][0] - x[2]) for x in H if x[2] is not None]
        lr = [100 * (x[4][0] - x[3]) for x in H if x[3] is not None]
        fa = sum(1 for x in P if not x[4] and x[3] is not None)
        print(f"{name}: {len(P)} probes from {len({x[0] for x in P})} starts (base already at up to {max(x[1] for x in P):.2f} deg); "
              f"{len(H)} tilt 0.5 deg further: margin red at that moment on {red}, yellow or red on {yel}; "
              f"warning ahead of it, median cm: yellow {np.median(ly):.0f} (on {len(ly)}), red {np.median(lr):.0f} (on {len(lr)}); "
              f"red without that tilt: {fa} of {len(P) - len(H)}")
