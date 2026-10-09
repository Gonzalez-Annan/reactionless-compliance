"""Does the envelope probed from rest still hold when the hand starts somewhere else? Run from the repo root.
12 probe directions x goals at 0.6, 0.9, 1.15 of the 2 deg edge, each rehearsed from rest and after a 0.15 m
detour (hand parked at +y, and at +z). The base tilt the detour leaves is kept, as it would be in use.
-> counts of goals where the detour changes the answer."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T

FRACS, DETOURS, PICK = (0.6, 0.9, 1.15), ((0, 0.15, 0), (0, 0, 0.15)), 12


def one(i):
    tips = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    out = []
    for det in (None,) + DETOURS:
        m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy(); base = 0.0
        if det is not None:
            for k in range(int(6 / m.opt.timestep)):          # 6 s: 1.5 s of travel at V_GO, the rest to settle
                if k % T.DECIM == 0:
                    p, base = T.ctrl(m, d, sid, Rb0, p0 + np.array(det), 5, "3d")
                mujoco.mj_step(m, d)
            assert np.linalg.norm(p - p0 - det) < T.TOL, (det, p - p0)
        out.append([base] + [bool(T.check(m, d, sid, Rb0, p0 + f * (tips[i] - p0), 5, "3d")[0]) for f in FRACS])
    return out


if __name__ == "__main__":
    idx = np.random.default_rng(0).choice(len(T.DIRS), PICK, replace=False)
    with Pool(10) as pl:
        R = np.array(pl.map(one, idx), float)                 # (direction, start, [base tilt, answer per frac])
    A = R[:, :, 1:].astype(bool)
    print("from rest, reachable at 0.6 / 0.9 / 1.15 of the edge: %d %d %d of %d" % (*A[:, 0].sum(0), PICK))
    for s, det in enumerate(DETOURS, 1):
        same = A[:, s] == A[:, 0]
        print("after detour %s (base left at %.2f deg): same answer on %d of %d goals; by depth %s; go->no-go %d, no-go->go %d"
              % (det, R[0, s, 0], same.sum(), same.size, same.sum(0).tolist(), (A[:, 0] & ~A[:, s]).sum(), (~A[:, 0] & A[:, s]).sum()))
