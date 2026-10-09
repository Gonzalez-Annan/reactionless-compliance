"""Does the envelope probed from rest still hold when the hand starts somewhere else? Run from the repo root.
Each goal is rehearsed from rest and after a detour that parks the hand elsewhere first. The base tilt the detour
leaves is kept, as it would be in use.
easy: 12 probe directions, goals at 0.6, 0.9, 1.15 of the 2 deg edge, two detours.
hard: 20 random directions between the probe lines, goals at 0.95, 1.0, 1.05 of the map's edge, four detours.
-> counts of goals where the detour changes the answer."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T

Y, Z = (0, 0.15, 0), (0, 0, 0.15)
SETS = dict(easy=((0.6, 0.9, 1.15), (Y, Z)), hard=((0.95, 1.0, 1.05), (Y, Z, (-0.15, 0, 0), (0.1, 0, 0.1))))


def one(job):
    edge, fracs, dets = job                                   # edge: the map's 2 deg edge along one direction, from the rest hand position
    out = []
    for det in (None,) + dets:
        m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy(); base, ok = 0.0, True
        if det is not None:
            for k in range(int(6 / m.opt.timestep)):          # 6 s: 1.5 s of travel at V_GO, the rest to settle
                if k % T.DECIM == 0:
                    p, base = T.ctrl(m, d, sid, Rb0, p0 + np.array(det), 5, "3d")
                mujoco.mj_step(m, d)
            ok = np.linalg.norm(p - p0 - det) < T.TOL         # the detour itself has to arrive
        out.append([base, ok] + [bool(T.check(m, d, sid, Rb0, p0 + f * edge, 5, "3d")[0]) for f in fracs])
    return out


if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    rng = np.random.default_rng(0)
    U = rng.normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]
    E = dict(easy=tips[rng.choice(len(tips), 12, replace=False)] - p0,
             hard=np.array([u / T.depth(list(tips), p0, p0 + u) for u in U]))
    assert all(abs(T.depth(list(tips), p0, p0 + e) - 1) < 1e-6 for e in E["hard"])       # depth 1 is the map's edge
    for name in sys.argv[1:] or SETS:
        fracs, dets = SETS[name]
        with Pool(10) as pl:
            R = np.array(pl.map(one, [(e, fracs, dets) for e in E[name]]), float)      # (direction, start, [base, arrived, answers])
        A = R[:, :, 2:].astype(bool)
        print("%s: from rest, reachable at %s of the edge: %s of %d" % (name, fracs, A[:, 0].sum(0).tolist(), len(A)))
        for s, det in enumerate(dets, 1):
            same = A[:, s] == A[:, 0]
            print("  detour %s (arrived %d, base left at %.2f deg): same answer on %d of %d goals; by depth %s; go->no-go %d, no-go->go %d"
                  % (det, R[0, s, 1], R[0, s, 0], same.sum(), same.size, same.sum(0).tolist(), (A[:, 0] & ~A[:, s]).sum(), (~A[:, 0] & A[:, s]).sum()))
