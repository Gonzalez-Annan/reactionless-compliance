"""Is the cage static? Run from the repo root. Exits 1 if it is not.
The claim under test: a cage drawn once with the arm at rest stays right wherever the hand has been.
The cage is the saved rest map and is never redrawn here. Every goal is inside it or on its edge (0.6 to 1.0 of the
way out, 20 random directions) and is fixed in the world. Each one is rehearsed from rest and again after one 0.3 m
move, with the base tilt that move leaves kept, as it would be in use. A static cage gives the same answer both times.
-> the goals where the answer changes, and what stopped the hand."""
import sys; sys.path.insert(0, "src")
from collections import Counter
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T

FRACS = (0.6, 0.8, 0.9, 1.0)
DETS = ((0, 0, 0.3), (0, 0, -0.3), (-0.3, 0, 0), (0.3, 0, 0), (0, 0.3, 0))    # the five moves of start.py far
WHY = dict(tilt="base tilt passed 2 deg", bus="arm would hit the bus", stall="hand stalled short of the goal", time="ran out of time")


def one(job):
    edge, det = job                                           # edge: the cage's 2 deg edge along one direction, from the rest hand position
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy(); base = 0.0
    if det is not None:
        for k in range(int(6 / m.opt.timestep)):              # 6 s: 3 s of travel at V_GO, the rest to settle
            if k % T.DECIM == 0:
                p, base = T.ctrl(m, d, sid, Rb0, p0 + np.array(det), 5, "3d")
            mujoco.mj_step(m, d)
        assert np.linalg.norm(p - p0 - det) < T.TOL, "the move itself did not arrive"
    out = []
    for f in FRACS:                                           # T.check, opened up to read why a line ended
        e = p0 + f * edge - d.site_xpos[sid]; dist = np.linalg.norm(e); why = []
        far = T.ray(m, d, sid, Rb0, e / dist, 5, "3d", dist, T.STALL, why=why)
        out.append("" if np.linalg.norm(far - p0 - f * edge) < T.TOL else why[0] if why else "time")
    return base, out


if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    U = np.random.default_rng(0).normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]    # the 20 directions of start.py hard
    E = [u / T.depth(list(tips), p0, p0 + u) for u in U]      # depth 1 is the cage's edge
    with Pool(10) as pl:
        R = pl.map(one, [(e, det) for det in (None,) + DETS for e in E])
    R = [R[i:i + len(E)] for i in range(0, len(R), len(E))]   # [start][direction] -> (base tilt left, [what stopped each goal, "" if it arrived])
    go = [(k, j) for k in range(len(E)) for j in range(len(FRACS)) if R[0][k][1][j] == ""]
    print("CLAIM UNDER TEST: the cage is static. Drawn once at rest, it is right wherever the hand has been.")
    print("The cage is the saved rest map. It is not redrawn anywhere in this test.")
    print("%d goals fixed in the world: %d directions, each at %s of the way to the cage edge." % (len(E) * len(FRACS), len(E), FRACS))
    print("\nFrom rest: %d of %d goals GO.\n" % (len(go), len(E) * len(FRACS)))
    bad = []
    for det, rows in zip(DETS, R[1:]):
        flips = [(k, j, rows[k][1][j]) for k, j in go if rows[k][1][j]]
        bad += [(j, k, det, w) for k, j, w in flips]
        print("After one move of %s m (base left at %.2f deg, cage not redrawn): %d of %d now REFUSED. By depth %s. %s"
              % (det, rows[0][0], len(flips), len(go), [sum(j == x for _, j, _ in flips) for x in range(len(FRACS))],
                 dict(Counter(w for _, _, w in flips))))
    print("\nEvery goal that is GO from rest and REFUSED after one move, deepest inside the cage first:")
    for j, k, det, w in sorted(bad):
        print("  direction %2d, goal %.2f of the way out (%4.1f cm inside the edge), after move %s: %s"
              % (k, FRACS[j], 100 * (1 - FRACS[j]) * np.linalg.norm(E[k]), det, WHY[w]))
    print("\nRESULT: %d of %d checks on goals that are GO from rest are REFUSED after one 0.3 m move." % (len(bad), len(go) * len(DETS)))
    print("CLAIM BROKEN: the cage is not static." if bad else "CLAIM HOLDS on every goal tested.")
    sys.exit(1 if bad else 0)
