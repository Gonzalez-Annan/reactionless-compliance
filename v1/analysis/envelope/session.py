"""Does the fixed map last a whole session? Run from the repo root. Needs data/second_medium.npz (second.py).
Per controller, 10 sessions of 12 moves. Each goal is a point inside that controller's rest map pulled in to 0.8
(0.2 to 0.8 of the way along a random line). The move is rehearsed first: yes -> it is made, no -> the hand stays
and the map was wrong for that move. Half way and at the end, a straight move to every point of the 0.8 map is tried.
ponytail: the hand only makes moves that pass, so it never sits over the tilt budget. A session of failed moves
and recoveries is not this test.
-> data/session_medium.npz and a table. About 7 min per controller on 10 processes."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/envelope")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T
from schemes import NAMES
from second import go, count

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12            # session.py 40: 40 moves, -> session40_medium.npz. session.py 40 6: only controller 6, added to it
NS, AT = 10, (N // 2, N)


def run(a):
    sc, seed, tips = a; rng = np.random.default_rng(seed)
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    ok, why, tilt, full = [], [], [], []
    for k in range(1, N + 1):
        goal = p0 + rng.uniform(0.2, 0.8) * tips[rng.integers(98)]
        r = go(m, d, sid, Rb0, goal, sc); ok.append(r[0]); why.append(r[1])
        for i in range(int((2 * np.linalg.norm(goal - d.site_xpos[sid]) / T.V_GO + 2) / m.opt.timestep) if r[0] else 0):
            if i % T.DECIM == 0:
                p, base = T.ctrl(m, d, sid, Rb0, goal, sc, "3d")
                if np.linalg.norm(p - goal) < T.TOL / 2:
                    break
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        tilt.append(T.ctrl(m, d, sid, Rb0, goal, sc, "3d")[1])
        if k in AT:
            full.append([go(m, d, sid, Rb0, p0 + 0.8 * t, sc)[0] for t in tips])
    return ok, why, tilt, full


if __name__ == "__main__":
    z = np.load(T.OUT.parent / "second_medium.npz")
    f = T.OUT.parent / ("session_medium.npz" if N == 12 else "session%d_medium.npz" % N)
    out = dict(np.load(f)) if sys.argv[2:] and f.exists() else {}
    with Pool(10) as pl:
        for sc in [int(a) for a in sys.argv[2:]] or (5, 1, 4, 3):
            R = pl.map(run, [(sc, s, z["tips%d" % sc]) for s in range(NS)], chunksize=1)
            ok, why, tilt, full = (np.array([r[i] for r in R]) for i in range(4))
            print("%d %s. moves the 0.8 map said yes to and the rehearsal no: %d of %d (ended by: %s). by move: %s" % (
                sc, NAMES[sc], (~ok).sum(), ok.size, count(why[~ok]), " ".join("%d" % v for v in (~ok).sum(0))))
            Q = [0] + [N * q // 4 - 1 for q in (1, 2, 3, 4)]
            print("     tilt after a move: median %.2f, most %.2f deg. medians after move %s: %s" % (
                np.median(tilt), tilt.max(), ", ".join(str(q + 1) for q in Q), ", ".join("%.2f" % v for v in np.median(tilt, 0)[Q])))
            for i, k in enumerate(AT):
                print("     after move %d: 0.8 map holds for %.0f%% of pairs, every tip from %d of %d sessions" % (
                    k, 100 * full[:, i].mean(), full[:, i].all(1).sum(), NS), flush=True)
            out.update({"ok%d" % sc: ok, "why%d" % sc: why, "tilt%d" % sc: tilt, "full%d" % sc: full})
            np.savez(f, **out)
