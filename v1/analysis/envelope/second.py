"""Is the moving edge the arm's or the controller's? The same test under each of four controllers. Run from the repo root.
Per controller: the map from rest (98 lines), 10 starts one move out (0.3 to 0.8 of the way along a line of its own map),
then from each start a straight move to every point of the rest map pulled in to 0.95, 0.8 and 0.5.
Also what ended each line: the tilt budget, the bus, or a stall, and how many joints sat at a limit then.
  5 Imp.+RNS (ours) and 4 RNS hold the base level while they can. 1 DLS and 3 Impedance do not look at the base at all.
-> data/second_medium.npz and a table. About 6 min per controller on 10 processes."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T
from schemes import NAMES

K, NS = (0.95, 0.8, 0.5), 10


def go(m, d, sid, Rb0, goal, sc):
    """-> arrived, what ended it ("" if nothing did), joints at a limit then"""
    e = goal - d.site_xpos[sid]; dist = np.linalg.norm(e); why = []
    far = T.ray(m, d, sid, Rb0, e / dist, sc, "3d", dist, T.STALL, why=why) if dist > T.TOL else goal
    return (np.linalg.norm(far - goal) < T.TOL, *(why or ["", 0]))


def rest(a):
    sc, u = a; m, d, sid, Rb0 = T.setup("medium"); why = []
    tip = T.ray(m, d, sid, Rb0, u, sc, "3d", why=why) - d.site_xpos[sid]
    return (*tip, *(why or ["", 0]))


def start(a):
    sc, s, tips = a; m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    for k in range(int(12 / m.opt.timestep)):
        if k % T.DECIM == 0:
            p, base = T.ctrl(m, d, sid, Rb0, p0 + s, sc, "3d")
        mujoco.mj_step(m, d)
    d.qvel[:] = 0; mujoco.mj_forward(m, d)
    return base, [[go(m, d, sid, Rb0, p0 + k * t, sc) for t in tips] for k in K]


def count(r):
    return ", ".join("%s %d" % (k or "none", (r == k).sum()) for k in sorted(set(r)))


if __name__ == "__main__":
    out = {}
    with Pool(10) as pl:
        for sc in (5, 4, 1, 3):
            R = pl.map(rest, [(sc, u) for u in T.DIRS], chunksize=1)
            tips = np.array([r[:3] for r in R], float); n = np.linalg.norm(tips, axis=1)
            why, lock = np.array([r[3] for r in R]), np.array([r[4] for r in R])
            print("%d %s. rest map: reach %.2f to %.2f m, median %.2f. ended by: %s. joints at a limit then, median %d" % (
                sc, NAMES[sc], n.min(), n.max(), np.median(n), count(why), np.median(lock)), flush=True)
            for w in sorted(set(why)):
                print("     %-5s reach median %.2f m, joints at a limit median %d" % (w or "none", np.median(n[why == w]), np.median(lock[why == w])))
            rng = np.random.default_rng(3)
            S = [rng.uniform(0.3, 0.8) * tips[j] for j in rng.choice(98, NS, replace=False)]
            G = pl.map(start, [(sc, s, tips) for s in S], chunksize=1)
            base = np.array([g[0] for g in G])
            ok = np.array([[[c[0] for c in k] for k in g[1]] for g in G])            # start, pull-in, tip
            wy = np.array([[[c[1] for c in k] for k in g[1]] for g in G])
            lk = np.array([[[c[2] for c in k] for k in g[1]] for g in G])
            print("     tilt at the %d starts: median %.2f, most %.2f deg" % (NS, np.median(base), base.max()))
            for i, k in enumerate(K):
                bad = ~ok[:, i]
                print("     map pulled in to %.2f: holds for %.0f%% of start-tip pairs, every tip from %d of %d starts. the rest ended by: %s, "
                      "joints at a limit median %d" % (k, 100 * ok[:, i].mean(), ok[:, i].all(1).sum(), NS, count(wy[:, i][bad]),
                                                       np.median(lk[:, i][bad]) if bad.any() else 0), flush=True)
            out.update({"tips%d" % sc: tips, "why%d" % sc: why, "lock%d" % sc: lock, "base%d" % sc: base, "ok%d" % sc: ok,
                        "okwhy%d" % sc: wy, "oklock%d" % sc: lk, "S%d" % sc: np.array(S)})
            np.savez(T.OUT.parent / "second_medium.npz", **out)
