"""How long does the cage take if the 98 probes are shared over the cores, from wherever the hand is?
Run from the repo root. The pool is started once and kept warm, as the tool would, so its start-up is not counted.
Timed from rest and after a 0.3 m detour. From rest the tips are checked against the cached cage."""
import sys; sys.path.insert(0, "src")
import os, time, numpy as np, mujoco
from multiprocessing import Pool
import teleop as T, livemap


def cage(pl, d):
    return np.array(livemap.ask(pl, d).get())


if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    old = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    for n in (10, os.cpu_count()):
        m, d, sid, Rb0 = T.setup("medium")
        with Pool(n, livemap.init) as pl:
            cage(pl, d)                                       # warm up
            t = time.time(); tips = cage(pl, d); t0 = time.time() - t
            for k in range(int(6 / m.opt.timestep)):
                if k % T.DECIM == 0:
                    T.ctrl(m, d, sid, Rb0, p0 + np.array((0, 0.3, 0)), 5, "3d")
                mujoco.mj_step(m, d)
            t = time.time(); cage(pl, d); t1 = time.time() - t
        print("%2d processes: cage from rest %.1f s, after a detour %.1f s, tips against the cached cage: worst %.4f m"
              % (n, t0, t1, np.linalg.norm(tips - old, axis=1).max()))
