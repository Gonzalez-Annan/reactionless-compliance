"""The cage from wherever the hand is, probed on a pool of processes so it comes back in seconds, not minutes.
    pl = Pool(10, livemap.init)          once, kept warm
    job = livemap.ask(pl, d)             returns at once
    job.ready(), np.array(job.get())     98 tips, same order as teleop.DIRS, lines start at the hand as it was at ask()
About 13 s from rest and 7 s from a detour on 10 processes (analysis/envelope/live.py). Tilt is counted from rest.
The map is as old as the wait: it is for a hand that has settled, not one that is moving."""
import mujoco
import teleop as T


def init():
    global G
    G = T.setup("medium")


def probe(a):
    qpos, qvel, u = a; m, d, sid, Rb0 = G
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)                              # or the line starts from where the hand was at rest
    return T.ray(m, d, sid, Rb0, u, 5, "3d")


def ask(pl, d, dirs=T.DIRS):
    return pl.map_async(probe, [(d.qpos.copy(), d.qvel.copy(), u) for u in dirs], chunksize=1)
