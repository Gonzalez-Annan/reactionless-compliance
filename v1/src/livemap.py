"""The cage from wherever the hand is, probed on a pool of processes so it comes back in seconds, not minutes.
    pl = Pool(10, livemap.init)          once, kept warm
    job = livemap.ask(pl, d)             returns at once
    job.ready(), np.array(job.get())     98 tips, same order as teleop.DIRS, lines start at the hand as it was at ask()
About 13 s from rest and 12 s from a detour on 10 processes (analysis/envelope/live.py). Tilt is counted from rest.
The map is as old as the wait: it is for a hand that has settled, not one that is moving."""
import copy, numpy as np, mujoco
import teleop as T
from dynamics import kin


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


def fast(m, d, sid, Rb0, u, dt=0.02):
    """T.ray without the physics: the same control law, the base moved by momentum conservation (v_b = V qd),
    in steps of dt. -> the furthest point with the base level.
    Measured (analysis/envelope/fastmap.py, rest + 4 starts): at dt 0.02 the median error is 8 mm, 95% under 0.06 m,
    and 98 lines take about 60 s on one process, roughly half of T.ray. A bigger dt is faster and wrong: at 0.05
    most lines stop 0.05 m or more short, the control law itself does not survive the coarse step.
    ponytail: joints follow the command exactly and the hand starts at rest, so it is an estimate, T.check stays
    the authority. Not wired into the tool: 2x is not enough to change what the operator sees."""
    d2 = copy.copy(d); d2.qvel[:] = 0
    p0 = best = d.site_xpos[sid].copy()
    last, w = 0.0, max(1, round(0.5 / dt))               # stall test every 0.5 s, as in T.ray
    for i in range(int((2 * T.REACH / T.V_GO + 1) / dt)):
        p, base = T.ctrl(m, d2, sid, Rb0, p0 + T.REACH * u, 5, "3d")
        if base > T.BASE_MAX or T.clear(m, d2) < T.CLEAR:
            break
        best = p
        if i and i % w == 0:
            if (p - p0) @ u - last < 0.002:
                break
            last = (p - p0) @ u
        k = kin(m, d2)
        d2.qvel[6:] = d2.ctrl; d2.qvel[:6] = -np.linalg.solve(k.H_b, k.H_bm @ d2.ctrl)
        mujoco.mj_integratePos(m, d2.qpos, d2.qvel, dt)
        d2.qpos[7:] = np.clip(d2.qpos[7:], T.Q_LIM[:, 0], T.Q_LIM[:, 1])
    return best
