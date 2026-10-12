"""The cage from wherever the hand is, probed on a pool of processes so it comes back in seconds, not minutes.
    pl = Pool(10, livemap.init)          once, kept warm
    job = livemap.ask(pl, d)             returns at once
    job.ready(), np.array(job.get())     98 tips, same order as teleop.DIRS, lines start at the hand as it was at ask()
About 13 s from rest and 12 s from a detour on 10 processes (analysis/envelope/live.py). Tilt is counted from rest.
The map is as old as the wait: it is for a hand that has settled, not one that is moving.
    tips = livemap.lookup(m, d, sid, Rb0)   the same 98 tips at once, looked up from stored states: no pool, no wait"""
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


def near(d, Rb0, k):
    """-> (the reaches stored for the k states nearest to this one, k x 98, nearest first; base under the tilt budget).
    Nearest in 7 joint angles + base tilt as a rotation vector, z-scored with the stored set's own mean and std."""
    global S
    if "S" not in globals():                             # the stored states, loaded once
        a, b = (np.load(T.OUT.parent / f"learn{n}_medium.npz") for n in ("", 2))
        X, Y = np.r_[a["X"], b["X"]], np.r_[a["Y"], b["Y"]]
        S = X.mean(0), X.std(0) + 1e-9, X, Y
    mu, sd, X, Y = S
    x = np.r_[d.qpos[7:], (Rb0.inv() * T.Rot.from_quat(d.qpos[[4, 5, 6, 3]])).as_rotvec()]
    return Y[np.argsort(np.linalg.norm((X - mu) / sd - (x - mu) / sd, axis=1))[:k]], np.linalg.norm(x[7:]) < np.radians(T.BASE_MAX)


def lookup(m, d, sid, Rb0, k=10, pull=0.9):
    """The cage looked up instead of probed: per line the shortest reach among the k stored states nearest to this one
    (7 joint angles + base tilt as a rotation vector, z-scored with the stored set's own mean and std), times pull.
    No reach at all with the base at or over the tilt budget. -> 98 tips from the hand, as ask(...).get(), at once.
    Stored: 1000 states one or two moves from rest (analysis/envelope/learn.py gen). This is learn.py safe: with 600
    of them stored, on 200 it never saw it was 0.05 m too far on 1.5% of the lines and kept 0.69 of the reach (D54).
    ponytail: the stored reaches are fast() estimates and a hand more than two moves out has no near neighbour, so
    T.check stays the authority. Medium servicer only, as init()."""
    Yk, lvl = near(d, Rb0, k)
    r = pull * Yk.min(0)
    return d.site_xpos[sid] + lvl * r[:, None] * np.array(T.DIRS)


def band(m, d, sid, Rb0, k=10, pull=0.9, push=1.1):
    """Two edges instead of one (D70), from the same k neighbours as lookup. A single edge is wrong both ways: pull it
    in and it hides goals the arm can reach, let it out and it shows goals it cannot.
    Inner: lookup's edge, pull x the shortest stored reach. Inside it a go is expected to work.
    Outer: push x the longest stored reach. Past it nothing was reachable in any of the k states.
    Between the two the picture does not know, and says so. T.check answers every send.
    -> per line [inner tip, outer tip, outer tip], the three shells view.draw_envelope draws.
    learn.py safe scores both edges: against the full simulation the reach is past the outer one by 0.05 m on 0.4% of
    the lines and the inner one is 0.05 m too far on 1.4%. The band is wide, 0.29 m on a 0.40 m reach.
    ponytail: one k and two factors for every line and state. Narrow it per line if the width is what hurts."""
    Yk, lvl = near(d, Rb0, k)
    return [d.site_xpos[sid] + lvl * np.array([pull * a, push * b, push * b])[:, None] * e
            for a, b, e in zip(Yk.min(0), Yk.max(0), np.array(T.DIRS))]
