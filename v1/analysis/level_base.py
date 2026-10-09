"""Level-base findings behind DECISIONS.md D17 and the reach table in the paper. [P2]

    python analysis/level_base.py     # about 4 minutes, prints only

A free direction of the hand, B base authority with the hand still and re-levelling, C reach by servicer size,
D base turn per metre of hand travel, E re-levelling at short goals, F out-and-back and closed loops.
ponytail: two scratch scripts joined end to end, hence the second import block.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np, mujoco, scipy.linalg as sl
import teleop
from teleop import setup, ray, ctrl, DECIM
from dynamics import kin
from task import Q_LIM
np.set_printoptions(precision=3, suppress=True, linewidth=200)
U = np.r_[np.eye(3), -np.eye(3)]
rng = np.random.default_rng(1)

# A. which way is the hand free when the base is level? hypothesis: along the line through the centre of mass
m, d, sid, Rb0 = setup("medium")
ang, ratio, auth = [], [], []
for q in rng.uniform(Q_LIM[:, 0] + 0.3, Q_LIM[:, 1] - 0.3, (2000, 7)):
    d.qpos[7:] = q; mujoco.mj_forward(m, d); k = kin(m, d); J, W = k.Jstar[:3], k.W
    u, s, _ = np.linalg.svd(J @ sl.null_space(W, rcond=1e-9))
    r = d.site_xpos[sid] - d.subtree_com[0]; r /= np.linalg.norm(r)
    ang.append(np.degrees(np.arccos(abs(u[:, 0] @ r)))); ratio.append(s[0] / s[1])
    auth.append(np.linalg.svd(W @ sl.null_space(J), compute_uv=False) / np.linalg.svd(W, compute_uv=False))
print("A free direction vs line to centre of mass, deg: median %.0f, 90th pct %.0f (random would be 60)" % (np.median(ang), np.percentile(ang, 90)))
print("A free direction is this many times freer than the next: median %.1f" % np.median(ratio))
print("B share of base-turning authority left with the hand held still (3 axes, median):", np.median(auth, 0))

# B. does the base come back level after the hand arrives?
teleop.BASE_MAX = 90
for g in ([0, 0, .6], [0, .5, 0]):
    m, d, sid, Rb0 = setup("medium"); p0 = d.site_xpos[sid].copy(); log, pk, arr = [], 0, None
    for i in range(int(25 / m.opt.timestep)):
        if i % DECIM == 0:
            p, b = ctrl(m, d, sid, Rb0, p0 + g, 5, "3d"); pk = max(pk, b)
            if arr is None and np.linalg.norm(p - p0 - g) < 0.005: arr = i
            if arr is not None and (i - arr) in (0, 4000, 10000, 20000): log.append(round(b, 2))
        mujoco.mj_step(m, d)
    print("B goal", g, "peak base deg %.2f" % pk, "base at arrival, +2 s, +5 s, +10 s:", log)

# C. how reach scales with the size of the servicer
for size in ("small", "medium", "large"):
    out = []
    for bm in (0.5, 2.0):
        teleop.BASE_MAX = bm
        m, d, sid, Rb0 = setup(size); p0 = d.site_xpos[sid].copy()
        out.append(np.mean([(ray(m, d, sid, Rb0, u, 5, "3d") - p0) @ u for u in U]))
    print("C", size, "base mass %.0f kg" % m.body_mass[1], "mean axis reach m at 0.5 deg, 2 deg:", np.round(out, 2))
import numpy as np, mujoco, scipy.linalg as sl
import teleop
from teleop import setup, ctrl, DECIM
from dynamics import kin
from task import Q_LIM
np.set_printoptions(precision=3, suppress=True, linewidth=200)
teleop.BASE_MAX = 90


def go(m, d, sid, Rb0, target, hold=8.0, tmax=30.0):
    """Drive to target, then hold. -> base deg at arrival (None if never), base deg at the end, peak."""
    arr, at, pk, b = None, None, 0, 0
    for i in range(int(tmax / m.opt.timestep)):
        if i % DECIM == 0:
            p, b = ctrl(m, d, sid, Rb0, target, 5, "3d"); pk = max(pk, b)
            if arr is None and np.linalg.norm(p - target) < 0.005: arr, at = i, b
            if arr is not None and (i - arr) * m.opt.timestep > hold: break
        mujoco.mj_step(m, d)
    return at, b, pk


# D. base turn per metre of hand travel (the coupling W J+), by servicer size
for size in ("small", "medium", "large"):
    m, d, sid, Rb0 = setup(size); k = kin(m, d)
    print("D", size, "deg of base turn per 10 cm of hand travel, worst/mid/best direction:",
          np.degrees(np.linalg.svd(k.W @ np.linalg.pinv(k.Jstar[:3]), compute_uv=False)) / 10)

# E. why the base does not come back level: short goals, joints at limits, authority left
for g in ([0, 0, .3], [0, .25, 0], [0, 0, .6]):
    m, d, sid, Rb0 = setup("medium"); p0 = d.site_xpos[sid].copy()
    at, end, pk = go(m, d, sid, Rb0, p0 + g); q = d.qpos[7:]; k = kin(m, d)
    lim = int(np.sum((q < Q_LIM[:, 0] + 0.06) | (q > Q_LIM[:, 1] - 0.06)))
    a = np.linalg.svd(k.W @ sl.null_space(k.Jstar[:3]), compute_uv=False)
    print("E goal", g, "base at arrival", at, "after 8 s hold %.2f" % end, "joints at a limit", lim, "base authority with hand still", a)

# F. out and back, and round a square: does the base come home when the hand does?
m, d, sid, Rb0 = setup("medium"); p0 = d.site_xpos[sid].copy(); q0 = d.qpos[7:].copy()
go(m, d, sid, Rb0, p0 + [0, .3, 0], 0); _, b, _ = go(m, d, sid, Rb0, p0, 3)
print("F out 0.3 m and straight back: base %.2f deg, joints off start %.2f rad" % (b, np.abs(d.qpos[7:] - q0).max()))
m, d, sid, Rb0 = setup("medium"); p0 = d.site_xpos[sid].copy()
for lap in range(3):
    for c in ([0, .25, 0], [0, .25, .25], [0, 0, .25], [0, 0, 0]):
        _, b, _ = go(m, d, sid, Rb0, p0 + c, 0)
    print("F square lap", lap + 1, "hand home, base %.2f deg" % b)
