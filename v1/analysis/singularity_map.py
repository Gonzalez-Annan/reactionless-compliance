"""Is there a pose from which the hand moves freely with the base held level? [P2]

    python analysis/singularity_map.py     # about a minute, prints only

Score = smallest singular value of the hand Jacobian restricted to the joint motions that do not turn the base.
Near zero = the hand is stuck unless the base is allowed to turn. Result and what it means: DECISIONS.md D16.
ponytail: random poses plus a hill-climb, not a global optimum; the collision filter rejected nothing, so
check the models have arm-base contact enabled before trusting individual poses.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np, mujoco, scipy.linalg as sl
import teleop
from teleop import setup, ray
from dynamics import kin
from task import Q_LIM, Q_REST
np.set_printoptions(precision=2, suppress=True, linewidth=200)
U = np.r_[np.eye(3), -np.eye(3)]


def sig(m, d, q):
    """Smallest singular value of the hand Jacobian on base-level joint motions; 0 if the pose collides."""
    d.qpos[7:] = q
    mujoco.mj_forward(m, d)
    if d.ncon:
        return 0.0
    k = kin(m, d)
    return np.linalg.svd(k.Jstar[:3] @ sl.null_space(k.W, rcond=1e-9), compute_uv=False)[-1]


def reach(q, bm):
    teleop.Q_REST, teleop.BASE_MAX = q, bm
    m, d, sid, Rb0 = setup("medium"); p0 = d.site_xpos[sid].copy()
    return np.array([(ray(m, d, sid, Rb0, u, 5, "3d") - p0) @ u for u in U])


for size in ("small", "medium", "large"):
    m, d, *_ = setup(size)
    print(size, "at the current rest pose: %.3f" % sig(m, d, Q_REST))
m, d, *_ = setup("medium")
rng, t = np.random.default_rng(0), time.time()
Q = rng.uniform(Q_LIM[:, 0] + 0.3, Q_LIM[:, 1] - 0.3, (6000, 7))
S = np.array([sig(m, d, q) for q in Q])
print("6000 poses in %.0f s; collide %.0f%%; score percentiles 50/90/99/max:" % (time.time() - t, 100 * (S == 0).mean()),
      np.round(np.percentile(S, [50, 90, 99, 100]), 3))
best = []
for i in np.argsort(S)[-4:]:                 # hill-climb the four best
    q, s = Q[i].copy(), S[i]
    for _ in range(300):
        q2 = np.clip(q + rng.normal(0, 0.08, 7), Q_LIM[:, 0] + 0.3, Q_LIM[:, 1] - 0.3)
        s2 = sig(m, d, q2)
        if s2 > s:
            q, s = q2, s2
    best.append((s, q))
print("reach +x +y +z -x -y -z, base limit 0.5 deg then 2 deg")
print("current rest  score 0.043", reach(Q_REST, 0.5), reach(Q_REST, 2.0))
for s, q in sorted(best, key=lambda b: -b[0]):
    print("score %.3f q" % s, q, "\n   ", reach(q, 0.5), reach(q, 2.0))
