"""Scheme 7 = scheme 5 while every joint has room. A joint short of room and tilt budget left: the base is let go a little and that
joint slows far more than under scheme 6. Budget spent, or a plan that would leave it: the base rows are hard again. The hand
gets its velocity throughout, exactly: only the base rows are a cost."""
import sys; sys.path.insert(0, "src")
from types import SimpleNamespace
import numpy as np
from schemes import qdot, MARGIN, BAND, KA, TILT
from task import Q_LIM


def test_scheme7():
    rng = np.random.default_rng(0)
    k = SimpleNamespace(Jstar=rng.normal(size=(6, 7)), W=0.005 * rng.normal(size=(3, 7)))   # W about the size it has on the servicer
    xd, q, z = np.r_[0.1, 0.05, -0.08, 0, 0, 0], np.zeros(7), np.zeros(3)
    a = lambda sc, q, att=z, s=1: qdot(sc, k, s * xd, q, q, "3d", att=att, lim=Q_LIM)
    hand = lambda qd, s=1: np.abs(k.Jstar[:3] @ qd - s * xd[:3]).max() < 1e-9 * s   # 1e-10 as built, 5e-9 were its rows soft too
    t = 0.5 * TILT * np.array([0.6, 0, 0.8])
    assert np.allclose(a(5, q), a(7, q))                       # room everywhere, base level: the same controller
    assert np.allclose(a(5, q, t), a(7, q, t)) and t @ k.W @ a(7, q, t) < -0.9 * KA * t @ t   # room, base tilted: pulled back
    j = int(np.argmax(abs(a(5, q)))); q = q.copy()
    q[j] = (Q_LIM[j, 1] - MARGIN - 0.1) * np.sign(a(5, q)[j])   # 0.1 rad of room left, and scheme 5 drives into it
    q6, q7 = a(6, q), a(7, q)
    assert abs(q7[j]) < 0.5 * abs(q6[j]) and hand(q7)           # the joint is spared, the hand is not
    assert np.linalg.norm(k.W @ q7) > 1e-4 > np.linalg.norm(k.W @ q6)   # paid for in base rate, which scheme 6 does not spend
    u = k.W @ q7 / np.linalg.norm(k.W @ q7)                     # the way the trade tilts the base
    for f, s in ((0.5, 1), (0.9, 1), (0.99, 1), (0.5, 110), (0.5, 200)):   # part of the budget gone that way: the plan stays inside it.
        qd = a(7, q, f * TILT * u, s)                           # x110, x200 (an 11, a 20 m/s hand): the soft rows alone would plan 1.01, 1.4 TILT
        assert hand(qd, s) and np.linalg.norm(f * TILT * u + k.W @ qd / KA) <= TILT
    # budget spent, or overspent: scheme 5's hard rows, the pull back toward level cut to the room the joint has left (0.1 of BAND)
    for t in (TILT * u, 1.3 * TILT * u):
        assert np.allclose(a(7, q, t), a(5, q, 0.1 / BAND * t)) and u @ k.W @ a(7, q, t) < 0
