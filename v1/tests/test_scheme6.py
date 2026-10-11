"""Scheme 6 = scheme 5 until a joint heads for a limit, then that joint slows (not by much) and the hand still gets its velocity."""
import sys; sys.path.insert(0, "src")
from types import SimpleNamespace
import numpy as np
from schemes import qdot, MARGIN
from task import Q_LIM


def test_scheme6():
    rng = np.random.default_rng(0)
    k = SimpleNamespace(Jstar=rng.normal(size=(6, 7)), W=rng.normal(size=(3, 7)))
    xd, q = np.r_[0.1, 0.05, -0.08, 0, 0, 0], np.zeros(7)
    a = lambda sc, q: qdot(sc, k, xd, q, q, "3d", lim=Q_LIM)
    assert np.allclose(a(5, q), a(6, q))                       # far from every limit: the same controller
    j = int(np.argmax(abs(a(5, q)))); q = q.copy()
    q[j] = (Q_LIM[j, 1] - MARGIN - 0.1) * np.sign(a(5, q)[j])   # 0.1 rad of room left, and scheme 5 drives into it
    q5, q6 = a(5, q), a(6, q)
    assert q5[j] * q[j] > 0 and abs(q6[j]) < abs(q5[j])   # the joint is spared, a little: hand + base take 6 of 7 joints, one is left to spare it with
    assert np.allclose(k.Jstar[:3] @ q6, xd[:3], atol=1e-3) and np.allclose(k.W @ q6, 0, atol=1e-3)   # hand and base still served
