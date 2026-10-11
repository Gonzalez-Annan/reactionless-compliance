"""Scheme 6 on the model at rest (D52, D57). What is left after the hand and the base is the tool roll, so slowing
a joint near its limit costs the hand and the base nothing, and with room on every joint it is scheme 5."""
import sys; sys.path.insert(0, "src")
import numpy as np
import teleop as T
from dynamics import kin
from schemes import qdot
from task import Q_LIM, Q_REST


def test_scheme6():
    m, d, *_ = T.setup("medium")
    k = kin(m, d)
    n = np.linalg.svd(np.vstack([T.TW_BASE * k.Jstar[:3], k.W]))[2][-1]
    assert n[6]**2 > 0.99                                # the spare direction is joint 7, not a redundant elbow
    go = lambda sc, u, lim: qdot(sc, k, np.r_[T.V_GO * u, 0, 0, 0], Q_REST.copy(), Q_REST, "3d", 10.0, None,
                                 np.zeros(3), lim, T.TW_BASE)
    wide, slowed = np.c_[Q_REST - 3, Q_REST + 3], 0
    for u in T.DIRS:
        q5, q6 = go(5, u, Q_LIM), go(6, u, Q_LIM)
        assert np.abs(go(6, u, wide) - q5).max() < 1e-12                           # room everywhere: scheme 5
        assert np.linalg.norm(k.Jstar[:3] @ q6 - T.V_GO * u) < 1e-3 * T.V_GO       # the hand still gets its speed
        assert np.linalg.norm(k.W @ q6) < np.linalg.norm(k.W @ q5) + 1e-5          # the base is moved no more
        assert np.abs(q6[:6] - q5[:6]).max() < 2e-3                                # joints 1 to 6 as under scheme 5
        slowed += abs(q6[6]) < 0.95 * abs(q5[6])
    assert slowed > 30      # joint 7 rests 0.49 rad from its limit: slowed on the lines that drive it there
