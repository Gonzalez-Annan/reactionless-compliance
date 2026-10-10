"""The five redundancy-resolution schemes, velocity level, 3-D and 6-D variants. [P3]

1 DLS   2 gradient projection (manipulability)   3 null-space impedance
4 RNS   5 impedance + RNS
"""
from types import SimpleNamespace
import numpy as np

NAMES = {1: "DLS", 2: "Grad. proj.", 3: "Impedance", 4: "RNS", 5: "Imp.+RNS"}
LAM, KG, KP, GAMMA, KA = 0.02, 5.0, 1.0, 1e-2, 5.0   # tuning log: DECISIONS.md D5
TW, MARGIN = 100.0, 0.05   # task-over-base weight once joints lock; rad kept clear of a joint limit (D13)
I7 = np.eye(7)


def dpinv(A, eps=1e-6):
    return A.T @ np.linalg.inv(A @ A.T + eps * np.eye(len(A)))


def qdot(scheme, k, xd, q, q_rest, mode="6d", beta=10.0, grad=None, att=np.zeros(3), lim=None, tw=TW):
    """Joint geofence around _solve: a joint at its limit and still pushing outward is locked and the
    rest re-solve. tw > 1: the hand task keeps priority, so what the locked joints can no longer cancel
    goes to the base. tw < 1: the base keeps priority and the hand falls behind instead (teleop). lim = (7, 2) array of [lo, hi], or None for no limits."""
    free = np.ones(7, bool)
    for _ in range(7):
        kk = SimpleNamespace(Jstar=k.Jstar * free, W=k.W * free)
        qd = _solve(scheme, kk, xd, q, q_rest, mode, beta, grad, att, tw) * free
        if lim is None:
            break
        # ponytail: hard lock at the margin, no smooth slow-down; add a velocity ramp if operators feel a jolt
        bad = free & (((q >= lim[:, 1] - MARGIN) & (qd > 0)) | ((q <= lim[:, 0] + MARGIN) & (qd < 0)))
        if not bad.any():
            break
        free &= ~bad
    return qd


def _solve(scheme, k, xd, q, q_rest, mode, beta, grad, att, tw=TW):
    """k = dynamics.kin(); xd = desired [v; w]; grad = manipulability gradient (scheme 2);
    att = base attitude error (rotation vector, body frame), fed back by schemes 4 and 5."""
    J = k.Jstar if mode == "6d" else k.Jstar[:3]
    xd = xd[:len(J)]
    q0 = -KP * (q - q_rest)
    if scheme == 1:
        return J.T @ np.linalg.solve(J @ J.T + LAM**2 * np.eye(len(J)), xd)
    if scheme in (2, 3):
        Jp = dpinv(J)
        return Jp @ xd + (I7 - Jp @ J) @ (q0 if grad is None else KG * grad)
    if scheme == 4:
        q0 = np.zeros(7)
    if mode == "3d":   # 3 task + 3 reactionless = 6 <= 7: exact, one spare DOF for q0
        S = np.vstack([tw * J, k.W])   # row weight changes nothing while both fit; the heavier side wins when they do not
        Sp = dpinv(S)
        return Sp @ np.r_[tw * xd, -KA * att] + (I7 - Sp @ S) @ q0
    # 6-D: 6 + 3 = 9 > 7, so weighted least squares; beta large = reactionless first
    A = J.T @ J + beta * k.W.T @ k.W + GAMMA * I7
    return np.linalg.solve(A, J.T @ xd - beta * KA * k.W.T @ att + GAMMA * q0)
