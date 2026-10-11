"""The redundancy-resolution schemes, velocity level, 3-D and 6-D variants. [P3]

1 DLS   2 gradient projection (manipulability)   3 null-space impedance
4 RNS   5 impedance + RNS   6 = 5 with joint limits inside the solve (D52, envelope tests only)
7 = 6 with the reactionless rows a cost: base tilt traded for joint room (3-D, envelope tests only), see soft()
"""
from types import SimpleNamespace
import numpy as np

NAMES = {1: "DLS", 2: "Grad. proj.", 3: "Impedance", 4: "RNS", 5: "Imp.+RNS", 6: "Imp.+RNS+limits", 7: "Imp.+RNS+tilt"}
LAM, KG, KP, GAMMA, KA = 0.02, 5.0, 1.0, 1e-2, 5.0   # tuning log: DECISIONS.md D5
TW, MARGIN = 100.0, 0.05   # task-over-base weight once joints lock; rad kept clear of a joint limit (D13)
BAND = 0.5   # rad short of the margin where scheme 6 starts to weigh a joint out. Not tuned, first value tried
TILT, SOFT = np.radians(1.5), 1e-3   # scheme 7: base tilt it may spend (teleop rejects at 2 deg); how soft the base rows get. Not tuned, first values tried
I7 = np.eye(7)


def dpinv(A, eps=1e-6):
    return A.T @ np.linalg.inv(A @ A.T + eps * np.eye(len(A)))


def soft(c, att):
    """Scheme 7, the law (3-D). Hand rows as in scheme 5. Joints weighed by their room as in scheme 6 (c in; r = its least,
    squared), but only by the share of the tilt budget still left: c -> 1 - (1 - tilt / TILT) (1 - c), all 1 once it is spent.
    The reactionless rows are a cost, not a constraint: min |qd - q0|^2 + |W qd + r KA att|^2 / lam over the hand's
    solutions, lam = SOFT (1 - least c^2). Room everywhere or budget spent: lam = 0, the rows are hard as in scheme 5.
    KA pulls the attitude back as far as r allows. qdot hardens the rows when the plan, held 1 / KA s, would leave TILT."""
    r, c = c.min()**2, 1 - np.clip(1 - np.linalg.norm(att) / TILT, 0, 1) * (1 - c)
    return c, SOFT * (1 - c.min()**2), r


def qdot(scheme, k, xd, q, q_rest, mode="6d", beta=10.0, grad=None, att=np.zeros(3), lim=None, tw=TW):
    """Joint geofence around _solve: a joint at its limit and still pushing outward is locked and the
    rest re-solve. tw > 1: the hand task keeps priority, so what the locked joints can no longer cancel
    goes to the base. tw < 1: the base keeps priority and the hand falls behind instead (teleop). lim = (7, 2) array of [lo, hi], or None for no limits."""
    free, c, lam, r = np.ones(7, bool), np.ones(7), 0.0, 1.0
    for _ in range(8):
        kk = SimpleNamespace(Jstar=k.Jstar * free * c, W=k.W * free * c)
        qd = _solve(5 if scheme > 5 else scheme, kk, xd, q, q_rest, mode, beta, grad, r * att, tw, lam) * free * c
        if lim is None:
            break
        if scheme > 5 and (c == 1).all() and r == 1:
            # weighted least norm (Chan and Dubey 1995): a joint heading for its limit costs more the closer it is, so
            # the others take over before it gets there. Velocity level, in the spirit of Cocuzza 2011, not his solver.
            # ponytail: weights from one unweighted pass, a joint that turns round after reweighting keeps its weight
            room = np.where(qd > 0, lim[:, 1] - q, q - lim[:, 0]) - MARGIN
            c = np.sqrt(np.clip(room / BAND, 0.05, 1))
            if (c < 1).any():
                if scheme == 7:
                    c, lam, r = soft(c, att)
                continue
        if lam and np.linalg.norm(att + k.W @ qd / KA) > TILT:   # scheme 7 never plans to pass the budget
            c, lam = np.ones(7), 0.0
            continue
        # ponytail: hard lock at the margin, no smooth slow-down; add a velocity ramp if operators feel a jolt
        bad = free & (((q >= lim[:, 1] - MARGIN) & (qd > 0)) | ((q <= lim[:, 0] + MARGIN) & (qd < 0)))
        if not bad.any():
            break
        free &= ~bad
    return qd


def _solve(scheme, k, xd, q, q_rest, mode, beta, grad, att, tw=TW, lam=0.0):
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
    if mode == "3d":   # 3 task + 3 reactionless = 6 <= 7: exact, one spare DOF. At rest it is the tool roll, joint 7 (D57)
        S = np.vstack([tw * J, k.W])   # row weight changes nothing while both fit; the heavier side wins when they do not
        Sp = dpinv(S, 1e-6 + lam * (np.arange(6) > 2))   # lam > 0 (scheme 7): the three base rows are a cost, see soft()
        return Sp @ np.r_[tw * xd, -KA * att] + (I7 - Sp @ S) @ q0
    # 6-D: 6 + 3 = 9 > 7, so weighted least squares; beta large = reactionless first
    A = J.T @ J + beta * k.W.T @ k.W + GAMMA * I7
    return np.linalg.solve(A, J.T @ xd - beta * KA * k.W.T @ att + GAMMA * q0)
