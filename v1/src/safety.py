"""Pose-dependent contact-velocity limit. [P4]

m_eff along u = 1 / (u' L^-1 u), with L^-1 = J M^-1 J' over the DOFs free to move at impact.
Joints locked (stiff servo) is the conservative case: only the 6 base DOFs give way.
v_max follows the ISO/TS 15066 energy model: v = F_max / sqrt(mu k), mu = two-body reduced mass.
Body-region numbers below are the chest values as recalled from ISO/TS 15066 Annex A.
ponytail: P4 must check them against the standard before the paper cites them.
"""
import numpy as np
import mujoco
from dynamics import mass_matrix, jacobian

F_MAX, K_BODY, M_HUMAN = 280.0, 25e3, 40.0   # N (transient), N/m, kg


def m_eff(m, d, u=None, locked=True):
    """Effective mass at the hand along u (default: tool approach axis)."""
    M, J = mass_matrix(m, d), jacobian(m, d)[:3]
    if u is None:
        u = d.site_xmat[m.site("ee").id].reshape(3, 3)[:, 2]
    n = 6 if locked else m.nv
    return 1.0 / (u @ J[:, :n] @ np.linalg.solve(M[:n, :n], J[:, :n].T) @ u)


def v_max(meff):
    mu = 1.0 / (1.0 / meff + 1.0 / M_HUMAN)
    return F_MAX / np.sqrt(mu * K_BODY)


def vmax_map(m, q_rest, n=41):
    """v_max over shoulder (j2) x elbow (j4), other joints at rest."""
    d = mujoco.MjData(m)
    q2, q4 = np.linspace(-1.5, 1.5, n), np.linspace(0.0, 2.6, n)
    out = np.zeros((n, n))
    for a, s in enumerate(q2):
        for b, e in enumerate(q4):
            d.qpos[7:] = q_rest
            d.qpos[8], d.qpos[10] = s, e
            mujoco.mj_forward(m, d)
            out[b, a] = v_max(m_eff(m, d))
    return q2, q4, out
