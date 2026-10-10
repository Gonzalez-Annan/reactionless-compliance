"""Free-floating dynamics: mass-matrix blocks, generalized Jacobian, reaction null space. [P2]

qvel = [base linear (world), base angular (body frame), 7 joint rates].
With zero initial momentum:  H_b v_b + H_bm qd = 0.
"""
from types import SimpleNamespace
import numpy as np
import mujoco


def mass_matrix(m, d):
    # mj_mulM column by column: mj_fullM's inertia argument changed name across MuJoCo versions
    M, e = np.zeros((m.nv, m.nv)), np.eye(m.nv)
    for i in range(m.nv):
        mujoco.mj_mulM(m, d, M[i], e[i])
    return M


def jacobian(m, d, site="ee"):
    """6 x nv site Jacobian, world frame: [linear; angular]."""
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, m.site(site).id)
    return np.vstack([jp, jr])


def kin(m, d):
    """Call after mj_forward. Returns everything the schemes need."""
    M = mass_matrix(m, d)
    H_b, H_bm = M[:6, :6], M[:6, 6:]
    J = jacobian(m, d)
    V = -np.linalg.solve(H_b, H_bm)            # v_b = V qd
    Jstar = J[:, 6:] + J[:, :6] @ V            # xdot = Jstar qd
    # Base angular rate per unit joint rate. Reactionless (zero attitude change) means W qd = 0.
    # This is H_bm,ang with the linear-momentum rows eliminated; N(H_bm,ang) alone is only
    # correct when the base frame sits at the system centre of mass (see DECISIONS.md D3).
    W = V[3:]
    return SimpleNamespace(M=M, H_b=H_b, H_bm=H_bm, J=J, Jstar=Jstar, W=W)


def rns_projector(W):
    """Projector onto the reaction null space: joint motions that leave base attitude alone."""
    return np.eye(W.shape[1]) - np.linalg.pinv(W) @ W


def manipulability(Jstar):
    return np.sqrt(max(np.linalg.det(Jstar @ Jstar.T), 0.0))


def singularity_distance(Jstar):
    """Smallest singular value of J*: zero at a dynamic singularity."""
    return np.linalg.svd(Jstar, compute_uv=False)[-1]


def momentum_residual(m, d):
    """[linear (3), angular (3)] generalized momentum of the base rows; stays 0 from rest."""
    return (mass_matrix(m, d) @ d.qvel)[:6]
