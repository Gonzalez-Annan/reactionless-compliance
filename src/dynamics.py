"""Free-floating dynamics: H_b, H_bm, generalized Jacobian J*, reaction null space. Owner: P2.

Ported by P3 from v1/src/dynamics.py (P2's verified version) to the root models, so the
root schemes and the momentum gate can run. P2: please review; the maths is unchanged.

State layout (MuJoCo free joint, see README):
    qvel = [v_b (3, world), w_b (3, base frame), qdot (7)]
Every matrix MuJoCo returns (M, site Jacobians) uses the same convention, so they are
consistent with each other.

With zero initial momentum (the base starts at rest, gravity is off):
    H_b v_b + H_bm qdot = 0          ->  v_b = V qdot,   V = -H_b^-1 H_bm
    xdot = J_b v_b + J_m qdot        ->  xdot = J* qdot, J* = J_m + J_b V
Base angular rate per unit joint rate:  W = V[3:]   (3 x 7, base frame).
Reactionless (no base attitude change) means W qdot = 0. W is H_bm,ang with the
linear-momentum rows eliminated; N(H_bm,ang) alone is only right when the base frame
sits at the system centre of mass (v1 DECISIONS.md D3).
"""
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

EE_SITE = "end_effector"


def mass_matrix(model, data):
    """Dense nv x nv joint-space inertia. Needs mj_forward (or mj_step) first."""
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return M


def site_jacobian(model, data, site=EE_SITE):
    """6 x nv site Jacobian, world frame, rows [linear; angular] (same as the operator twist)."""
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, model.site(site).id)
    return np.vstack([jp, jr])


def kin(model, data, site=EE_SITE):
    """Everything the schemes need at the current state. Call after mj_forward / mj_step."""
    M = mass_matrix(model, data)
    H_b, H_bm = M[:6, :6], M[:6, 6:]
    J = site_jacobian(model, data, site)
    V = -np.linalg.solve(H_b, H_bm)          # base twist per unit joint rate
    Jstar = J[:, 6:] + J[:, :6] @ V          # generalized Jacobian, 6 x 7
    return SimpleNamespace(M=M, H_b=H_b, H_bm=H_bm, J=J, V=V, Jstar=Jstar, W=V[3:])


def rns_projector(W):
    """Projector onto the reaction null space (joint motions that leave the base attitude alone).
    W is 3 x 7 and full rank away from singular poses, so the null space is 4-D."""
    return np.eye(W.shape[1]) - np.linalg.pinv(W) @ W


def manipulability(J):
    """Yoshikawa manipulability sqrt(det(J J^T)); pass J* or J*[:3]."""
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def singularity_distance(J):
    """Smallest singular value: zero at a (dynamic) singularity."""
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def momentum_residual(model, data):
    """H_b v_b + H_bm qdot, i.e. the first 6 rows of M qvel. Stays 0 from rest (the 1e-6 gate)."""
    return (mass_matrix(model, data) @ data.qvel)[:6]


def base_attitude_error(model, data, quat0=None):
    """Base rotation since quat0 (default: the "home" keyframe) as a rotation vector in the
    base frame -- the same frame as W qdot, so d/dt of it is ~ W qdot for small angles."""
    if quat0 is None:
        quat0 = model.key("home").qpos[3:7]
    R0 = Rotation.from_quat(quat0, scalar_first=True)
    R = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
    return (R0.inv() * R).as_rotvec()
