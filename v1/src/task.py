"""Closed-loop task, trial runner and per-trial metrics. [P1]

The hand traces a circle in the inertial frame and returns to its start, `loops` times.
6-D mode also commands a tool tilt that returns to zero. The default operator is scripted
(feed-forward + proportional); teleop.py swaps in a gamepad.
"""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

from dynamics import kin, manipulability, singularity_distance, momentum_residual
from schemes import qdot, MARGIN, TW

MODELS = Path(__file__).parent.parent / "models"
# best of 40 random postures by smallest singular value of J* along the path (DECISIONS.md D4)
Q_REST = np.array([1.59, -0.3, 0.07, -1.52, 0.07, 1.42, -2.41])   # ready pose: largest 2 deg tilt-bounded reach of 160 (analysis/envelope/ready.py, D30); was [-0.26, 0.88, -0.17, 1.5, 0.57, 0.83, -0.52], hand 1 cm above the bus
Q_LIM = np.array([[-2.9, 2.9], [-2.0, 2.0]] * 3 + [[-2.9, 2.9]])   # rad; z joints 166 deg, y joints 115 deg (D13)
T_LOOP, RADIUS, TILT = 15.0, 0.10, 0.25      # s, m, rad
KX, KO, DECIM, QD_MAX = 4.0, 4.0, 20, 2.0    # operator gains, control = 100 Hz, rad/s clip


def reference(t, T, p0, R0):
    s = min(t, T) / T                       # smoothstep in time: starts and ends at rest
    th = 2 * np.pi * (T / T_LOOP) * s * s * (3 - 2 * s)
    thd = 2 * np.pi / T_LOOP * 6 * s * (1 - s)
    p = p0 + RADIUS * np.array([1 - np.cos(th), 0, np.sin(th)])     # x-z plane: from the ready pose a loop out along y runs into the joint limits (D32)
    v = RADIUS * thd * np.array([np.sin(th), 0, np.cos(th)])
    R = Rot.from_rotvec([TILT * np.sin(th), 0, 0]).as_matrix() @ R0
    w = np.array([TILT * np.cos(th) * thd, 0, 0])
    return p, v, R, w


def scripted_operator(t, p, R, ref):
    pr, vr, Rr, wr = ref
    return np.r_[vr + KX * (pr - p), wr + KO * Rot.from_matrix(Rr @ R.T).as_rotvec()]


def manip_grad(m, d2, qpos, mode, eps=1e-4):
    def w(i=None):
        d2.qpos[:] = qpos
        if i is not None:
            d2.qpos[7 + i] += eps
        mujoco.mj_forward(m, d2)
        J = kin(m, d2).Jstar
        return manipulability(J if mode == "6d" else J[:3])
    w0 = w()
    return np.array([(w(i) - w0) / eps for i in range(7)])


def run_trial(size="medium", scheme=1, mode="6d", beta=10.0, loops=2,
              operator=scripted_operator, on_step=None, log=None, lim=Q_LIM, tw=TW):
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"ff_{size}.xml"))
    if lim is not None:                      # hard stops in the physics as well as the controller's geofence
        m.jnt_range[1:], m.jnt_limited[1:] = lim, 1
    d, d2 = mujoco.MjData(m), mujoco.MjData(m)
    d.qpos[7:] = Q_REST
    mujoco.mj_forward(m, d)
    sid = m.site("ee").id
    p0, R0 = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    Rb0 = Rot.from_quat(d.qpos[[4, 5, 6, 3]])
    T = loops * T_LOOP
    rows = []
    for i in range(int((T + 1.0) / m.opt.timestep)):     # 1 s of hold after the last loop
        if i % DECIM == 0:
            mujoco.mj_forward(m, d)
            k = kin(m, d)
            p, R = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
            ref = reference(d.time, T, p0, R0)
            xd = operator(d.time, p, R, ref)
            q = d.qpos[7:].copy()
            grad = manip_grad(m, d2, d.qpos, mode) if scheme == 2 else None   # scheme 3 passes None
            Rb = Rot.from_quat(d.qpos[[4, 5, 6, 3]])
            qd = qdot(scheme, k, xd, q, Q_REST, mode, beta, grad, (Rb0.inv() * Rb).as_rotvec(), lim, tw)
            d.ctrl[:] = qd / max(1.0, np.abs(qd).max() / QD_MAX)   # scale, not clip: keeps direction
            Jt = k.Jstar if mode == "6d" else k.Jstar[:3]
            base = (Rb0.inv() * Rb).magnitude()
            res = momentum_residual(m, d)
            rows.append([d.time, np.linalg.norm(ref[0] - p),
                         Rot.from_matrix(ref[2] @ R.T).magnitude(), np.degrees(base),
                         np.linalg.norm(q - Q_REST), singularity_distance(Jt), manipulability(Jt),
                         np.abs(res[:3]).max(), np.abs(res[3:]).max()]
                        + ([0, 0] if lim is None else
                           [((q <= lim[:, 0] + MARGIN) | (q >= lim[:, 1] - MARGIN)).any(),
                            max(0, (q - lim[:, 1]).max(), (lim[:, 0] - q).max())]))
            if on_step and on_step(m, d, ref) is False:
                break
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            break
    a = np.array(rows)
    if log:
        np.savetxt(log, a, delimiter=",", comments="", header=
                   "t,pos_err_m,ori_err_rad,base_deg,q_dist_rad,sigma_min,manip,res_lin,res_ang,at_limit,lim_viol_rad")
    run = a[a[:, 0] <= T]
    return dict(size=size, scheme=scheme, mode=mode, beta=beta,
                stable=bool(np.isfinite(d.qpos).all() and a[-1, 0] >= T),
                rms_pos_mm=1e3 * np.sqrt((run[:, 1]**2).mean()),
                rms_ori_deg=np.degrees(np.sqrt((run[:, 2]**2).mean())),
                max_base_deg=run[:, 3].max(), final_base_deg=a[-1, 3],
                q_return_rad=a[-1, 4], min_sigma=run[:, 5].min(), mean_manip=run[:, 6].mean(),
                res_lin=a[:, 7].max(), res_ang=a[:, 8].max(),
                limit_frac=a[:, 9].mean(), lim_viol_rad=a[:, 10].max())
