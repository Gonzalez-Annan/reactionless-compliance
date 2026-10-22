"""The five redundancy-resolution schemes, 3-D and 6-D variants. Owner: P3.

Plan v2, Section 3 (J* = generalized Jacobian, W = base angular rate per joint rate,
q0dot = null-space / posture motion):

  1 dls            qd = J*^T (J* J*^T + lam^2 I)^-1 xd
  2 grad_proj      qd = J*^+ xd + (I - J*^+ J*) KG grad(H),   H = manipulability of J*
  3 impedance      qd = J*^+ xd + (I - J*^+ J*) q0d,  q0d = -KP (q - q_rest) - KD qd_now
  4 rns            3-D: [J*; W] qd = [xd; -KA att]  exact, the one spare DOF left still
                   6-D: min |J* qd - xd|^2 + beta |W qd + KA6 att|^2 + gamma |qd|^2
  5 impedance_rns  as 4, with q0d injected inside the reaction null space
                   (3-D: the spare DOF; 6-D: gamma |qd - q0d|^2)
                   Prior art: Abiko et al. 2006, Dimitrov & Yoshida 2004, Nenchev 2015 -- cite.

DOF budget: 3-D task 3 + 3 reactionless = 6 <= 7, exact. 6-D task 6 + 3 = 9 > 7, so 4 and 5
are a weighted least-squares trade-off; beta large = reactionless first, small = task first.
Sweep beta for Fig. 1:  python -m src.experiments sweep

Shared by all five, so the comparison stays fair:
  * pose servo: xd += K_POSE * (commanded target - hand), the target being the blue sphere
    the operator flies (teleop passes it in; without it the scheme is pure rate control);
  * attitude feedback KA on the base (schemes 4, 5 only -- 1-3 do not know the base floats
    beyond J*);
  * joint geofence: a joint within MARGIN of its limit and still pushing out is locked and
    the rest re-solve.

Interface (src/teleop.py): controller(model, data, xdot, mode[, target]) -> qdot (7,), rad/s.

    python -m src.teleop --controller impedance_rns --mode 6d     # PS4 pad is picked up automatically
    python -m src.experiments gate        # all 5 schemes x 2 modes x 3 servicers, scripted
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.dynamics import base_attitude_error, kin, manipulability

NAMES = ["dls", "grad_proj", "impedance", "rns", "impedance_rns"]   # plan schemes 1-5, session keys
LABELS = {"dls": "1 DLS", "grad_proj": "2 Grad. proj.", "impedance": "3 Impedance",
          "rns": "4 RNS", "impedance_rns": "5 Imp.+RNS"}

# Gains. Starting values from v1 (P2's tuning, v1 DECISIONS D5), re-checked on the root
# models by the scripted gate; log every change in DECISIONS.md.
LAM = 0.02        # DLS damping [m]
KG = 5.0          # gradient-projection gain
KP, KD = 1.0, 0.1 # posture impedance: 1/s, - (velocity level)
KA = 5.0          # base attitude feedback, 1/s (schemes 4, 5, 3-D: free, the task keeps priority)
KA6 = 1.0         # 6-D attitude feedback. With 5, beta >= 1 left the hand stuck 2-5 cm short of a
                  # waypoint (it pulls the hand to level the base) so the trial never ends; with 0
                  # (the plan's bare formula) 4 and 5 ended with MORE base drift than DLS
BETA = 3.0        # 6-D: weight of the reactionless rows. Largest that still finishes the 2-loop task
                  # for both 4 and 5 on medium with KA6 = 1 (grid KA6 x beta, DECISIONS 2026-10-14)
GAMMA = 1e-3      # 6-D: weight of |qd - q0d|. 1e-2 fought the task (14 mm rms vs 5 mm, beta 0.3)
TW = 100.0        # 3-D: task-row weight in the stacked solve (task wins when rows conflict)
K_POSE = 2.0      # 1/s pull toward the commanded target, all schemes
POS_CLIP, ROT_CLIP = 0.05, 0.3   # m, rad: the pull saturates (no lunging at a far target)
MARGIN = np.radians(3.0)         # geofence distance to a joint limit
W_SCALE = 1.0     # 3-D: weight of the reactionless rows against the EPS damping (see _solve)
EPS = 1e-6        # damping in the pseudo-inverses (keeps 2, 3, 4-3d finite at singularities)
I7 = np.eye(7)


def dpinv(A, eps=EPS):
    return A.T @ np.linalg.inv(A @ A.T + eps * np.eye(len(A)))


def pose_error(model, data, target):
    """[position; orientation] error from the hand to the target, world frame, clipped."""
    sid = model.site("end_effector").id
    pos, rot = target
    ep = np.asarray(pos, float) - data.site_xpos[sid]
    er = (rot * Rotation.from_matrix(data.site_xmat[sid].reshape(3, 3)).inv()).as_rotvec()
    clip = lambda e, c: e if np.linalg.norm(e) <= c else e * (c / np.linalg.norm(e))
    return np.r_[clip(ep, POS_CLIP), clip(er, ROT_CLIP)]


class Scheme:
    """One scheme with its gains. Call it like a function; teleop and the scripted task do."""
    accepts_target = True

    def __init__(self, name, beta=BETA, lam=LAM, kg=KG, kp=KP, kd=KD, ka=KA, ka6=KA6, gamma=GAMMA,
                 k_pose=K_POSE, geofence=True, w_scale=None):
        if name not in NAMES:
            raise ValueError(f"unknown scheme {name!r}, use one of {NAMES}")
        self.name, self.beta, self.lam, self.kg = name, beta, lam, kg
        self.kp, self.kd, self.ka, self.ka6, self.gamma, self.k_pose = kp, kd, ka, ka6, gamma, k_pose
        self.geofence = geofence
        self.w_scale = W_SCALE if w_scale is None else w_scale
        self._cache = None   # (model, scratch MjData, scratch for the gradient)

    def __repr__(self):
        return f"Scheme({self.name!r}, beta={self.beta})"

    # -- helpers ----------------------------------------------------------------------
    def _scratch(self, model):
        if self._cache is None or self._cache[0] is not model:
            self._cache = (model, mujoco.MjData(model), mujoco.MjData(model))
        return self._cache[1], self._cache[2]

    def _manip_grad(self, model, d2, qpos, mode, eps=1e-4):
        """Finite-difference gradient of the manipulability of J* (task rows) over the 7 joints."""
        def w(i=None):
            d2.qpos[:] = qpos
            if i is not None:
                d2.qpos[7 + i] += eps
            mujoco.mj_forward(model, d2)
            J = kin(model, d2).Jstar
            return manipulability(J if mode == "6d" else J[:3])
        w0 = w()
        return np.array([(w(i) - w0) / eps for i in range(7)])

    # -- the law ----------------------------------------------------------------------
    def __call__(self, model, data, xdot, mode, target=None):
        s, s2 = self._scratch(model)
        s.qpos[:], s.qvel[:] = data.qpos, data.qvel
        mujoco.mj_forward(model, s)
        k = kin(model, s)

        xd = np.asarray(xdot, float).copy()
        if target is not None and self.k_pose:
            xd = xd + self.k_pose * pose_error(model, s, target)
        if mode == "3d":
            xd[3:] = 0.0

        q, q_rest = s.qpos[7:], model.key("home").qpos[7:]
        att = base_attitude_error(model, s)
        grad = self._manip_grad(model, s2, s.qpos.copy(), mode) if self.name == "grad_proj" else None
        q0 = -self.kp * (q - q_rest) - self.kd * s.qvel[6:]
        lim = model.jnt_range[1:]
        return self.solve(k, xd, q0, att, mode, grad, lim if self.geofence else None, q)

    def solve(self, k, xd, q0, att, mode, grad=None, lim=None, q=None):
        """Geofence around _solve: lock joints at a limit that still push outward, re-solve."""
        free = np.ones(7, bool)
        for _ in range(8):
            qd = self._solve(k.Jstar * free, k.W * free, xd, q0 * free, att, mode, grad) * free
            if lim is None:
                return qd
            bad = free & (((q >= lim[:, 1] - MARGIN) & (qd > 0)) | ((q <= lim[:, 0] + MARGIN) & (qd < 0)))
            if not bad.any():
                return qd
            free &= ~bad
        return qd

    def _solve(self, Jstar, W, xd, q0, att, mode, grad):
        J = Jstar if mode == "6d" else Jstar[:3]
        v = xd[:len(J)]
        n = self.name
        if n == "dls":
            return J.T @ np.linalg.solve(J @ J.T + self.lam**2 * np.eye(len(J)), v)
        if n in ("grad_proj", "impedance"):
            Jp = dpinv(J)
            z = self.kg * grad if n == "grad_proj" else q0
            return Jp @ v + (I7 - Jp @ J) @ z
        z = np.zeros(7) if n == "rns" else q0
        if mode == "3d":   # 3 task + 3 reactionless = 6 rows, exact; z moves the one spare DOF
            # Singularity-robust on purpose: W is ~100x smaller than J*, so the EPS damping
            # softens the reactionless rows where the stack loses rank (edge of the reactionless
            # reach). Scaling W to unit rows made it exact (W qd ~ 0) but then 3-D 4 and 5
            # saturated at 1 rad/s and failed the loop on all three servicers (DECISIONS 14 Oct);
            # this form finishes on medium / large with 0.000-0.003 deg final base error.
            sw = self.w_scale
            S = np.vstack([TW * J, sw * W])
            Sp = dpinv(S)
            return Sp @ np.r_[TW * v, -sw * self.ka * att] + (I7 - Sp @ S) @ z
        # 6-D: 9 rows > 7 joints -> weighted least squares, beta trades task vs base
        A = J.T @ J + self.beta * W.T @ W + self.gamma * I7
        return np.linalg.solve(A, J.T @ v - self.beta * self.ka6 * W.T @ att + self.gamma * z)


# Per-scheme gains from the tuning grid (python -m src.experiments tune -> data/results/tuning.csv).
# Rule fixed before the runs: per scheme, its own knob; among values that finish the task in both
# modes, the lowest summed rms hand error, ties (within 10 %) to the lower peak base drift.
# Medium servicer, 2 loops, 3-D + 6-D. Defaults above are the pre-tuning (v1 D5) values.
TUNED = {
    "dls": {"lam": 0.01},
    "grad_proj": {"kg": 1.0},
    "impedance": {"kp": 4.0},
    "rns": {"ka": 1.0},
    "impedance_rns": {"kp": 1.0},
}


def make_scheme(name, **gains):
    """A scheme with non-default gains, e.g. make_scheme("rns", beta=100) for the beta-sweep."""
    return Scheme(name, **{**TUNED.get(name, {}), **gains})


SCHEMES = {name: make_scheme(name) for name in NAMES}


def main():
    print("Scripted experiments moved to src/experiments.py:\n"
          "  python -m src.experiments tune | gate | sweep | all")


if __name__ == "__main__":
    main()