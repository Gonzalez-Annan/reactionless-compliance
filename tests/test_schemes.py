"""Tests for the five schemes. Owner: P3."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.dynamics import kin
from src.schemes import NAMES, SCHEMES, make_scheme
from src.session import SCHEMES as SESSION_KEYS
from src.task import run_scripted
from tests.test_momentum import load

XD = np.array([0.03, -0.02, 0.01, 0.05, -0.1, 0.08])


def state():
    m, d = load("medium")
    d.ctrl[:] = 0.3
    for _ in range(400):            # move off the keyframe so the base is not at identity
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    return m, d


def test_keys_match_the_session_runner():
    assert list(SCHEMES) == NAMES == SESSION_KEYS


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mode", ["3d", "6d"])
def test_returns_finite_seven_vector(name, mode):
    m, d = state()
    qd = SCHEMES[name](m, d, XD, mode)
    assert qd.shape == (7,) and np.all(np.isfinite(qd))


@pytest.mark.parametrize("name", ["dls", "grad_proj", "impedance"])
def test_task_tracked(name):
    """Without a target the scheme is pure rate control: J* qd = xd (DLS: nearly)."""
    m, d = state()
    k = kin(m, d)
    for mode, rows in (("3d", 3), ("6d", 6)):
        qd = make_scheme(name, geofence=False)(m, d, XD, mode)
        tol = 2e-3 if name == "dls" else 1e-5
        xd = XD.copy()
        if mode == "3d":
            xd[3:] = 0
        assert np.allclose(k.Jstar[:rows] @ qd, xd[:rows], atol=tol)


@pytest.mark.parametrize("name", ["rns", "impedance_rns"])
def test_3d_tracks_the_task_and_is_near_reactionless(name):
    """Hand exact; base rate at least 20x below DLS's for the same motion (measured ~25x at
    this pose, where W's weakest direction has sigma ~1e-3). Not exactly 0 by design: the
    damping softens the reactionless rows near rank loss (see _solve)."""
    m, d = state()
    k = kin(m, d)
    qd = make_scheme(name, ka=0.0, geofence=False)(m, d, XD, "3d")
    qd_dls = make_scheme("dls", geofence=False)(m, d, XD, "3d")
    assert np.allclose(k.Jstar[:3] @ qd, XD[:3], atol=1e-5)
    assert np.linalg.norm(k.W @ qd) < np.linalg.norm(k.W @ qd_dls) / 20


def test_6d_beta_trades_task_for_base():
    """Larger beta -> less base rotation and more task error (the Fig. 1 trade-off)."""
    m, d = state()
    k = kin(m, d)
    task_err, base_rate = [], []
    for beta in (0.01, 1, 100):
        qd = make_scheme("rns", beta=beta, geofence=False)(m, d, XD, "6d")
        task_err.append(np.linalg.norm(k.Jstar @ qd - XD))
        base_rate.append(np.linalg.norm(k.W @ qd))
    assert task_err[0] < task_err[1] < task_err[2]
    assert base_rate[0] > base_rate[1] > base_rate[2]


def test_pose_servo_pulls_toward_target():
    m, d = state()
    sid = m.site("end_effector").id
    p = d.site_xpos[sid] + np.array([0.01, 0, 0])
    R = Rotation.from_matrix(d.site_xmat[sid].reshape(3, 3))
    qd = SCHEMES["dls"](m, d, np.zeros(6), "6d", target=(p, R))
    assert (kin(m, d).Jstar @ qd)[0] > 0.015      # K_POSE * 1 cm = 0.02 m/s, toward +x


def test_geofence_stops_a_joint_at_its_limit():
    m, d = load("medium")
    d.qpos[8] = m.jnt_range[2, 1] - 0.01          # j2 just inside its upper limit
    mujoco.mj_forward(m, d)
    for xd in np.eye(6) * 0.05:
        qd = SCHEMES["dls"](m, d, xd, "6d")
        assert qd[1] <= 0                          # never pushes further out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mode", ["3d", "6d"])
def test_gate_scripted_loop_completes_on_medium(name, mode):
    """Week 7 gate on the study servicer (all three sizes: python -m src.schemes --gate)."""
    r = run_scripted("medium", mode, name)
    assert r["completed"], r
    assert r["final_antenna_err_deg"] < 5.0
