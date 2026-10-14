"""Gates: momentum (R3), J* against simulated hand velocity, DOF argument, 3-D reactionless (R4)."""
import sys
from pathlib import Path
import numpy as np
import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dynamics import kin, jacobian, momentum_residual, rns_projector  # noqa: E402
from task import MODELS, Q_REST, run_trial  # noqa: E402

SIZES = ["small", "medium", "large"]


def load(size):
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"ff_{size}.xml"))
    d = mujoco.MjData(m)
    d.qpos[7:] = Q_REST
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize("size", SIZES)
def test_momentum(size):
    """Random joint-rate commands for 10 s from rest; base-row momentum stays zero every step."""
    m, d = load(size)
    rng = np.random.default_rng(0)
    worst = np.zeros(2)
    for i in range(20_000):
        if i % 1000 == 0:
            d.ctrl[:] = rng.uniform(-0.5, 0.5, 7)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        r = np.abs(momentum_residual(m, d))
        worst = np.maximum(worst, [r[:3].max(), r[3:].max()])
    print(f"{size}: max linear {worst[0]:.2e} kg m/s, angular {worst[1]:.2e} kg m^2/s")
    assert worst.max() < 1e-6


@pytest.mark.parametrize("size", SIZES)
def test_jstar_predicts_hand_velocity(size):
    m, d = load(size)
    d.ctrl[:] = [0.3, -0.2, 0.4, 0.3, -0.5, 0.2, 0.6]
    for _ in range(1000):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    k = kin(m, d)
    actual = jacobian(m, d) @ d.qvel
    assert np.allclose(k.Jstar @ d.qvel[6:], actual, atol=1e-6)
    assert np.allclose(k.W @ d.qvel[6:], d.qvel[3:6], atol=1e-6)


@pytest.mark.parametrize("size", SIZES)
def test_dof_argument(size):
    m, d = load(size)
    k = kin(m, d)
    rank = np.linalg.matrix_rank
    assert rank(k.W) == 3
    assert round(np.trace(rns_projector(k.W))) == 4                 # 7 - 3
    assert rank(np.vstack([k.Jstar[:3], k.W])) == 6                 # 3-D task: 1 spare DOF
    S = np.vstack([k.Jstar, k.W])                                   # 6-D task: 9 rows, 7 joints
    assert rank(S) == 7
    b = np.r_[np.random.default_rng(1).normal(size=6), 0, 0, 0]
    assert np.linalg.norm(S @ np.linalg.pinv(S) @ b - b) > 1e-3     # no exact solution


@pytest.mark.parametrize("scheme", [4, 5])
def test_3d_reactionless(scheme):
    """R4 on the medium servicer: base returns to its start attitude, peak drift well below DLS."""
    r, ref = run_trial("medium", scheme, "3d", loops=1), run_trial("medium", 1, "3d", loops=1)
    print(f"scheme {scheme}: {r['max_base_deg']:.3f} deg vs DLS {ref['max_base_deg']:.3f} deg")
    assert r["stable"] and r["max_base_deg"] < ref["max_base_deg"] / 3
    # scheme 4 has no posture pull: joint 5 wanders 3 rad, parks on its limit and leaves 0.066 deg (0.000 with limits off, D32)
    assert r["final_base_deg"] < (0.02 if scheme == 5 else 0.1)


def test_joint_geofence():
    """Small servicer, pure RNS: without limits the joints wander far outside them; with limits they stay inside."""
    free, fenced = run_trial("small", 4, "3d", lim=None), run_trial("small", 4, "3d")   # 2 loops: 1 never reaches a limit
    print(f"limits engaged {fenced['limit_frac']:.0%} of the run, worst overshoot {fenced['lim_viol_rad']:.4f} rad, "
          f"base {free['max_base_deg']:.2f} -> {fenced['max_base_deg']:.2f} deg, "
          f"hand {free['rms_pos_mm']:.1f} -> {fenced['rms_pos_mm']:.1f} mm")
    assert fenced["stable"] and fenced["limit_frac"] > 0 and fenced["lim_viol_rad"] < 0.01


def test_teleop_reach():
    """Set-and-go: a near goal is accepted, one past the arm's reach is refused, and going there keeps the base level (D14)."""
    import mujoco
    from task import DECIM
    from teleop import setup, reachable, ctrl
    m, d, sid, Rb0 = setup("medium")
    p = d.site_xpos[sid].copy()
    assert not reachable(m, d, sid, Rb0, p + [0, 1.0, 0], 5, "3d")
    goal = p + [0, 0.1, 0]
    assert reachable(m, d, sid, Rb0, goal, 5, "3d")
    peak = 0
    for i in range(6000):
        if i % DECIM == 0:
            hand, base = ctrl(m, d, sid, Rb0, goal, 5, "3d")
            peak = max(peak, base)
        mujoco.mj_step(m, d)
    assert abs(hand - goal).max() < 0.005 and peak < 0.5
    from teleop import ray
    m, d, sid, Rb0 = setup("medium")
    edge = ray(m, d, sid, Rb0, np.array([0, 0, 1.0]), 5, "3d")     # the drawn envelope must match what Enter accepts
    assert edge[2] - p[2] > 0.1 and reachable(m, d, sid, Rb0, p + 0.9 * (edge - p), 5, "3d")


def test_teleop_camera_relative():
    """Stick right = right on screen, whichever way the camera faces; then drive a goal with pad inputs and send it."""
    import mujoco
    from task import DECIM
    from teleop import move, setup, reachable, ctrl, G_MAX
    o = np.zeros(3)
    assert np.allclose(move(o, 1, 0, 0, 90, 0, 1), [1, 0, 0]) and np.allclose(move(o, 0, 1, 0, 90, 0, 1), [0, 0, 1])
    assert np.allclose(move(o, 0, 0, 1, 90, 0, 1), [0, 1, 0]) and np.allclose(move(o, 1, 0, 0, 0, 0, 1), [0, -1, 0])
    assert np.allclose(move(o, 0, 0, 1, 0, -90, 1), [0, 0, -1], atol=1e-9)     # looking straight down: away = down
    m, d, sid, Rb0 = setup("medium")
    goal = d.site_xpos[sid].copy()
    for _ in range(100):                       # 1 s of stick up-and-right, camera turned to az 30, el -20
        goal = move(goal, 0.7, 0.7, 0, 30, -20, G_MAX * 0.01)
    assert 0.14 < np.linalg.norm(goal - d.site_xpos[sid]) < 0.16 and reachable(m, d, sid, Rb0, goal, 5, "3d")
