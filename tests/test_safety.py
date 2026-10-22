"""Contact-velocity limit, speed limiter and impact model. Owner: P4 (first version by P3)."""
import mujoco
import numpy as np
import pytest

from src.safety import (BODY, GOVERNOR_MARGIN, SpeedGovernor, impact_test, m_design, m_eff, montecarlo,
                        op_space_inertia, reduced_mass, set_payload, summarize, v_max_design, v_max_impulse,
                        v_max_iso, vmax_map)
from tests.test_momentum import load


def test_iso_table_values():
    """Spot checks against ISO/TS 15066:2016 Tables A.2-A.4."""
    assert BODY["chest"] == dict(F_qs=140.0, k=25e3, m=40.0)
    assert BODY["hand"] == dict(F_qs=140.0, k=75e3, m=0.6)
    assert (2 * 140.0) ** 2 / (2 * 25e3) == pytest.approx(1.6, abs=0.05)    # Table A.4: chest 1.6 J
    assert "skull" not in BODY and "face" not in BODY                     # transient contact not applicable


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_lambda_is_symmetric_positive_definite(size):
    m, d = load(size)
    for locked in (True, False):
        L = op_space_inertia(m, d, locked)
        assert np.allclose(L, L.T, atol=1e-8 * np.abs(L).max())
        assert np.all(np.linalg.eigvalsh((L + L.T) / 2) > 0)


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_m_eff_bounds(size):
    m, d = load(size)
    total = m.body_subtreemass[1]
    for u in np.eye(3):
        locked, free = m_eff(m, d, u, locked=True), m_eff(m, d, u, locked=False)
        assert 0 < free <= locked + 1e-9 <= total + 1e-6


def test_rigid_bound_grows_with_the_base_but_design_mass_does_not():
    rigid = [m_eff(*load(s)) for s in ("small", "medium", "large")]
    design = [m_design(*load(s)) for s in ("small", "medium", "large")]
    assert rigid[0] < rigid[1] < rigid[2]
    assert max(design) / min(design) < 1.05


def test_formulas():
    assert reduced_mass(10.0) == 10.0 and reduced_mass(10.0, 10.0) == 5.0
    assert v_max_impulse(2.0, 1.0) == pytest.approx(0.5)
    assert v_max_impulse(2.0, 1.0, e=1.0) == pytest.approx(0.25)
    for r in BODY:                                   # pinned is always the worse case
        assert v_max_iso(8.0, r, "clamped") < v_max_iso(8.0, r, "free")
    v, case = v_max_design(8.0)
    assert v == pytest.approx(min(v_max_iso(8.0, r, c) for r in BODY for c in ("free", "clamped")))
    assert case.endswith("/clamped")


def test_payload_raises_effective_mass_and_undoes():
    m, d = load("medium")
    before = m_eff(m, d, locked=False)
    undo = set_payload(m, 5.0)
    mujoco.mj_forward(m, d)
    assert m_eff(m, d, locked=False) > before + 2.0
    undo()
    mujoco.mj_forward(m, d)
    assert m_eff(m, d, locked=False) == pytest.approx(before)


def test_pose_dependence_comes_from_joint_stiffness():
    """Fig. 5: with rigid joints v_max varies strongly over the workspace (the plan's
    expectation); with the real back-drivable servos it is almost flat."""
    m, _ = load("medium")
    _, _, rigid = vmax_map(m, n=9, locked=True, direction="tool")
    _, _, servo = vmax_map(m, n=9, direction="tool")
    assert np.nanmax(rigid) > 2 * np.nanmin(rigid)
    assert np.nanmax(servo) < 1.15 * np.nanmin(servo)
    assert np.nanmin(servo) > np.nanmax(rigid)


def test_governor_scales_fast_commands_and_keeps_slow_ones():
    m, d = load("medium")
    gov = SpeedGovernor()
    slow, s = gov(m, d, np.full(7, 0.01))
    assert s == 1.0 and np.allclose(slow, 0.01)
    fast, s = gov(m, d, np.array([0, 1.0, 0, -1.0, 0, 1.0, 0]))
    assert 0 < s < 1
    from src.dynamics import kin
    v = kin(m, d).Jstar[:3] @ fast
    assert np.linalg.norm(v) <= GOVERNOR_MARGIN * v_max_design(m_design(m, d, v / np.linalg.norm(v)))[0] + 1e-9


def test_impact_force_independent_of_base_and_design_is_conservative():
    """The +/-5 Nm servos give way: a 20 kg and a 1500 kg base hit equally hard."""
    f = {s: impact_test(s, "lower_leg", "free", v=0.05) for s in ("small", "large")}
    assert f["small"]["F_peak"] == pytest.approx(f["large"]["F_peak"], rel=0.1)
    m, d = load("large")
    sim_vmax = 2 * BODY["lower_leg"]["F_qs"] / (f["large"]["F_peak"] / f["large"]["v_contact"])
    assert GOVERNOR_MARGIN * v_max_iso(m_design(m, d), "lower_leg", "free") <= sim_vmax


def test_montecarlo_runs_and_summarizes():
    rows = montecarlo(sizes=("medium",), payloads=(0.0,), n=5)
    s = summarize(rows)
    assert len(rows) == 5 and set(s.restraint) == {"free", "clamped"}
    assert (s.v_max_min > 0).all()
