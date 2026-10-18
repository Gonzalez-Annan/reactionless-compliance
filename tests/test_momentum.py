"""Momentum-conservation gate (the Week 6 gate). Owner: P2.

Asserts || H_b v_b + H_bm qdot || < 1e-6 at every physics step, for random qdot over 10 s
of simulated motion, for each of the three servicers. All downstream results are invalid
until this passes for ff_small, ff_medium and ff_large. (Filled in by P3 with the ported
src/dynamics.py; P2 to review.)
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.dynamics import kin, momentum_residual

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GATE = 1e-6


def load(size):
    m = mujoco.MjModel.from_xml_path(str(MODELS_DIR / f"ff_{size}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_momentum_conserved(size):
    m, d = load(size)
    rng = np.random.default_rng(0)
    hold = int(0.5 / m.opt.timestep)          # a new random qdot every 0.5 s
    worst = 0.0
    for k in range(int(10.0 / m.opt.timestep)):
        if k % hold == 0:
            d.ctrl[:] = rng.uniform(-1, 1, m.nu)
        mujoco.mj_step(m, d)
        worst = max(worst, np.abs(momentum_residual(m, d)).max())
    assert worst < GATE, f"ff_{size}: momentum residual {worst:.2e} >= {GATE}"


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_generalized_jacobian_predicts_hand_and_base(size):
    """J* qdot is the hand twist and W qdot the base angular rate, with the base floating."""
    m, d = load(size)
    d.ctrl[:] = np.random.default_rng(1).uniform(-1, 1, m.nu)
    for _ in range(2000):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    k = kin(m, d)
    assert np.allclose(k.J @ d.qvel, k.Jstar @ d.qvel[6:], atol=1e-7)
    assert np.allclose(d.qvel[3:6], k.W @ d.qvel[6:], atol=1e-7)


def test_6d_reactionless_is_infeasible():
    """Plan Section 1, correction 2: 6 task + 3 reactionless rows on 7 joints have no exact
    solution in general; 3 task + 3 reactionless do."""
    m, d = load("medium")
    k = kin(m, d)
    assert np.linalg.matrix_rank(np.vstack([k.Jstar, k.W])) == 7      # 9 rows, rank 7 < 9
    assert np.linalg.matrix_rank(np.vstack([k.Jstar[:3], k.W])) == 6  # 6 rows, rank 6: exact
