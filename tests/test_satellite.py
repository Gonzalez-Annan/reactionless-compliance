"""The satellite look (analysis/satellite.py) is paint only: the dressed servicers move exactly as the bare ones."""
import mujoco
import numpy as np
import pytest

from analysis.satellite import load, start


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_dressing_is_paint_only(size):
    a, b = load(size, look=False), load(size, earth_w=64)
    n = a.nbody
    assert b.nbody == n + 1 and b.body_mass[n] == 0 and b.body_jntnum[n] == 0          # the sky body
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        assert np.array_equal(getattr(a, f), getattr(b, f)[:n]), f
    assert (a.geom_contype != 0).sum() == (b.geom_contype != 0).sum()                   # no new contacts
    da, db = start(a), start(b)
    for k in range(200):
        da.ctrl[:] = db.ctrl[:] = .8 * np.sin(.05 * k * np.arange(1, 8))
        mujoco.mj_step(a, da, nstep=20)
        mujoco.mj_step(b, db, nstep=20)
    assert np.array_equal(da.qpos, db.qpos) and np.array_equal(da.qvel, db.qvel)
