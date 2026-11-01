"""The satellite look is paint: against the bare model, same masses, same framing, same geoms the code reads, same motion."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "models"))
import numpy as np, mujoco
from make_models import SIZES, xml


def test_look_is_paint():
    for name, mass in SIZES.items():
        a, b = (mujoco.MjModel.from_xml_string(xml(name, mass, look)) for look in (False, True))
        n = a.nbody                                              # the robot's bodies come first, the sky after
        for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos"):
            assert np.array_equal(getattr(a, f), getattr(b, f)[:n]), (name, f)
        assert b.nbody == n + 1 and b.body_parentid[n] == 0 and b.body_dofnum[n] == 0 and b.body_mass[n] == 0
        for f in ("center", "extent", "meansize", "meanmass", "meaninertia"):
            assert np.array_equal(getattr(a.stat, f), getattr(b.stat, f)), (name, f)
        k = (b.geom_bodyid == 1).sum() - 2                       # the paint sits on the bus, after its two geoms
        assert (b.geom_bodyid[:2 + k] == 1).all() and not b.geom_contype.any() and not b.geom_conaffinity.any()
        robot = b.geom_bodyid < n
        assert (b.geom_bodyid[robot.sum():] == n).all(), "the sky's geoms come after every geom on the robot"
        for f in ("geom_type", "geom_size", "geom_pos", "geom_bodyid"):   # geom 0 is the bus, the arm keeps its geoms
            assert np.array_equal(getattr(a, f)[:2], getattr(b, f)[:2]), (name, f)
            assert np.array_equal(getattr(a, f)[2:], getattr(b, f)[2 + k:robot.sum()]), (name, f)
        da, db = mujoco.MjData(a), mujoco.MjData(b)
        u = np.random.default_rng(0).uniform(-1, 1, (10, 7))
        for i in range(2000):
            da.ctrl[:] = db.ctrl[:] = u[i // 200]
            mujoco.mj_step(a, da); mujoco.mj_step(b, db)
        assert np.array_equal(da.qpos, db.qpos) and np.array_equal(da.qvel, db.qvel), name
