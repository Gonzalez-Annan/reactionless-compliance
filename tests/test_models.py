"""Smoke tests for the three servicer MuJoCo models. Owner: P1."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = ["ff_small", "ff_medium", "ff_large"]


def load(name):
    return mujoco.MjModel.from_xml_path(str(MODELS_DIR / f"{name}.xml"))


@pytest.mark.parametrize("name", MODELS)
def test_model_loads_and_steps(name):
    model = load(name)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    assert model.nq == 14  # 7 free-joint DOF (pos+quat) + 7 hinge joints
    assert model.nv == 13  # 6 free-joint velocity DOF + 7 hinge joints
    assert model.nu == 7   # 7 actuated joints
    assert np.all(model.opt.gravity == 0)


@pytest.mark.parametrize("name", MODELS)
def test_actuators_are_velocity_servos(name):
    model = load(name)
    for i in range(model.nu):
        kv = model.actuator_gainprm[i, 0]
        assert kv > 0
        assert model.actuator_biasprm[i, 2] == -kv  # MuJoCo's <velocity> encoding


@pytest.mark.parametrize("name", MODELS)
def test_arm_mounted_on_base_face(name):
    model = load(name)
    base_half_x = model.geom("base_geom").size[0]
    assert model.body("mount").pos[0] == pytest.approx(base_half_x)


@pytest.mark.parametrize("name", MODELS)
def test_required_sites_exist(name):
    model = load(name)
    model.site("end_effector")
    model.site("antenna")


def test_mass_ratios_are_distinct():
    ratios = {}
    for name in MODELS:
        model = load(name)
        base_mass = model.body("base").mass[0]
        arm_mass = sum(
            model.body(i).mass[0]
            for i in range(model.nbody)
            if model.body(i).name.startswith("link")
        )
        ratios[name] = arm_mass / base_mass

    # small should have the highest arm/base ratio, large the lowest
    assert ratios["ff_small"] > ratios["ff_medium"] > ratios["ff_large"]


@pytest.mark.parametrize("name", MODELS)
def test_integrator_conserves_total_momentum(name):
    """Model-level check that the integrator settings are accurate enough
    for P2's 1e-6 gate. Drives random qdot steps for 2 s and tracks the
    total spatial momentum about the world origin, which starts at zero."""
    model = load(name)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    steps_per_cmd = int(0.5 / model.opt.timestep)
    worst = 0.0
    for k in range(int(2.0 / model.opt.timestep)):
        if k % steps_per_cmd == 0:
            data.ctrl[:] = rng.uniform(-1, 1, model.nu)
        mujoco.mj_step(model, data)
        mujoco.mj_subtreeVel(model, data)
        p = model.body_subtreemass[0] * data.subtree_linvel[0]
        L = data.subtree_angmom[0] + np.cross(data.subtree_com[0], p)
        worst = max(worst, np.abs(np.r_[p, L]).max())
    assert worst < 1e-6
