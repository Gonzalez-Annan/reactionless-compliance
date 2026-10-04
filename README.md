# Reactionless Compliance

A Human-in-the-Loop Benchmark of Redundancy Resolution and Contact-Safety
Limits for Teleoperated Free-Floating Space Manipulators.

MA6223 CA3 Group Project -- Nanyang Technological University, MSc Robotics
and Intelligent Systems, Semester 1 AY2026/27.

## Repository Structure

```
reactionless-compliance/
  DECISIONS.md              Running log of every design decision + date + who made it
  models/
    ff_small.xml            CubeSat-class servicer (20 kg base, high arm/base ratio)
    ff_medium.xml           Mid-size servicer (150 kg base) -- used in the human study
    ff_large.xml            MRV-class servicer (1500 kg base, low arm/base ratio)
    common/
      arm_defs.xml          Shared sim options, defaults, velocity actuators
      arm_body.xml          Shared 7-DOF arm, mounted on the base +x face
  src/
    dynamics.py             H_b, H_bm, generalized Jacobian J*, RNS projector [P2]
    schemes.py              The five redundancy-resolution laws, 3-D and 6-D variants [P3]
    teleop.py               Gamepad -> task-space velocity [P1]
    task.py                 Closed-loop ORU task, --mode 3d|6d [P1]
    safety.py               Lambda(q), effective mass, v_max map [P4]
    logger.py               Single-clock CSV logging [P1]
  tests/
    test_models.py          Model smoke tests + integrator momentum check [P1]
    test_momentum.py        Momentum-conservation gate -- must pass before any
                            other result is trusted; runs in CI [P2]
  analysis/
    metrics.py              Drift, cost, manipulability [P5]
    figures.py              All five paper figures, final quality [P5]
  data/
    pilot/                  Pilot-run data
    participants/           P01-P12, anonymised (git-ignored except .gitkeep)
  paper/                    Overleaf mirror, IEEEtran source
```

## Models

All three servicers share the same 7-DOF arm (6.6 kg) and differ only in
the base, giving arm/base mass ratios of 0.33 / 0.044 / 0.0044.

| Name | Value |
|------|-------|
| Base | free joint, gravity 0, contacts off |
| Arm mount | shoulder on the base +x face (`mount` body) |
| Actuators | velocity servos: `data.ctrl` is the commanded qdot [rad/s], kv = 20, torque limit +/-5 Nm |
| Integrator | RK4, timestep 0.25 ms (needed for the 1e-6 momentum gate) |
| Sites | `end_effector` (tool point), `antenna` (z-axis = boresight for the +/-5 deg limit) |

Change the arm in `models/common/` only; the servicer files hold just the base.

## Integration notes -- read before writing code against the models

Changes made on 2026-10-04 (P1). Reasons and measurements are in DECISIONS.md.

### Everyone

- **Load models by path** (`models/ff_medium.xml` etc.). Never copy or edit a
  servicer file to change the arm, solver or actuators -- edit
  `models/common/` so all three sizes stay identical except for the base.
- **State layout** (`nq = 14`, `nv = 13`, `nu = 7`):
  - `qpos[0:3]` base position (world), `qpos[3:7]` base quaternion `w x y z`,
    `qpos[7:14]` arm joints `j1..j7`.
  - `qvel[0:3]` base linear velocity **in the world frame**, `qvel[3:6]` base
    angular velocity **in the base (local) frame**, `qvel[6:13]` arm qdot.
    MuJoCo's free-joint convention, not a choice we made -- but every matrix
    MuJoCo gives you (`M`, Jacobians) uses the same convention, so they are
    consistent with each other.
- **Timestep is 0.25 ms (RK4).** Do not change it to speed things up: at the
  old 2 ms / implicitfast the momentum error was ~1e-3 and the 1e-6 gate
  cannot pass. Controllers should run slower than physics and hold their
  command over several physics steps (P1's teleop loop will do this).
- **Named sites:** `end_effector` (tool point) and `antenna` (its z-axis is
  the boresight for the ±5° pointing limit). Base attitude drift = angle
  between the antenna z-axis now and at t = 0.
- **MuJoCo 3.14 API change:** `data.qM` no longer exists. Use
  `mujoco.mj_fullM(model, data, M)` where `M = np.zeros((model.nv, model.nv))`
  (note the argument order: `data`, then the output array).

### P2 -- Dynamics

The plan's equations map directly onto MuJoCo's matrices, verified on all
three servicers:

```python
mujoco.mj_forward(model, data)
M = np.zeros((model.nv, model.nv)); mujoco.mj_fullM(model, data, M)
H_b, H_bm = M[:6, :6], M[:6, 6:]

jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, model.site("end_effector").id)
J = np.vstack([jacp, jacr])            # world-frame [lin; ang] EE velocity
J_b, J_m = J[:, :6], J[:, 6:]
J_star = J_m - J_b @ np.linalg.solve(H_b, H_bm)
```

Measured: `||H_b v_b + H_bm qdot||` ≤ 4.4e-8 over 10 s of random qdot
(gate is 1e-6); `J_star @ qdot` matches the simulated end-effector velocity
to 6e-10. For the reaction null space, the angular rows `H_bm[3:6]` are in
the base frame -- fine for a null space (same null space in any frame), but
rotate by the base orientation if you report angular momentum in world axes.

### P3 -- Control

- The actuators are **velocity servos**: write the scheme's output qdot to
  `data.ctrl[:] = qdot` (rad/s). Do **not** write `data.qvel` directly -- that
  bypasses the dynamics and the base will not react, which defeats the study.
- `ctrl` is clipped to ±1 rad/s per joint and joint torque saturates at
  ±5 Nm (gain kv = 20). Smooth commands track within ~0.02--0.2 rad/s; step
  changes saturate the torque and lag. If a scheme commands more than ±1 rad/s
  it will be clipped silently -- scale the whole qdot vector down instead so
  the direction is preserved.
- Use `J_star` from the snippet above, not the fixed-base `J_m`.

### P4 -- Safety

The free-floating operational-space inertia comes straight from the full
13×13 `M` and the full 6×13 `J` (base columns included):
`Lambda = inv(J @ inv(M) @ J.T)`. Using the full matrices is what makes it
the free-floating value; using only the arm block gives the fixed-base value.

### Requests back to P1

If anything above does not fit your code, raise it before the end of the
week -- the actuator choice in particular is marked provisional in
DECISIONS.md.

## Testing what's here

```bash
source .venv/bin/activate
pytest -v                                        # all tests
python scripts/check_models.py                   # momentum + tracking numbers, all 3 servicers
python scripts/check_models.py --view ff_small   # watch the base react to the arm
```

## Roles

| Role | Owns |
|------|------|
| P1 -- Sim & Integration | Repo, MuJoCo models (3 servicer sizes), teleoperation loop, logging, session runner, CI, demo video |
| P2 -- Dynamics | Generalized Jacobian, reaction null space, momentum verification, literature check |
| P3 -- Control | The five schemes, 3-D and 6-D task-priority variants, gain tuning |
| P4 -- Safety | Operational-space inertia, velocity-limit map, hazard table, verification matrix |
| P5 -- Experiment & Report | Protocol, participants, statistics, all figures, Overleaf, slides. Lead author |

## Reproduction

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `pytest` -- includes the momentum-conservation gate. All downstream
   results are invalid until it passes on all three servicers.
4. See DECISIONS.md for the history of design choices and why they were made.
