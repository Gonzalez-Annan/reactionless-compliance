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
    teleop.py               Operator input -> task-space velocity, fixed-rate loop, window [P1]
    workspace.py            Reachable workspace of the arm (shown with V in teleop) [P1]
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
| Base | free joint, gravity 0 |
| Joint limits | ±170° (j1, j3, j5, j7), ±120° (pitch joints j2, j4, j6) |
| Collisions | on: the arm cannot pass through the base or itself |
| Start pose | keyframe `home` (elbow and wrist bent, away from the straight-arm singularity) |
| Arm mount | shoulder on the base +x face (`mount` body) |
| Actuators | velocity servos: `data.ctrl` is the commanded qdot [rad/s], kv = 20, torque limit +/-5 Nm |
| Integrator | RK4, timestep 0.25 ms, solver tolerance 1e-12, soft constraints (needed for the 1e-6 momentum gate) |
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
- **Joint limits and collisions are on** (arm vs base, arm vs itself). Use
  `mj_resetDataKeyframe(model, data, model.key("home").id)` to start from the
  home pose; the all-zero pose is a straight-arm singularity.
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

- **Actuators are velocity servos:** `data.ctrl` is the commanded arm qdot
  (rad/s). Write qdot to `data.ctrl`, never to `data.qvel` -- writing `qvel`
  bypasses the dynamics and the base will not react. `ctrl` is limited to
  ±1 rad/s per joint and joint torque saturates at ±5 Nm (kv = 20).
  **Provisional** (see DECISIONS.md) -- raise it now if it does not fit
  your code.

### Controller interface (P3: plug your schemes in here)

`src/teleop.py` runs the loop and calls one function every control tick
(100 Hz):

```python
controller(model, data, xdot, mode) -> qdot   # qdot: shape (7,), rad/s
```

- `xdot`: operator's commanded end-effector twist `[vx vy vz wx wy wz]`,
  world frame (same row order as `mj_jacSite`'s `[jacp; jacr]`). In `"3d"`
  mode the last three entries are zero.
- `mode`: `"3d"` or `"6d"`.
- The loop writes `qdot` to `data.ctrl`. If any joint exceeds ±1 rad/s the
  whole vector is scaled down, keeping its direction.
- To register schemes, define in `src/schemes.py`:
  `SCHEMES = {"dls": dls, "rns": rns, ...}`. Then
  `python -m src.teleop --controller dls` runs that scheme.

### Logs (P5)

`--log path.csv` writes one row per control tick. `# key: value` lines at
the top record model, mode, controller and rates; read with
`pd.read_csv(path, comment="#")`. Columns: `t_wall`, `t_sim` (the clock),
`xdot_cmd_0..5`, `qdot_cmd_0..6`, `qpos_0..13`, `qvel_0..12`,
`ee_pos_0..2`, `ee_quat_0..3`, `target_pos_0..2`, `target_quat_0..3`
(quaternions `w x y z`), `antenna_err_deg`.

## Running the teleop window

```bash
source .venv/bin/activate
python -m src.teleop                                  # medium servicer, 6-D, arm held still
python -m src.teleop --model small --mode 3d
python -m src.teleop --log data/pilot/test.csv        # also record a CSV
```

| Input | Command |
|-------|---------|
| W / S, A / D, R / F | move target ±x, ±y, ±z |
| Q / E | roll |
| hold left mouse + drag | pitch / yaw |
| Shift | fine mode (×0.3) |
| scroll | zoom |
| G | snap target back to the hand |
| V | show / hide the reachable workspace (cyan cubes; moves with the base) |
| P / Backspace / Esc | pause / reset / quit |

`--controller demo` makes the arm chase the blue sphere with a crude
fixed-base rule, so you can see the base turn as the arm moves (try
`--model small`). It is a demo, **not** one of the five benchmark schemes.

The blue sphere is the commanded target. The green line is the antenna
direction at the start and the yellow line is now (red past ±5°). With the
default `--controller hold` the arm stays still and only the target moves. A PlayStation gamepad will be added
as a second input device.

## Tests

```bash
pytest -v
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
