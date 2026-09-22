# Reactionless Compliance

Integrable Redundancy Resolution and Contact-Safety Limits for
Teleoperated Free-Floating Space Manipulators.

MA6223 CA3 Group Project -- Nanyang Technological University, MSc Robotics
and Intelligent Systems, Semester 1 AY2026/27.

## Repository Structure

reactionless-compliance/
  DECISIONS.md              Running log of every design decision + date + who made it
  models/
    ff_manipulator.xml      7-DOF arm on a free-floating base (MuJoCo)
  src/
    dynamics.py             H_b, H_bm, generalized Jacobian J*, RNS projector [P2]
    schemes.py              The five redundancy resolution laws [P3]
    teleop.py                Gamepad -> task-space velocity [P1]
    task.py                   Closed-loop return-to-origin trajectory [P1]
    safety.py                 Lambda(q), effective mass, v_max map [P4]
    logger.py                 Single-clock CSV logging [P1]
  tests/
    test_momentum.py        Momentum-conservation gate -- must pass before any
                             other result is trusted; runs in CI
  analysis/
    metrics.py               Drift, cost, manipulability [P5]
    figures.py                All five paper figures, final quality [P5]
  data/
    pilot/                    Pilot-run data
    participants/             P01-P12, anonymised (git-ignored except .gitkeep)
  paper/                      Overleaf mirror, IEEEtran source

## Roles

| Role | Owns |
|------|------|
| P1 -- Sim & Integration | Repo, MuJoCo model, teleoperation loop, logging, session runner, demo video |
| P2 -- Dynamics | Generalized Jacobian, reaction null space, momentum verification |
| P3 -- Control | The five resolution schemes, gain tuning |
| P4 -- Safety | Operational-space inertia, velocity-limit map, hazard table, verification matrix |
| P5 -- Experiment & Report | Protocol, participants, statistics, all figures, Overleaf, slides |

## Reproduction
1. pip install -r requirements.txt (add this file once dependencies are pinned)
2. pytest tests/test_momentum.py -- the momentum-conservation gate. All
   downstream results are invalid until this passes.
3. See DECISIONS.md for the history of design choices and why they were made.
