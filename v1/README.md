# Reactionless Compliance (MA6223 CA3)

Five ways of resolving the redundancy of a 7-joint arm on a free-floating servicer, compared on
hand tracking, base attitude disturbance, joint drift and permitted contact speed.

## Reproduce

Needs Python 3.12 with `mujoco numpy scipy pandas matplotlib pytest`.

```
python models/make_models.py     # writes the three servicer models
python -m pytest tests -q        # 14 gates: momentum, J*, DOF count, 3-D reactionless, joint limits
python analysis/sweep.py         # ~10 min, writes data/sweep.csv
python analysis/figures.py       # writes paper/figs/fig{1,2,3,5}_*.pdf
python analysis/singularity_map.py   # ~1 min, prints the level-base mobility of 6000 poses (D16)
python analysis/level_base.py        # ~4 min, re-levelling, out-and-back, reach by size (D17)
python analysis/sizing.py            # ~5 min, fig6: reach and wheel momentum against base mass (D18)
```

## Layout

| Path | What | Owner |
|---|---|---|
| `models/make_models.py` | generates `ff_small/medium/large.xml` | P1 |
| `src/dynamics.py` | mass matrix blocks, generalised Jacobian J*, base rate map W | P2 |
| `src/schemes.py` | the five schemes, 3-D and 6-D | P3 |
| `src/task.py` | reference path, scripted operator, trial runner, metrics | P1 |
| `src/safety.py` | effective mass and v_max map | P4 |
| `src/view.py` | replay scripted trials in the MuJoCo viewer | P1 |
| `src/teleop.py` | set a goal dot in 3-D inside the drawn work envelope and send it (keyboard or `--pad` PS4, controls in the file header); `--session N` runs the study | P5 |
| `tests/test_dynamics.py` | the gates (the plan calls this `test_momentum.py`) | P2 |
| `DECISIONS.md` | every choice that departs from or fills in the plan | all |
| `paper/main.tex` | IEEE draft | all |

## Not done, needs people

- Participant study and Fig. 4 (predictability and NASA-TLX). No data exists; nothing was invented.
- A pilot run of `teleop.py`: nobody has driven it by hand yet, and the gamepad path has never run (needs `pygame`).
- Checking the ISO/TS 15066 chest values in `safety.py` against the standard.
- Checking every reference in the paper against `P2_Literature_Survey.xlsx`.
- Demo video, slides, defence.
