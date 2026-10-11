# Reactionless Compliance (MA6223 CA3)

Five ways of resolving the redundancy of a 7-joint arm on a free-floating servicer, compared on
hand tracking, base attitude disturbance, joint drift and permitted contact speed.

## Reproduce

Needs Python 3.12 with `mujoco numpy scipy pandas matplotlib pytest`.

```
python models/make_models.py     # writes the three servicer models
python -m pytest tests -q        # the gates: momentum, J*, DOF count, 3-D reactionless, joint limits
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

## Checking the numbers

`python reproduce.py` from this folder re-scores the data stored in `data/` and checks what the scripts print against
the numbers in DECISIONS.md. Nine commands, one process each, about 2 minutes, PASS or FAIL per line, exit code 1 if
anything fails. It writes nothing into the repo. It is simulation only: no participant has been run.

`python reproduce.py --full` lists the commands that rebuild the data, in order, with runtimes. Most use 10 processes.
The longest are `wide.py --98` (77 min) and `learn.py gen more` (about 2.7 h). `--full --run` runs them and deletes the
old files first, so only use it if you mean it.

Not covered: anything that needs the 10-process pool (D38 to D44, D47 to D49, D52), the sweep table, part B of
`planar.py` and the scripted pre-study counts. The end of the output lists each one with the reason.

`python analysis/study/power.py` prints the sample size table of the study design (D61), about two minutes.

`python analysis/study/pairs.py` draws the goal sets of the maps study into `data/pairs_medium.csv` (D64).
`python src/teleop.py --maps N` runs the study for participant N, keyboard only (D63, D65).
`python src/teleop.py --maps N --dry` is the same session as a pilot. It saves to `data/pilot`, not to the participant folder (D67).
`python analysis/study/feedback.py --maps --check` checks the analysis on made-up rows.
