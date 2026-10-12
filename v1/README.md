# Reactionless Compliance, v1 (P2)

A 7-joint arm on a free-floating servicer, driven by a person. MA6223 CA3. Simulation only (MuJoCo).

## The statement

On a floating base, where the hand can safely go next depends on what the arm has already done. A workspace
drawn once is right for the first move only. After that it is wrong near its edge.

So we do two things. We measure how wrong the fixed map gets. And we test which picture of the workspace lets
a person drive the arm with the fewest wrong goes.

## The test

```
python analysis/envelope/static.py
```

About 3 minutes, 10 processes. It exits with code 1, that is the result and not a crash.

The cage drawn at rest is loaded and never redrawn. 140 goals are fixed in the world: 80 inside the cage or on
its edge, 60 just outside. Every goal is rehearsed from rest and again after one 0.3 m move of the hand. If the
cage were static, the answer would be the same both times.

- From rest the cage is right: 80 of 80 goals inside go, 54 of 60 goals outside are refused.
- **Wrong go.** 36 of 400 checks: the goal went from rest and is refused after one move. The cage still shows it
  inside. The deepest one is 19 cm inside the edge.
- **Wrong no.** 30 of 270 checks: the goal was refused from rest and goes after one move. The cage still shows it
  outside. The furthest one is 7 cm outside the edge.

The stored sweep says the same on 98 directions and 40 starts (`data/wide98_medium.npz`): after one move the
edge is more than 5% inside the rest edge on 14.1% of the lines and more than 5% outside it on 7.9%.

What this does not say: that the whole cage moves. Most answers stay the same. It is wrong near the edge, and
near the edge is where a person needs it.

## What each display does about it

Five ways to show the workspace are built. `python src/teleop.py --show NAME` opens any of them with no goals.

| name | what is on the screen | wrong go | wrong no | price |
|---|---|---|---|---|
| `rest` | the cage from rest, never redrawn | yes | yes | none |
| `guaranteed` | the small cage that held from every start one move out | no | yes, most of the edge is given up | reach |
| `live` | the cage probed from where the arm is now | no | almost none | 11 to 20 s wait, the old cage is shown meanwhile |
| `gate` | no cage, every command is rehearsed and refused if it would fail | no | no | no picture to plan with |
| `band` | two looked-up edges and the rehearsal on every go (D70) | no | 0.4% of lines by 5 cm | a third of the reach is "not known" until you send |

`band` is the new one. One edge can not close both gaps: pull it in and it hides goals the arm can reach, let
it out and it shows goals it can not. So it draws two. Inside the inner edge a go is expected to work. Past the
outer edge nothing was reachable. Between them the picture says it does not know, and the rehearsal answers.
Both edges are there at once, there is nothing to wait for.

`python analysis/envelope/learn.py safe` scores both edges against the full simulation on 199 states it was not
built from: the reach is past the outer edge by 5 cm on 0.4% of the lines, the inner edge is 5 cm too far on
1.4%, and the band is 0.29 m wide on a 0.40 m reach.

A scripted rule on the 40 scored goals of the study. This is a rule, not people:

| display | reached | wrong go | missed | refused | gave up | s per goal |
|---|---|---|---|---|---|---|
| cage from rest | 24 | 8, all on traps | 0 | 0 | 16 | 26.8 |
| cage that held | 8 | 0 | 16 | 0 | 32 | 31.1 |
| cage from now | 22 | 0 | 2 | 0 | 18 | 30.3 |
| no cage | 24 | 0 | 0 | 16 | 16 | 18.7 |
| band | 24 | 0 | 0 | 11 | 16 | 18.7 |

24 of the 40 goals can be reached. Under the band the rule reached all 24 and never sent a command that had to
be brought back. 10 goals were inside the inner edge and all 10 went. 5 were past the outer edge and were given
up without a send. The other 25 were in the band: 14 went, 11 were refused. So the picture alone answered 15 of
40 and the rehearsal answered the rest. Against no cage that is 5 refusals less and the same time.

Whether any of this holds for people is not known. That is the study.

## The study

Four displays (`rest`, `guaranteed`, `live`, `gate`), 20 people, each sees all four. The one confirmatory test:
more test goals with a wrong go under the cage from rest than under the cage from now. It may fail. With 20
people the power is 0.93 or more if people act the way the rule does, and 0.32 to 0.47 if they are more careful.

- `analysis/study/SESSION.md`: how to run a session, the briefing word for word, what to do when it breaks
- `analysis/study/PROTOCOL_maps.md`: the plan and every change to it, dated
- `python src/teleop.py --maps N`: the session for participant N, saved to `data/participants`
- `python src/teleop.py --maps N --dry`: the same session as a pilot, saved to `data/pilot`
- `python analysis/study/feedback.py --maps`: the analysis, written before any data
- `python analysis/study/power.py`: the sample size table (D61), about two minutes
- `python analysis/study/pairs.py`: draws the goal sets into `data/pairs_medium.csv` (D64)
- `python analysis/study/prestudy.py`: the scripted rule above, saved to `data/scripted`

`band` is not one of the four. Putting it in is the team's call and has to be made before participant 1.

No participant has been run. Nothing was invented.

## Run it

Needs Python 3.12 with `mujoco numpy scipy pandas matplotlib pytest`. Windows. Everything is run from this folder.

```
python -m pytest tests -q        # 28 tests, about 4 min
python reproduce.py              # re-scores the stored data against DECISIONS.md, 39 checks, about 2 min
python src/teleop.py             # drive the arm: arrows, PgUp/PgDn, Enter to send. Keys are in the file header
python src/teleop.py --show band # the band display, no goals, nothing saved
```

`python reproduce.py` runs nine commands, one process each, and prints PASS or FAIL per line. Exit code 1 if
anything fails. It writes nothing into the repo.

`python reproduce.py --full` lists the commands that rebuild the data, in order, with runtimes. Most use 10
processes. The longest are `wide.py --98` (77 min) and `learn.py gen more` (about 2.7 h). `--full --run` runs
them and deletes the old files first, so only use it if you mean it.

Not covered by `reproduce.py`: anything that needs the 10-process pool (D38 to D44, D47 to D49, D52), the sweep
table, part B of `planar.py`, the scripted counts and `static.py`. The end of its output lists each one with the
reason.

## The CA3 paper

Five ways of resolving the redundancy of the arm, compared on hand tracking, base attitude disturbance, joint
drift and permitted contact speed. `paper/main.tex`.

```
python models/make_models.py         # writes the three servicer models
python analysis/sweep.py             # ~10 min, writes data/sweep.csv
python analysis/figures.py           # writes paper/figs/fig{1,2,3,5}_*.pdf
python analysis/singularity_map.py   # ~1 min, the level-base mobility of 6000 poses (D16)
python analysis/level_base.py        # ~4 min, re-levelling, out-and-back, reach by size (D17)
python analysis/sizing.py            # ~5 min, fig6: reach and wheel momentum against base mass (D18)
```

## Layout

| Path | What |
|---|---|
| `src/dynamics.py` | mass matrix blocks, generalised Jacobian J*, base rate map W |
| `src/schemes.py` | the redundancy schemes, 3-D and 6-D |
| `src/task.py` | reference path, scripted operator, trial runner, metrics |
| `src/safety.py` | effective mass and v_max map |
| `src/teleop.py` | the tool: set a goal dot in 3-D and send it, the cage, the displays, the study sessions |
| `src/livemap.py` | the cage from where the arm is now: probed (`live`) and looked up (`lookup`, `band`) |
| `src/view.py` | drawing, and replay of scripted trials in the MuJoCo viewer |
| `analysis/envelope/` | every measurement of the cage. `static.py` is the test above, `learn.py` builds and scores the looked-up edges |
| `analysis/study/` | the study: run sheet, protocol, goal sets, power, scripted rule, analysis |
| `tests/` | the gates: momentum, J*, 3-D reactionless, joint limits, the fence, the displays, the session |
| `data/` | stored maps and sweeps, goal sets, scripted runs. `participants/` is empty |
| `DECISIONS.md` | every choice, with its numbers. D70 is the two gaps and the band |
| `paper/main.tex` | the CA3 paper (IEEE) |
| `paper/journal/draft.tex` | the journal draft, the study design is in it |

## Not done, needs people

- The study. Ethics approval, consent form, recruiting 20 people, two pilots first. Nobody has driven a
  session by hand yet.
- The gamepad path has never run (needs `pygame`). The study is keyboard only.
- Checking the ISO/TS 15066 chest values in `safety.py` against the standard.
- Checking every reference in the paper against `P2_Literature_Survey.xlsx`.
- Demo video, slides, defence.
