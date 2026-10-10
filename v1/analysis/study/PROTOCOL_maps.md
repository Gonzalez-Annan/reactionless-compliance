# Maps study: protocol

Written before any participant was run. If we change it after seeing data, the change and the reason go at the
bottom with a date. Nothing here is a result.

## Question

The arm's reach with the base held within 2 deg is not fixed. It moves with where the arm starts (D38, D39).
A map drawn once from rest is wrong on roughly 1 goal in 14 near the edge. What should the operator be shown?

## Conditions (within subject, one block each)

| name | what the operator sees | what it costs |
| --- | --- | --- |
| rest | the full cage, drawn once from rest. Dot stops at it. | wrong sometimes: a go that the fence has to bring back |
| guaranteed | the small cage that held from every tested start. Dot stops at it. | reachable goals outside it have to be given up |
| live | the cage probed again from wherever the hand stopped. Dot stops at it. | 12 to 13 s old after every move |
| gate | no cage. Every go is rehearsed and refused if it cannot work. | the rehearsal time on every go, and no picture to plan with |

`rest` is the control. It is what the CA3 tool already does.

## Task

Put the hand on the pink ball, or give the goal up as out of reach (Right Ctrl, +30 s).
Goals come in pairs. Leg 1 is an easy goal that moves the arm off rest, it is not scored. Leg 2 is the real one and
starts from where leg 1 left the arm. 15 goals a block, so 7 scored goals per condition per person.

## Counterbalancing

4 conditions, balanced Latin square, 4 orders. 20 participants = 5 per order.
The goal set that goes with a condition rotates with the participant id, so no map always gets the same goals.

## Measures

Scored on leg 2 goals only.
- Primary: seconds per goal, with the give-up penalty and the rehearsal time in it.
- Wrong goes (the fence brought the arm back).
- Reachable goals given up, unreachable goals given up. Truth is rehearsed from the arm's real state when the goal
  appears, not taken from the rest map.
- Worst base tilt.
- After each block: workload (NASA-TLX, raw, six scales) and one trust question, 1 to 7.

## What we expect, stated now

- H1: `rest` has more wrong goes than the other three.
- H2: `guaranteed` has no wrong goes and the most reachable goals given up.
- H3: `live` is the fastest per goal, if waiting for the map costs less than the wrong goes it saves.
- H4: `gate` has no wrong goes but is slower than `live` and rated as more work.

Any of these can fail and we report it as it comes out. If the four conditions do not differ on the primary
measure, that is the result.

## Analysis

`python analysis/study/feedback.py --maps`. Friedman across conditions on seconds per goal, Wilcoxon pairs after,
Holm corrected. Written and checked on made-up rows before data (`--maps --check`).

## Risks we already know

- 7 scored goals per block is thin. The maps only disagree near the edge, so most goals will not separate them.
  The scripted pre-study (a scripted operator through all blocks) is there to see this before people are used.
- The guaranteed cage only holds for starts one move away from rest. From 39 starts two moves away, 24 had at
  least one cage tip the rehearsal refused (4.3% of pairs, D43). The study keeps to one move (leg 2 is one move
  from rest), and the name is the label of the condition, not a claim. Do not add a third leg without redoing it.
- The live map's delay depends on the laptop. Record the machine and the number of processes per session.

## Open, needs a decision from us

- Ethics. A class demo is one thing, a journal paper with participants needs consent forms and most likely
  approval from the university. Ask before the first participant, not after.
- Who the 20 are (classmates is fine, but say so in the paper) and whether they get a practice block.

## Not participant data

Everything in the repo so far is scripted runs. `maps_0.csv` is the self-test id and is left out of the analysis.
