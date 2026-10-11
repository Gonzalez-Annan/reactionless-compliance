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
| lookup | the cage looked up from stored states for wherever the hand stopped, there at once. Dot stops at it. | an estimate, pulled in to 0.9: smaller than the probed cage, and it can still be wrong |

`rest` is the control. It is what the CA3 tool already does.

## Task

Put the hand on the pink ball, or give the goal up as out of reach (Right Ctrl, +30 s).
Goals come in pairs. Leg 1 is an easy goal that moves the arm off rest, it is not scored. Leg 2 is the real one and
starts from where leg 1 left the arm. 15 goals a block, so 7 scored goals per condition per person.

## Counterbalancing

5 conditions, Williams design, 10 orders (the tool takes the order from the participant id). 20 participants = 2 per
order: every map 4 times in every place and 4 times right after every other.
The goal set that goes with a condition rotates with the participant id, so no map always gets the same goals.
There are 4 goal sets for 5 conditions: `rest` and `lookup` get the same 15 goals in every participant, in a
different order. Open, see below.

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

Any of these can fail and we report it as it comes out. If the conditions do not differ on the primary
measure, that is the result. H1 to H4 were written for the first four. `lookup` has no hypothesis yet: open, below.

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
- The looked-up cage against 199 fully simulated states (D55): 0.05 m too far on 1.4% of lines, in 43 of the 199
  states, worst 0.44 m, 0.67 of the reach kept. One fixed cage: 0.7%, 9 states, worst 0.16 m, 0.29 kept. It is only
  this good near the stored states, one or two moves from rest.

## Open, needs a decision from us

- Ethics. A class demo is one thing, a journal paper with participants needs consent forms and most likely
  approval from the university. Ask before the first participant, not after.
- Who the 20 are (classmates is fine, but say so in the paper) and whether they get a practice block.
- A fifth goal set, so `rest` and `lookup` do not share one. It means drawing the pool again (`teleop.pool`), which
  changes the sets the scripted logs were run on. Or accept the shared set and say so in the paper.
- What we expect of `lookup`, written down before the first participant.

## Not participant data

Everything in the repo so far is scripted runs. `maps_0.csv` is the self-test id and is left out of the analysis.

## Changes

- 2026-10-11, before any participant: `lookup` added as a fifth condition (D54, D55: a looked-up cage keeps 2.3
  times the reach of a fixed one and is wrong in more states, 43 of 199 against 9). The block orders went from 4 to 10 with it, so an id no longer
  gets the order it had in the scripted logs of before this date. Those logs have four blocks and still read.
- 2026-10-12, before any participant: the design was reviewed and changed (D61). From this date the Design section
  of `paper/journal/draft.tex` is the protocol, and the sections above are kept as they were written. What changed:
  1. Primary measure: test goals with a command the fence had to bring back. Seconds per goal is secondary.
  2. The participant gets one aim, lowest total time, and is told the costs (a give-up adds 30 s, a command brought
     back costs only its time). The order "no wrong go first, then speed" is gone.
  3. Goal sets are drawn again: 10 pairs a block (2 traps, 4 edge, 2 inner, 2 outer), edge goals at the depth of
     the traps, a pair kept only if its class holds with the lead move 2 cm off. The set index gets one more step
     every four participants.
  4. Hypotheses: H1 more test goals with a command brought back under the rest map than under the current-state
     map (the one confirmatory test). H2 no map is faster per test goal than the rest map. H3 reached goals take
     longer with the current-state map than with the rest map. The held map carries no hypothesis. Old H1 to H4 above
     are replaced.
  5. Practice runs under the participant's own first display, and each block starts with one edge and one trap
     practice pair.
  6. Tests: exact Wilcoxon signed-rank per participant, H1 alone at 0.05, five secondary tests under Holm. No
     Friedman test on the primary measure.
  The tool does not do 3, 5 and 6 yet. The list is in the draft.
- 2026-10-12, later, before any participant: four conditions decided (D63). `lookup` is out of the study, the tool
  can still draw it. The tool now does 3, 5 and 6 of the entry above (D64, D65). What got fixed while building it:
  1. Goal sets come from `analysis/study/pairs.py`, file `data/pairs_medium.csv`: 4 sets of 10 scored pairs.
     "Cannot be reached" means the rehearsed hand ends 4 cm or more short, "reachable" within 5 mm, "outside the map
     that held" 4 cm or more outside it. The rehearsal starts 1 s after the lead move ends.
  2. Scored traps lie at 0.68 to 0.72 and at 0.94 to 0.97 of the rest edge in every set, each with an edge goal at
     its depth.
  3. A goal still open after 90 s counts as given up: +30 s, and `timeout` is set in the log.
  4. Log columns that are new: `use` (train, block, scored), `cls`, `pair`, `timeout`, `early`, `map_s`. Only rows
     with `use` scored and `leg` 2 are analysed.
  5. The map from the current state stops short of all 8 scored traps, and of 2 of the 16 scored edge goals that the
     arm can reach (sets 0 and 1). Known before data. They stay in and count as missed goes.
  6. Keyboard only. The gamepad has never been run.
- 2026-10-12, scripted run on the final sets (D66), a rule and not people: ids 1 2 3 16 went through without an
  error. Nothing in the plan changed because of it. One trap ran into the 90 s limit under the map from the current
  state.
