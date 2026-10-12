# Running a session of the maps study

For whoever sits next to the participant. The design is the Design section of `paper/journal/draft.tex`, the
changes are logged in `PROTOCOL_maps.md`. If this sheet says something else, those two are right and this sheet
gets fixed.

Windows only. Keyboard only. Simulation only.

## What a session tests

A cage drawn once at rest is wrong two ways after the arm has moved (`python analysis/envelope/static.py`):

- a wrong go: the cage shows the goal inside, the arm can no longer get there
- a wrong no: the cage shows the goal outside, the arm can get there now

The goals come in pairs. The first goal moves the arm off rest. The second one is the test:

| class | a block has | what it is | which mistake it catches |
|---|---|---|---|
| trap | 2 | inside the rest cage, out of reach after the first move | wrong go |
| edge | 4 | near the rest edge, still reachable, outside the small map that held | wrong no |
| inner | 2 | deep inside, always reachable | none, a check that people can drive |
| outer | 2 | outside every cage, never reachable | none, a check that people give up |

Four displays, one per block: the cage from rest, the small cage that held, the cage from where the arm is now
(comes late), no cage. The participant is never told which class a goal is.

## Once, before the first participant

1. Ethics approval, the info sheet and the consent form. Team. No session before all three are in hand.
2. Pick one laptop and keep it for all 20. The cage from where the arm is now took 11 to 20 s on a machine with
   10 processes. A slower laptop makes that display slower, so a change of laptop halfway is a change of
   condition. Mains power, no other programs open, sleep and updates off for the day.
3. Set up, from the repo root (`rc-team`):

```
python -m venv .venv
```

```
.venv\Scripts\python.exe -m pip install mujoco numpy scipy pandas matplotlib pytest
```

4. Check the install, from `v1`. 28 tests, then 39 PASS lines:

```
..\.venv\Scripts\python.exe -m pytest -q tests -p no:cacheprovider
```

```
..\.venv\Scripts\python.exe reproduce.py
```

5. Run the two pilots (next section). Write what they changed into `PROTOCOL_maps.md`, with the date.
6. Freeze the code. Write the commit hash on the log sheet. Nothing in `src/` or `data/pairs_medium.csv` changes
   between participant 1 and participant 20.

## The two pilots

Two people who are not in the 20. A pilot is not data. `--dry` saves to `data\pilot`, and the analysis never
reads that folder.

```
..\.venv\Scripts\python.exe src\teleop.py --maps 1 --dry
```

```
..\.venv\Scripts\python.exe src\teleop.py --maps 2 --dry
```

Run them exactly like a real session, consent talk and closing questions included, with a stopwatch from the
moment the person sits down. Write down:

- the stopwatch total. If either pilot is over 65 min, every block is cut to 8 scored pairs (2 traps, 3 edge,
  1 inner, 2 outer) before participant 1. That is already decided (D67).
- every question the person asked, word for word. Each one is a hole in the briefing.
- every place they misread the words on the screen.
- whether they could give the six workload numbers without help.
- how long the cage from where the arm is now took (the tool logs it as `map_s`), and whether the wait was
  clear to them.
- anything that hung, flickered or crashed.

Nobody has driven a session by hand yet. Expect the pilots to find things.

## Who takes part

- 20 people, ids 1 to 20 in the order they come. The id sets the order of the displays and which goals go with
  which display. Do not hand out ids any other way.
- Nobody from the team and nobody who has seen the tool or the goal list.
- To settle with the consent form, team: age limits, what is asked about the person (suggested: how many hours
  a week they play games with a keyboard or pad, whether they have driven a robot arm before, whether they
  have trouble telling red from green. The dot is green, yellow or red, so the last one is written down, not
  used to turn people away).

Order of the displays by id:

| id | block 1 | block 2 | block 3 | block 4 |
|---|---|---|---|---|
| 1, 5, 9, 13, 17 | cage that held | cage from now | cage from rest | no cage |
| 2, 6, 10, 14, 18 | cage from now | no cage | cage that held | cage from rest |
| 3, 7, 11, 15, 19 | no cage | cage from rest | cage from now | cage that held |
| 4, 8, 12, 16, 20 | cage from rest | cage that held | no cage | cage from now |

## On the table

- the laptop, on mains power, the terminal open in `v1`
- a stopwatch or a phone
- the key card and the workload card, printed (both at the end of this sheet)
- the info sheet, two consent forms, a pen
- the log sheet

## The session

Start the stopwatch when the person sits down.

**1. Consent (about 5 min).** Info sheet, questions, signature. Write the id on the log sheet. The name goes on the
consent form only, never in the repo and never next to the id in any file.

**2. Briefing.** Read it out, the same words for everyone:

> This is a simulation of a robot arm on a small spacecraft. The spacecraft floats, so when the arm moves, the
> spacecraft turns a little the other way. If it turns more than 2 degrees the arm is brought back to where it
> was.
>
> A pink ball shows where the hand has to go. You move a dot with the keys on this card. Put the dot on the
> pink ball and press Enter. The arm then goes there by itself. Then the next ball comes.
>
> Some balls can not be reached. If you think a ball is out of reach, press Right Ctrl and it is skipped.
>
> What counts is your total time. Skipping a ball adds 30 seconds. A ball still open after 90 seconds is skipped
> for you and also adds 30 seconds. A command that is brought back or refused costs only the time it took.
>
> There are four rounds. In each round the screen gives you a different kind of help. Before each round the
> screen says in two lines what the help is. Read it, ask me if the words are not clear, then press Enter.
>
> After each round I ask you for seven numbers about how the round felt.
>
> I can answer questions about the keys at any time. I can not tell you whether a ball can be reached.
>
> You can stop at any time without giving a reason.

**3. Start the tool.** `N` is the id:

```
..\.venv\Scripts\python.exe src\teleop.py --maps N
```

It prints `saving to data\participants\maps_N.csv`. If it says the file is there already, stop and read "When
something goes wrong".

**4. Training.** The terminal shows the words for this person's first display and the costs. The person reads
them, you press Enter in the terminal. Four pairs, not scored. If no inner ball was reached or the outer ball
was not given up, the tool runs the four pairs once more by itself. After that it goes on whatever happened.
Nobody is dropped for doing badly.

During training you may show the keys and say what the colours of the dot mean. After training you say nothing
about the task.

**5. Four blocks.** For each block:

1. The terminal shows `Block b/4`, the words for the display and the costs. The person reads them. You press
   Enter.
2. 12 pairs: 2 for practice, then 10 that count. The person does not see which is which. You watch and stay
   quiet.
3. The window closes. The terminal asks for six workload numbers, 0 to 100. The person reads them off the
   workload card, you type them with spaces between, Enter.
4. The terminal asks for trust, 1 to 7. Same way.
5. The tool prints the block time and the minutes so far, and saves the file. A short break here is fine. The
   next block starts when you press Enter.

If the person asks whether a ball can be reached, or whether the cage is right: "I can't say. Do what you think
is fastest."

**6. Close (about 3 min).** Stop the stopwatch. Three questions, write the answers on the log sheet in their words:

1. Which of the four rounds was the easiest, and why?
2. Was there a moment where the screen showed one thing and the arm did another? What did you do after that?
3. In the round with no cage, how did you decide when to give up?

Thank them. Ask them not to describe the task to anyone who might still take part.

**7. Right after.** Check that `data\participants\maps_N.csv` is there and the tool printed a block time four
times. Copy the file to the backup drive. Fill in the log sheet.

## When something goes wrong

- **"maps_N.csv is there already."** The id was used. Do not delete anything and do not rename the file. Check
  the log sheet for the right id.
- **The window was closed in the middle of a block, the tool crashed, or the laptop froze.** The blocks that
  were finished are in the file, it is saved after every block. There is no way to pick a session up again.
  Let the person finish if the tool is still running, thank them, write down what happened. Then move the file
  to `data\stopped\` and give the next new person the id plus 40 (41 for 1, 42 for 2). That id gets the same
  order of displays and the same goal set under each display. Only the order of the goals inside a block is
  different. A session with a block missing or cut short is not analysed.
- **The cage from where the arm is now never comes** (the marker on the hand does not turn green). Wait. The
  goal closes by itself after 90 s. Write down block and pair. If it happens on two goals in a row, the probing
  processes are gone: end the session, same rule as above.
- **The person wants to stop.** Stop. What happens to the file is what the consent form says.
- **You said something you should not have** (for example that a ball is out of reach). Write it on the log sheet
  with block and pair. The session stays in. The note goes into the paper.

Nothing is ever fixed by editing a CSV.

## What is in the file

One row per goal, two rows per pair. Only rows with `use` = scored and `leg` = 2 are analysed.

| column | what |
|---|---|
| `cue` | the display: rest, guaranteed, live, gate |
| `cls` | trap, edge, inner, outer |
| `trips` | commands the 2 degree limit brought back. Above 0 on a test goal is a wrong go, the primary measure |
| `parked`, `timeout` | the goal was given up, and whether the 90 s did it |
| `true_go` | could the goal be reached when it opened |
| `sends`, `refused` | commands sent, and how many the check refused (no cage only) |
| `reached`, `time_s` | got there, and simulated seconds with the 30 s of a give-up in it |
| `early`, `map_s` | cage from now: sends made on the old cage, and the longest wait for a cage |
| `zone` | where the dot was on the cage on screen at the last send |
| `tlx`, `tlx6`, `trust` | the ratings of the block |
| `run_s`, `clock_s` | wall clock, for the 65 min check only |

## After the last participant

```
..\.venv\Scripts\python.exe analysis\study\feedback.py --maps
```

It runs the one confirmatory test (H1: test goals with a wrong go, cage from rest against cage from now) and
the five secondary ones, and prints the rest with no test. The script was written before any data and is not
to be changed after the first participant. What it prints goes into the paper as it is, also if H1 fails.

## Not in the sessions

The band display (D70, `python src\teleop.py --show band`) is built and scored but it is not one of the four
conditions. Putting it in is a change of design and has to be decided and logged before participant 1.

## Key card

| key | does |
|---|---|
| arrows | dot left, right, up, down on the screen |
| PgUp / PgDn | dot away from you / toward you |
| Home / End | turn the view, to judge depth |
| Shift, held | slow, for the last bit |
| Delete | dot back onto the hand |
| Enter | send |
| Right Ctrl | give this ball up |

## Workload card

Six numbers from 0 to 100, in this order:

1. mental: how much thinking did it take (0 very little, 100 very much)
2. physical: how much work for the hands
3. hurry: how rushed did you feel
4. how well it went: 0 is perfect, 100 is a failure
5. effort: how hard did you have to work
6. frustration: how annoyed or stressed were you

Then one number from 1 to 7: how much did you trust what the screen showed (1 not at all, 7 fully).

## Log sheet, one line per session

id, date, start time, experimenter, laptop, commit hash, stopwatch total, the three answers from the close,
anything odd with block and pair. Kept by the team lead. No names on it.
