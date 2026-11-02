"""Reactionless Compliance: check the benchmark numbers against the stored data. Simulation only; no people were run.
Run from the repo root (the folder with src/ and data/), with a Python that has numpy, mujoco, scipy, pandas, matplotlib:
  python reproduce.py               re-score the stored data/*.npz, one process at a time. Prints PASS or FAIL per number.
                                    Writes nothing into the repo. Exit code 1 if any line fails.
  python reproduce.py --full        list the commands that regenerate the stored data, in order, with runtimes.
  python reproduce.py --full --run  run that list (hours, 10 processes, DELETES AND REWRITES the stored data), then re-score.
This file uses only the standard library. It re-implements no analysis: each check runs one of the repo's own scripts and
reads a number out of what that script prints. Expected values are what the script printed when I ran it on 2026-10-12,
and they match the decision record named on the line, except where the line says otherwise. A line passes if each number is within one unit of its last printed
digit (counts must match exactly).
Runtime tags: [m] I timed it, [d] the repo says so (DECISIONS.md or README), [e] my estimate, never timed."""
import os, pathlib, re, shutil, subprocess, sys, tempfile, time

PY, ROOT = sys.executable, pathlib.Path.cwd()
E = "analysis/envelope/"
D1, D2, D3, W = 0.11, 0.011, 0.0011, 1.1      # one unit of the last digit: 1 decimal, 2 decimals, 3 decimals, whole percent
POOL = "the script starts Pool(10) and has no score-only path"
# One row per step, in the order the data has to be made. run = argv after python (None: nothing new to run);
# rm = files deleted first, because the script only builds them when missing; score = the argv default mode runs;
# copy = files the score script reads, copied to a temp folder because it writes figures; why = why it is not (fully) in default mode.
ROWS = [
    dict(name="bus models", run=["models/make_models.py"], rm=[], writes="models/ff_{small,medium,large}.xml, models/earth.png", rt="seconds [e]",
         rec="D2", backs="the 30 / 300 / 3000 kg bus models", why="writes the models every other script loads; prints nothing to check"),
    dict(name="scripted benchmark", run=["analysis/sweep.py"], rm=[], writes="data/sweep.csv", rt="about 10 min, 1 process [d]",
         rec="D33", backs="84 runs; scheme 5 peaks at 0.20 deg against 1.22 for DLS (300 kg, position only); tracking 0.16 mm",
         why="I was not allowed to run it, and it rewrites data/sweep.csv; no re-score path"),
    dict(name="figures", run=["analysis/figures.py"], rm=[], writes="paper/figs/fig{1,2,3,5}_*.pdf", rt="not recorded",
         rec="D33", backs="figures 1, 2, 3, 5, drawn from sweep.csv", why="draws figures and prints no numbers"),
    dict(name="level base", run=["analysis/singularity_map.py"], rm=[], writes="(prints)", rt="about 1 min, 1 process [d]",
         rec="D16", backs="level-base mobility of 6000 poses: median 0.01, 90th percentile 0.05, best after hill-climbing 0.10",
         score=["analysis/singularity_map.py"], checks=[
             ("6000 poses, % colliding", r"6000 poses in \d+ s; collide (\d+)%", [0], 0),
             ("score percentiles 50 / 90 / 99 / max", r"score percentiles 50/90/99/max: \[([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)\]", [0.01, 0.05, 0.08, 0.09], D2),
             ("best after hill-climbing", r"^score ([\d.]+) q", [0.104], D3),
             ("rest pose score now, small / medium / large (D16 still says 0.037 / 0.043 / 0.048: the old pose)",
              r"small at the current rest pose: ([\d.]+)\nmedium at the current rest pose: ([\d.]+)\nlarge at the current rest pose: ([\d.]+)", [0.078, 0.077, 0.077], D3)],
         why="the D16 sentence about the rest pose (score 0.043) is out of date; the script's own label for it is hard-coded, line 57"),
    dict(name="drift is borrowed", run=["analysis/level_base.py"], rm=[], writes="(prints)", rt="about 4 min, 1 process [d]",
         rec="D17", backs="base turn per 10 cm of hand travel; the base comes home with the hand", why="I did not run it"),
    dict(name="sizing", run=["analysis/sizing.py"], rm=[], writes="paper/figs/fig6_sizing.{pdf,png}", rt="about 5 min, 1 process [d]",
         rec="D18, D33", backs="full reach needs about 1700 kg at 0.5 deg and 950 kg at 2 deg; K 8 to 11 kg m",
         why="I did not run it, and it writes a figure into paper/figs"),
    dict(name="ready pose", run=[E + "ready.py"], rm=[], writes=E + "ready.npz", rt="about 5 min, 10 processes [d]",
         rec="D30", backs="Q_REST is the largest 2 deg tilt-bounded volume of 160 random poses", why=POOL),
    dict(name="98 probe tips", run=["-c", "import sys; sys.path.insert(0,'src'); import teleop; teleop.pool('medium')"],
         rm=["data/decide4_medium.npz"], writes="data/decide4_medium.npz",
         rt="docstring: a few minutes; 240 rehearsals at about 8 s would be about 30 min, 1 process [d, e]",
         rec="D25", backs="the 98 probe-line tips and the 240 test goals every later number is measured against",
         why="no script re-scores it; the checks below read it"),
    dict(name="detector", run=[E + "v4.py"], rm=[E + "v4.npz"], writes=E + "v4.npz", rt="about 40 min, 1 process [e: 300 rehearsals at about 8 s]",
         rec="D33", backs="162 of 300 goals reachable; 98 lines: 8 false go, 21 false no-go, 271 right",
         score=[E + "v4.py"], checks=[
             ("goals, reachable", r"goals (\d+), reachable (\d+)", [300, 162], 0),
             ("98 lines at factor 1.00: false go, false no-go, right",
              r"98 lines\n(?:  .*\n)*?  factor 1\.00: false go +(\d+) of \d+ unreachable, false no-go +(\d+) of \d+ reachable, right (\d+) of \d+",
              [8, 21, 271], 0)]),
    dict(name="wrong mass", run=[E + "v6.py"], rm=[E + "v6_0.9.npz", E + "v6_1.1.npz"], writes=E + "v6_*.npz", rt="about 40 min, 1 process [e]",
         rec="D34", backs="bus mass 10% off: nominal map false go 4 / 1 / 1, false no-go 4 / 6 / 10",
         score=[E + "v6.py"], checks=[
             ("lines whose reach changes over 30% (of 98)", r"changes over 30% across masses: (\d+) of 98", [1], 0),
             ("mass x0.9: reachable, false go, false no-go", r"reality x0\.9 \((\d+) of 120 reachable\): nominal map false go (\d+), false no-go (\d+)", [64, 4, 4], 0),
             ("mass x1: reachable, false go, false no-go", r"reality x1 \((\d+) of 120 reachable\): nominal map false go (\d+), false no-go (\d+)", [69, 1, 6], 0),
             ("mass x1.1: reachable, false go, false no-go", r"reality x1\.1 \((\d+) of 120 reachable\): nominal map false go (\d+), false no-go (\d+)", [73, 1, 10], 0),
             ("rehearsal on the nominal model wrong: x0.9, x1.1", r"x0\.9 (\d+), x1\.1 (\d+) of 120 goals", [5, 4], 0)]),
    dict(name="bus mass band", run=[E + "v8.py"], rm=[E + "v8_*.npz"], writes=E + "v8_*.npz", rt="about 20 min, 1 process [e]",
         rec="D34", backs="lines under 0.25 m, and where the tilt limit binds, by bus mass and tilt limit",
         score=[E + "v8.py"], checks=[
             ("%d kg: under 0.25 m, tilt binds, at 0.5 / 1 / 2 deg" % kg,
              r"^ *%d kg 0\.5 deg: median [\d.]+ m, under 0\.25 m +(\d+), tilt binds +(\d+) \| 1 deg: median [\d.]+ m, under 0\.25 m +(\d+), tilt binds +(\d+)"
              r" \| 2 deg: median [\d.]+ m, under 0\.25 m +(\d+), tilt binds +(\d+)$" % kg, want, 0)
             for kg, want in ((30, [90, 96, 84, 95, 53, 88]), (150, [62, 97, 44, 96, 24, 81]), (300, [45, 97, 25, 93, 4, 63]),
                              (600, [26, 96, 2, 72, 2, 33]), (990, [3, 80, 2, 48, 2, 1]), (3000, [0, 15, 0, 0, 0, 0]))]),
    dict(name="edge moves, 20 lines", run=[E + "guar.py"], rm=[], writes="data/guar_medium.npz", rt="not recorded, 10 processes [d]",
         rec="D38", backs="77 of 100 direction and detour pairs within 0.05 of the rest edge; guaranteed edge median 0.97, worst 0.49", why=POOL),
    dict(name="40 one-move starts, 20 lines", run=[E + "wide.py"], rm=[], writes="data/wide_medium.npz", rt="18 min, 10 processes [d]",
         rec="D39", backs="80% of pairs within 0.05 of the rest edge, 7% under 0.8; guaranteed median 0.74, worst 0.35, 44% of the volume",
         score=[E + "widefig.py"], copy=["data/wide_medium.npz"], checks=[
             ("pairs, % within 0.05, % under 0.8, starts, median, worst, % volume",
              r"pairs (\d+), within 0.05 of rest (\d+)%, under 0.8 (\d+)%; guaranteed at (\d+) starts: median ([\d.]+) worst ([\d.]+) volume (\d+)%",
              [800, 80, 7, 40, 0.74, 0.35, 44], [0, W, W, 0, D2, D2, W])],
         why="only the stored file is re-scored; the 18 min regeneration is not run (" + POOL + ")"),
    dict(name="40 one-move starts, 98 lines", run=[E + "wide.py", "--98"], rm=[], writes="data/wide98_medium.npz", rt="77 min, 10 processes [d]",
         rec="D41", backs="3920 pairs; 74% within 0.05; guaranteed cage median 0.67, worst 0.24, 40% of the volume",
         score=[E + "widefig.py", "--98"], copy=["data/wide98_medium.npz"], checks=[
             ("pairs, % within 0.05, % under 0.8, starts, median, worst, % volume",
              r"pairs (\d+), within 0.05 of rest (\d+)%, under 0.8 (\d+)%; guaranteed at (\d+) starts: median ([\d.]+) worst ([\d.]+) volume (\d+)%",
              [3920, 74, 7, 40, 0.67, 0.24, 40], [0, W, W, 0, D2, D2, W])],
         why="only the stored file is re-scored; the 77 min regeneration is not run (" + POOL + ")"),
    dict(name="fill 22 empty pairs", run=[E + "wide.py", "--98", "--fill"], rm=[], writes="data/wide98_medium.npz", rt="not recorded, 10 processes [d]",
         rec="D41", backs="22 pairs whose near end was out of reach come out 0.24 to 0.29", why=POOL),
    dict(name="two-move starts", run=[E + "twoleg.py"], rm=[], writes="data/twoleg_medium.npz", rt="about 5 min, 10 processes [d]",
         rec="D43", backs="39 of 40 starts arrive; 24 have a refused tip; 4.3% of pairs; 70 of 98 lines", why=POOL),
    dict(name="two-move starts, pulled in to 0.8", run=[E + "twoleg.py", "0.8"], rm=[], writes="data/twoleg0.8_medium.npz", rt="about 5 min, 10 processes [d]",
         rec="D43", backs="11 of 39 starts, 0.9% of pairs, 23 lines", why=POOL),
    dict(name="fast map", run=[E + "fastmap.py"], rm=[], writes="data/fastmap_medium.npz", rt="about 60 s per 98 lines, 5 maps [d]",
         rec="D44", backs="median error 7 to 8 mm, 95% under 0.06 m, 0 to 2 lines promise 0.05 m too much", why=POOL),
    dict(name="goal sets of the maps study", run=["analysis/study/pairs.py"], rm=[], writes="data/pairs_medium.csv", rt="2 s on the stored rehearsals [m]",
         rec="D64", backs="2080 pairs: 79 traps, 55 hold with the lead move 2 cm off; 52 picked, scored traps at 0.68 to 0.72 and 0.94 to 0.97 of the rest edge",
         why="rewrites data/pairs_medium.csv, the goal sets of the study; rehearsing from nothing starts Pool(10)"),
    dict(name="scripted pre-study", run=["analysis/study/prestudy.py"], rm=[], writes="data/scripted/maps_{1,2,3,4}.csv", rt="30 to 40 min, 7 to 11 min an id; uses the live-map Pool [m]",
         rec="D68, D66", backs="40 test goals per display (rest / held / current / none): wrong goes 8 / 0 / 0 / 0, given up 16 / 32 / 18 / 16, missed 0 / 16 / 2 / 0. A fixed rule plays the operator, NOT people",
         why="no script re-scores data/scripted; feedback.py reads only data/participants, which is empty"),
    dict(name="looked-up map, 200 states", run=[E + "learn.py", "gen"], rm=[], writes="data/learn_medium.npz", rt="about 40 min, 10 processes [d]",
         rec="D46", backs="the 200 stored states (labels from the fast map)", why=POOL),
    dict(name="looked-up map, 800 more", run=[E + "learn.py", "gen", "more"], rm=[], writes="data/learn2_medium.npz", rt="about 2.7 h, 10 processes [d]",
         rec="D54", backs="800 more states, 1000 in all", why=POOL),
    dict(name="looked-up map, full simulation", run=[E + "learn.py", "sim"], rm=[], writes="data/learnsim_medium.npz", rt="not recorded, 10 processes [d]",
         rec="D55", backs="200 scoring states simulated in full, to score the fast labels and the maps against", why=POOL),
    dict(name="looked-up map, score", run=None, rec="D46", backs="left-out 5 nearest: median error 0.034 m, over by 0.05 on 17.9%; mean map 0.120 m, ridge 0.043 m",
         score=[E + "learn.py"], checks=[
             ("states, tilt median, tilt most, reach median, states with no reach",
              r"(\d+) states, tilt there median ([\d.]+), most ([\d.]+) deg\. reach median ([\d.]+) m, (\d+) states with no reach",
              [200, 0.31, 2.48, 0.40, 1], [0, D2, D2, D2, 0])] + [
             ("%s, %s: median, 95%%, over %%, short %%" % (blk, nm),
              r"%s:\n(?:  .*\n)*?  %s +error median ([\d.]+), 95%% ([\d.]+), worst [\d.]+ m, over by 0.05: ([\d.]+)%%, short by 0.05: ([\d.]+)%%"
              % (hd, nm), want, [D3, D3, D1, D1])
             for blk, hd, nm, want in (
                 ("all left-out states", "left-out states, all", "mean map", [0.120, 0.329, 42.6, 36.8]),
                 ("all left-out states", "left-out states, all", "ridge", [0.043, 0.205, 19.6, 24.2]),
                 ("all left-out states", "left-out states, all", "5 nearest", [0.034, 0.184, 17.9, 17.7]),
                 ("one move from rest", "left-out states, one move from rest", "5 nearest", [0.030, 0.144, 14.4, 15.9]),
                 ("two moves", "left-out states, two moves", "5 nearest", [0.037, 0.210, 20.5, 18.4]),
                 ("simulated map, rest + 4 starts", r"against the simulated map, rest \+ 4 starts", "5 nearest", [0.026, 0.096, 4.9, 18.8]))]),
    dict(name="looked-up map, careful", run=None, rec="D53, D54, D55",
         backs="against the full simulation: looked up (10 nearest x 0.9) over by 0.05 on 1.4%, in 43 of 199 states, keeps 0.67; one fixed map 0.7%, 9 states, keeps 0.29",
         score=[E + "learn.py", "safe"], checks=[
             ("stored states", r"\(5 folds, (\d+) states x 98 lines\)", [1000], 0),
             ("clean score, looked up: nearest, pull, % over by 0.05, states with a miss, worst m, reach kept",
              r"clean, looked up: picked \((\d+), ([\d.]+)\) on \d+ states\. on \d+ others: over by 0.05 on ([\d.]+)% of pairs \(95% over states: [\d.]+ to [\d.]+\), in (\d+) of \d+ states, worst ([\d.]+) m, reach kept ([\d.]+)",
              [10, 0.9, 1.5, 41, 0.44, 0.69], [0, 0, D1, 0, D2, D2]),
             ("the same against the full simulation: % over by 0.05, states with a miss, worst m, reach kept",
              r"clean, looked up:.*\n +against the full simulation, \d+ states: over at all [\d.]+%, over by 0.05 on ([\d.]+)% of pairs \(95% over states: [\d.]+ to [\d.]+\), in (\d+) of \d+ states, worst ([\d.]+) m, reach kept ([\d.]+)",
              [1.4, 43, 0.44, 0.67], [D1, 0, D2, D2]),
             ("clean score, one fixed map: percentile, % over by 0.05, states with a miss, worst m, reach kept",
              r"clean, one fixed map: picked \(([\d.]+),\) on \d+ states\. on \d+ others: over by 0.05 on ([\d.]+)% of pairs \(95% over states: [\d.]+ to [\d.]+\), in (\d+) of \d+ states, worst ([\d.]+) m, reach kept ([\d.]+)",
              [2, 0.9, 21, 0.16, 0.30], [0, D1, 0, D2, D2]),
             ("the same against the full simulation: % over by 0.05, states with a miss, worst m, reach kept",
              r"clean, one fixed map:.*\n +against the full simulation, \d+ states: over at all [\d.]+%, over by 0.05 on ([\d.]+)% of pairs \(95% over states: [\d.]+ to [\d.]+\), in (\d+) of \d+ states, worst ([\d.]+) m, reach kept ([\d.]+)",
              [0.7, 9, 0.16, 0.29], [D1, 0, D2, D2])] + [
             ("5 folds, %d nearest x %s: %% over by 0.05, reach kept" % (k, p),
              r"^ *%d +%s +[\d.]+%% +[\d.]+%% +([\d.]+)%% +([\d.]+) +\|" % (k, re.escape(p)), want, [D1, D2])
             for k, p, want in ((10, "1.0", [1.0, 0.77]), (10, "0.9", [0.6, 0.69]), (5, "0.8", [0.7, 0.68]))] + [
             ("5 folds, one fixed map at the %s percentile: %% over by 0.05, reach kept" % q,
              r"^  q +%s +[\d.]+%% +[\d.]+%% +([\d.]+)%% +([\d.]+)$" % re.escape(q), want, [D1, D2])
             for q, want in (("2.0", [0.6, 0.25]), ("5.0", [1.6, 0.38]))]),
    dict(name="controllers, one move out", run=[E + "second.py"], rm=[], writes="data/second_medium.npz", rt="about 6 min per controller, 4 controllers [d]",
         rec="D47", backs="reach from rest 0.24 to 0.75 m; the 0.8 map holds for 90% / 91% of pairs (Imp.+RNS, RNS), 100% for DLS and Impedance", why=POOL),
    dict(name="scheme 6, one move out", run=[E + "second.py", "6"], rm=[], writes="data/second_medium.npz", rt="about 6 min [d]",
         rec="D52", backs="scheme 6: the 0.8 map holds for 91% of pairs", why=POOL),
    dict(name="12-move sessions", run=[E + "session.py"], rm=[], writes="data/session_medium.npz", rt="about 7 min per controller, 4 controllers [d]",
         rec="D48", backs="refused 4 / 8 / 0 / 0 of 120; after move 12 the 0.8 map holds for 91 / 80 / 93 / 100% of pairs", why=POOL),
    dict(name="40-move sessions", run=[E + "session.py", "40"], rm=[], writes="data/session40_medium.npz", rt="about 25 min per controller [e: 7 min x 40/12]",
         rec="D49", backs="after move 40: 88 / 83 / 65 / 85%; refused of 400: 15 / 46 / 40 / 3; stuck sessions 0 / 1 / 2 / 0", why=POOL),
    dict(name="40-move sessions, scheme 6", run=[E + "session.py", "40", "6"], rm=[], writes="data/session40_medium.npz", rt="about 25 min [e]",
         rec="D52", backs="scheme 6: 23 of 400 refused; 90% of pairs after move 40", why=POOL),
    dict(name="textbook arm", run=[E + "planar.py"], rm=[], writes="(prints)", rt="part A about 3 min under load, whole script over 10 min [m]",
         rec="D45, D47", backs="no tilt limit: 0 of 131 moves clear of the band fail, 24 of 142 through it; band 0.41 to 2.36 m",
         score=[E + "planar.py", "A"], checks=[
             ("band: dynamic singularities from, to (m)", r"dynamic singularities at radius ([\d.]+) to ([\d.]+) m", [0.41, 2.36], D2),
             ("moves clear of the band: failed, of", r"clear of the band: (\d+) of (\d+) failed", [0, 131], 0),
             ("moves through the band: failed, of", r"through the band: (\d+) of (\d+) failed", [24, 142], 0)],
         why="part B (D45: 2 deg budget, 60% of pairs, 0 of 40 starts; pulled in to 0.8, 99% and 33 of 40) runs over 10 min; I stopped a full run at 626 s with nothing printed, so I never saw its numbers"),
]


def run(argv, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([PY, "-u"] + argv, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


def rescore():
    npass = nfail = 0; t0 = time.time()
    for r in ROWS:
        if not r.get("score"):
            continue
        print("== %s  %s: python %s" % (r["rec"], r["name"], " ".join(r["score"])), flush=True)
        t, tmp, argv = time.time(), None, list(r["score"])
        if r.get("copy"):                     # a script that writes figures: run it on a copy of its data, in a temp folder
            tmp = tempfile.mkdtemp(prefix="repro_")
            for f in r["copy"]:
                os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True); shutil.copy(f, os.path.join(tmp, f))
            os.makedirs(os.path.join(tmp, "paper", "figs")); argv[0] = str(ROOT / argv[0])
        code, out, err = run(argv, cwd=tmp)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        if code:
            print("FAIL  exit code %d; stderr ends: %s" % (code, " / ".join(err.strip().splitlines()[-3:]))); nfail += 1
        for label, pat, want, tol in r["checks"]:
            tol = tol if isinstance(tol, list) else [tol] * len(want)
            m = re.search(pat, out, re.M)
            got = [float(x) for x in m.groups()] if m else []
            ok = len(got) == len(want) and all(abs(g - w) <= e for g, w, e in zip(got, want, tol))
            print("%s  %-4s %s: got %s, want %s" % ("PASS" if ok else "FAIL", r["rec"].split(",")[0], label,
                                                    " ".join("%g" % g for g in got) or "no matching line", " ".join("%g" % w for w in want)))
            npass += ok; nfail += not ok
        print("   %d s" % (time.time() - t), flush=True)
    print("\n%d PASS, %d FAIL, %d s in all, one process at a time." % (npass, nfail, time.time() - t0))
    print("\nNot checked, or only partly, in default mode:")
    for r in ROWS:
        if r.get("why") and r["name"] != "98 probe tips":
            print("  %-4s %s: %s. Why: %s" % (r["rec"].split(",")[0], r["name"], r["backs"], r["why"]))
    print("\nNo participant has been run. Nothing here is human data.")
    return 1 if nfail else 0


def listing():
    print("Commands that regenerate the stored data, in this order, from the repo root. Tags: [m] timed by me, [d] the repo says so, [e] my estimate.")
    for n, r in enumerate([r for r in ROWS if r.get("run")], 1):
        print("%2d. python %s%s" % (n, " ".join(r["run"]), "   (deletes %s first)" % " ".join(r["rm"]) if r["rm"] else ""))
        print("      writes %s | %s | %s: %s" % (r["writes"], r["rt"], r["rec"], r["backs"]))
    print("Then: python -m pytest tests -q   (the dynamics checks; test_livemap.py::test_same_tips_as_cache starts Pool(10))")


def run_full():
    for r in [r for r in ROWS if r.get("run")]:
        for pat in r["rm"]:
            for f in pathlib.Path(".").glob(pat):
                print("deleting", f); f.unlink()
        print("== python %s" % " ".join(r["run"]), flush=True); t = time.time()
        code = subprocess.call([PY, "-u"] + r["run"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        print("   exit %d, %d s" % (code, time.time() - t), flush=True)
        if code:
            sys.exit("stopped: that step failed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not (ROOT / "src" / "teleop.py").exists():
        sys.exit("run this from the repo root (the folder with src/ and data/)")
    if "--run" in a and "--full" not in a:
        sys.exit("--run goes with --full")
    if "--full" in a:
        listing()
        if "--run" in a:
            run_full(); sys.exit(rescore())
        sys.exit(0)
    sys.exit(rescore())
