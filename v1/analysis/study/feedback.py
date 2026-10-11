"""Feedback study and maps study, the analysis. Written before any participant was run, so nothing here is tuned to data.
Run from the repo root: python analysis/study/feedback.py            feedback study, real participants (pid 0 is the self-test, left out)
                        python analysis/study/feedback.py --all      with pid 0
                        python analysis/study/feedback.py --check    made-up rows, only to prove the script runs
                        python analysis/study/feedback.py --maps     the maps study (files maps_*.csv). Also takes --all and --check

Feedback study. Per participant and cue: seconds per goal (the score), wrong goes (fence trips), reachable goals given
up, unreachable goals given up, worst base tilt. Then the mean over participants and a Friedman test across the cues on
seconds per goal, with Wilcoxon pairs (Holm corrected) if there are at least 6 participants.

Maps study (PROTOCOL_maps.md). Four conditions: rest = the map drawn once from rest, guaranteed = the small map that
held, live = the map from the current state, which comes late, gate = no map and every command rehearsed.
Only the test goal of a scored pair counts, and only if its lead goal was reached. Practice pairs are left out.
  H1, the one confirmatory test: test goals of a block with at least one wrong go, rest against live.
  Five more, Holm corrected: seconds per test goal, gate against rest (H2). Median seconds on the test goals that were
  reached, live against rest (H3). Missed goes on edge goals, rest against live. Friedman on workload. Friedman on trust.
The paired tests are the exact signed-rank test of power.py, two-sided. With each: the mean difference within
participants, a 95% interval from 10000 resamples of participants, and how many went each way.
Everything after that is printed with no test."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from power import wilcoxon

MAPS = "--maps" in sys.argv
CUES = ("rest", "guaranteed", "live", "gate") if MAPS else ("none", "colour", "stop")
PRE = "maps_" if MAPS else "feedback_"
DATA = Path(__file__).parent.parent.parent / "data" / "participants"
REPARK_S = 30.0         # s a give-up costs, as teleop.REPARK_S
MIN_PAIRS, MAP_MAX_S = 8, 20.0      # a participant is out with fewer scored pairs in a block, or a map that took longer


def per_block(df):
    g = df.groupby(["pid", "cue"])
    return pd.DataFrame({"s_per_goal": g.time_s.mean(), "wrong_goes": g.trips.sum(),
                         "gave_up_reachable": g.apply(lambda x: int((x.parked & x.true_go).sum()), include_groups=False),
                         "gave_up_unreachable": g.apply(lambda x: int((x.parked & ~x.true_go).sum()), include_groups=False),
                         "worst_tilt_deg": g.peak_base_deg.max()}).reset_index()


def report(df):
    b = per_block(df)
    n = b.pid.nunique()
    print(b.round(2).to_string(index=False))
    print("\nmean over %d participant(s):" % n)
    print(b.drop(columns="pid").groupby("cue").mean().reindex(CUES).round(2).to_string())
    w = b.pivot(index="pid", columns="cue", values="s_per_goal").reindex(columns=CUES).dropna()
    if len(w) < 3:
        print("\nno test: %d complete participant(s), a Friedman test needs 3" % len(w))
        return b, None
    chi, p = stats.friedmanchisquare(*[w[c] for c in CUES])
    print("\nFriedman on s per goal, n = %d: chi2 = %.2f, p = %.3f" % (len(w), chi, p))
    if len(w) >= 6:
        pr = sorted((stats.wilcoxon(w[a], w[c]).pvalue, a, c) for a, c in combinations(CUES, 2))
        h = 0.0
        for i, (q, a, c) in enumerate(pr):          # Holm: the smallest p times the number of pairs, and so on down
            h = max(h, min(1.0, q * (len(pr) - i)))
            print("  Wilcoxon %s vs %s: p = %.3f, Holm %.3f" % (a, c, q, h))
    return b, p


def paired(b, col, x, y, what):
    """Exact signed-rank test of col under x minus col under y, over the participants who have both. -> p"""
    w = b.pivot(index="pid", columns="cue", values=col)
    d = np.round((w[x] - w[y]).dropna().to_numpy(float), 9) if {x, y} <= set(w) else np.zeros(0)
    if not len(d):
        print("  %s: no participant has both %s and %s" % (what, x, y))
        return np.nan
    p = wilcoxon(d)[0] if d.any() else 1.0
    lo, hi = np.percentile(d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1), [2.5, 97.5])
    print("  %s, %s minus %s: mean %+.2f (95%% %+.2f to %+.2f), %d up, %d down, %d same, n = %d, p = %.4f"
          % (what, x, y, d.mean(), lo, hi, (d > 0).sum(), (d < 0).sum(), (d == 0).sum(), len(d), p))
    return p


def friedman(b, col):
    w = b.pivot(index="pid", columns="cue", values=col).reindex(columns=CUES).dropna()
    if len(w) < 3:
        print("  Friedman on %s: %d complete participant(s), it needs 3" % (col, len(w)))
        return np.nan
    chi, p = stats.friedmanchisquare(*w.T.to_numpy())
    print("  Friedman on %s, n = %d: chi2 = %.2f, p = %.4f, means %s" % (col, len(w), chi, p, " ".join("%s %.1f" % x for x in w.mean().items())))
    return p


def maps(df):
    if "use" not in df:
        sys.exit("these logs are from before the goal sets (no 'use' column): nothing to score")
    sc, key = df[df.use == "scored"], ["pid", "cue", "pair"]
    t = sc[sc.leg == 2].merge(sc[sc.leg == 1][key + ["reached"]].rename(columns={"reached": "lead_ok"}), on=key)
    print("%d scored pairs, %d of them left out: the lead goal was not reached" % (len(t), (~t.lead_ok).sum()))
    t = t[t.lead_ok].assign(missed=lambda x: x.parked & x.true_go, wrong=lambda x: x.trips > 0)
    g = t.groupby(["pid", "cue"])
    f = lambda fn: g.apply(fn, include_groups=False)
    b = pd.DataFrame({"block": g.block.first(), "set": g.set.first(), "pairs": g.size(), "wrong_go_goals": g.wrong.sum(),
                      "missed_edge": f(lambda x: int((x.missed & (x.cls == "edge")).sum())), "s_per_goal": g.time_s.mean(),
                      "s_no_30": f(lambda x: (x.time_s - REPARK_S * x.parked).mean()), "reached": g.reached.sum(),
                      "s_reached_med": f(lambda x: x.time_s[x.reached].median()), "tlx": g.tlx.first(), "trust": g.trust.first(),
                      "map_s_max": g.map_s.max()}).reset_index()
    out = sorted(set(b.pid[(b.pairs < MIN_PAIRS) | (b.map_s_max > MAP_MAX_S)]))
    if out:
        print("participants left out (under %d scored pairs in a block, or a map slower than %g s):" % (MIN_PAIRS, MAP_MAX_S), out)
    b, t = b[~b.pid.isin(out)], t[~t.pid.isin(out)]
    print(b.round(2).to_string(index=False))
    print("\nmean over %d participant(s):" % b.pid.nunique())
    print(b.drop(columns=["pid", "block", "set"]).groupby("cue").mean().reindex(CUES).round(2).to_string())

    print("\nH1, the confirmatory test, two-sided 0.05:")
    p1 = paired(b, "wrong_go_goals", "rest", "live", "test goals with a wrong go")
    print("secondary:")
    ps = [("H2", paired(b, "s_per_goal", "gate", "rest", "H2 seconds per test goal")),
          ("H3", paired(b, "s_reached_med", "live", "rest", "H3 median seconds on reached test goals")),
          ("missed goes", paired(b, "missed_edge", "rest", "live", "missed goes on edge goals")),
          ("workload", friedman(b, "tlx")), ("trust", friedman(b, "trust"))]
    h = 0.0
    for i, (p, n) in enumerate(sorted((p, n) for n, p in ps if p == p)):    # Holm over the five planned, also if one could not be run
        h = max(h, min(1.0, p * (len(ps) - i)))
        print("  Holm, %s: p = %.4f -> %.4f" % (n, p, h))

    print("\nno test on anything below")
    by = t.groupby("cue")
    tr, lv = t[t.cls == "trap"], t[t.cue == "live"]
    print("traps a command was run on:", ", ".join("%s %d of %d" % (c, (x.sends > x.refused).sum(), len(x)) for c, x in tr.groupby("cue")))
    print("live map: %d of %d traps sent before the map was in, %d of %d test goals. Wait for a map: mean %.1f s, longest %.1f s"
          % ((lv[lv.cls == "trap"].early > 0).sum(), (lv.cls == "trap").sum(), (lv.early > 0).sum(), len(lv), lv.map_s.mean(), lv.map_s.max()))
    print("missed goes (a goal given up that could be reached), by class:")
    print(t[t.missed].groupby(["cue", "cls"]).size().unstack(fill_value=0).reindex(CUES, fill_value=0).to_string())
    m = t[t.missed & t.cue.isin(["rest", "gate"])]
    print("  rest map and no map, given up with no command sent / after one:",
          ", ".join("%s %d / %d" % (c, (x.sends == 0).sum(), (x.sends > 0).sum()) for c, x in m.groupby("cue")) or "none")
    print(pd.DataFrame({"refused": by.refused.sum(), "commands_per_goal": by.sends.mean(), "timed_out": by.timeout.sum(),
                        "worst_tilt_deg": by.peak_base_deg.max()}).reindex(CUES).round(2).to_string())
    cols = ["wrong_go_goals", "missed_edge", "s_per_goal", "s_no_30", "tlx", "trust"]
    for k in ("block", "set"):
        print("by %s:" % ("place in the session" if k == "block" else "goal set"))
        print(b.groupby(k)[cols].mean().round(2).to_string())
    w, o = b.pivot(index="pid", columns="cue", values="wrong_go_goals"), b.pivot(index="pid", columns="cue", values="block")
    if {"rest", "live"} <= set(w):
        d, first = (w.rest - w.live).dropna(), (o.rest < o.live)
        print("H1 difference, rest minus live: %+.2f for the %d who had the rest map first, %+.2f for the %d who had it second"
              % (d[first[d.index]].mean(), first[d.index].sum(), d[~first[d.index]].mean(), (~first[d.index]).sum()))
    print("test goals whose truth in the run was not their class's: %d of %d" % ((t.true_go != t.cls.isin(["edge", "inner"])).sum(), len(t)))
    return b, p1


def made_up():
    """8 ids, 4 blocks of 10 scored pairs and 2 practice pairs. Rest: a wrong go on both traps. Guaranteed: the 4 edge
    goals given up. Gate: 2 s faster. One pair with its lead goal missed and the practice pairs carry absurd numbers."""
    rows = []
    for q in range(1, 9):
        for bl, c in enumerate(CUES, 1):
            for k, x in enumerate(["trap"] * 2 + ["edge"] * 4 + ["inner"] * 2 + ["outer"] * 2 + ["edge", "trap"]):
                go, bad = x in ("edge", "inner"), k >= 10 or (q, c, k) == (1, "rest", 9)
                up = not go or (c == "guaranteed" and x == "edge")
                trip = int(c == "rest" and x == "trap") + 9 * bad
                base = dict(pid=q, block=bl, cue=c, set=(bl + q) % 4, pair=100 * bl + k, cls=x, use="scored" if k < 10 else "block",
                            true_go=go, tlx=50.0 + q - 10 * (c == "gate"), trust=4 + (c == "live"), early=0, timeout=False,
                            map_s=12.0 if c == "live" else np.nan, peak_base_deg=1.0, refused=int(c == "gate" and not go))
                rows.append(base | dict(leg=1, reached=(q, c, k) != (1, "rest", 9), parked=False, trips=0, sends=1, time_s=999.0))
                rows.append(base | dict(leg=2, reached=not up, parked=up, trips=trip, sends=1 - up + trip + base["refused"],
                                        time_s=10 + q / 10 + REPARK_S * up + 5 * trip - 2 * (c == "gate") + 900 * bad))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "--check" in sys.argv and MAPS:  # made-up rows. Never written to disk.
        b, p = maps(made_up())
        assert len(b) == 8 * 4 and abs(p - 2 / 2 ** 8) < 1e-12 and b.s_per_goal.max() < 100, "8 of 8 one way: p = 2 / 2^8"
        assert b[(b.pid == 1) & (b.cue == "rest")].pairs.item() == 9 and b[b.cue == "guaranteed"].missed_edge.eq(4).all()
        assert b.wrong_go_goals.max() == 2, "practice pairs and the pair with a missed lead goal are not scored"
        print("ok (made-up rows, not data)")
    elif "--check" in sys.argv:         # made-up rows: the last condition is 2 s faster for everyone. Never written to disk.
        rng = np.random.default_rng(0)
        df = pd.DataFrame([dict(pid=q, cue=c, time_s=10 + q - 2 * (c == CUES[-1]) + rng.uniform(0, 0.1), trips=int(c == CUES[0]),
                                parked=k == 0, true_go=k != 0 or c == CUES[0], peak_base_deg=1.0 + k / 10)
                           for q in range(1, 7) for c in CUES for k in range(4)])
        b, p = report(df)
        assert len(b) == 6 * len(CUES) and p < 0.05 and b.s_per_goal.max() < 100
        assert b[b.cue == CUES[0]].gave_up_reachable.eq(1).all() and b[b.cue == CUES[-1]].gave_up_unreachable.eq(1).all()
        print("ok (made-up rows, not data)")
    else:
        fs = [f for f in sorted(DATA.glob(PRE + "*.csv")) if "--all" in sys.argv or f.stem != PRE + "0"]
        if not fs:
            sys.exit("no participant files in %s" % DATA)
        (maps if MAPS else report)(pd.concat(map(pd.read_csv, fs)))
