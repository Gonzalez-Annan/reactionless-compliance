"""Feedback study, the analysis. Written before any participant was run, so nothing here is tuned to the data.
Run from the repo root: python analysis/study/feedback.py            real participants (pid 0 is the self-test, left out)
                        python analysis/study/feedback.py --all      with pid 0
                        python analysis/study/feedback.py --check    made-up rows, only to prove the script runs
Per participant and cue: seconds per goal (the score), wrong goes (fence trips), reachable goals given up,
unreachable goals given up, worst base tilt. Then the mean over participants and a Friedman test across the
three cues on seconds per goal, with Wilcoxon pairs if there are at least 6 participants."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

CUES = ("none", "colour", "stop")
DATA = Path(__file__).parent.parent.parent / "data" / "participants"


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
        for a, c in (("none", "colour"), ("none", "stop"), ("colour", "stop")):
            print("  Wilcoxon %s vs %s: p = %.3f (uncorrected, three pairs)" % (a, c, stats.wilcoxon(w[a], w[c]).pvalue))
    return b, p


if __name__ == "__main__":
    if "--check" in sys.argv:           # made-up rows: cue "stop" is 2 s faster for everyone. Never written to disk.
        rng = np.random.default_rng(0)
        df = pd.DataFrame([dict(pid=q, cue=c, time_s=10 + q - 2 * (c == "stop") + rng.uniform(0, 0.1), trips=int(c == "none"),
                                parked=k == 0, true_go=k != 0 or c == "none", peak_base_deg=1.0 + k / 10)
                           for q in range(1, 7) for c in CUES for k in range(4)])
        b, p = report(df)
        assert len(b) == 18 and p < 0.05
        assert b[b.cue == "none"].gave_up_reachable.eq(1).all() and b[b.cue == "stop"].gave_up_unreachable.eq(1).all()
        print("ok (made-up rows, not data)")
    else:
        fs = [f for f in sorted(DATA.glob("feedback_*.csv")) if "--all" in sys.argv or f.stem != "feedback_0"]
        if not fs:
            sys.exit("no participant files in %s" % DATA)
        report(pd.concat(map(pd.read_csv, fs)))
