"""The maps study end to end as a game, on fewer goals: the scripted operator of analysis/study/prestudy.py presses
through the briefing and score screens and drives every display. Checks the screens come in order, the simulation
stands still on them, the score shown is the sum of the scored goal times, and the saved columns. Run from the repo root."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/study")
import builtins, shutil, pandas as pd, teleop as T, prestudy

COLS = ["goal", "reached", "time_s", "sends", "refused", "path_m", "peak_base_deg", "parked", "trips", "true_go", "timeout", "early",
        "map_s", "zone", "pid", "block", "cue", "set", "pair", "cls", "use", "leg", "depth", "run_s", "tlx", "tlx6", "procs", "trust", "clock_s"]


def test_game(tmp_path, monkeypatch):
    for f in ("decide4", "wide98", "learn", "learn2"):
        shutil.copy(f"data/{f}_medium.npz", tmp_path)
    pr = pd.read_csv("data/pairs_medium.csv")
    pr = pd.concat([pr[pr.cand.isin([148, 2044])], pr[pr.use != "train"].groupby(["set", "use"]).head(1)])   # an inner and an outer to train, a pair of each kind a set
    pr.to_csv(tmp_path / "pairs_medium.csv", index=False)
    monkeypatch.setattr(T, "OUT", tmp_path / "participants")
    monkeypatch.setattr(builtins, "input", lambda q="": "")
    monkeypatch.setattr(T, "ask", lambda q, n, lo, hi: [float("nan")] * n)
    axes, fresh = prestudy.operator()
    seen = []                                       # [head, body, sim times while it was up]

    def viewer_hook():
        s, hook = fresh()
        def h(m, d, pts):
            if s.get("say"):
                if not seen or seen[-1][:2] != list(s["say"]):
                    seen.append([*s["say"], set()])
                seen[-1][2].add(d.time)
            return hook(m, d, pts)
        return s, h
    monkeypatch.setattr(T, "viewer_hook", viewer_hook)
    T.maps_study(axes, 1, "medium")

    df = pd.read_csv(tmp_path / "participants" / "maps_1.csv")
    assert list(df.columns) == COLS
    blocks = [f"Block {b} of 4" for b in (1, 2, 3, 4)]
    assert [x[0] for x in seen] == ["Training", "Training done"] * (len(seen) // 2 - 4) + [y for b in blocks for y in (b, b + " done")]
    assert all(len(x[2]) == 1 for x in seen)        # no simulated time passes on a screen
    for b, (head, body, _) in zip(range(1, 5), seen[-7::2]):
        q = df[(df.block == b) & (df.use == "scored")]
        assert head == f"Block {b} of 4 done" and f"Score {q.time_s.sum():.0f} s" in body
    t = df[df.block == 0]                           # training has no practice, every goal counts
    assert f"Score {t[t.run_s == t.run_s.iloc[-1]].time_s.sum():.0f} s" in seen[-9][1]
