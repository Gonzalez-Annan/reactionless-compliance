"""Goal order of the maps study, without the viewer: free_drive is swapped for a stub that records the targets.
The first goal of every pair must sit at LEAD inside the small cage, the second must be an untouched pool goal.
Run from the repo root."""
import sys; sys.path.insert(0, "src")
import builtins, numpy as np, teleop as T


def test_order(monkeypatch):
    got = []
    monkeypatch.setattr(T, "free_drive", lambda axes, size, sc, mode, targets, aid: got.append(targets) or [{"time_s": 0.0} for _ in targets])
    monkeypatch.setattr(builtins, "input", lambda *a: "")
    monkeypatch.setattr(T.pd.DataFrame, "to_csv", lambda *a, **k: None)
    rays, goals, *_ = T.pool("medium")
    m, d, sid, _ = T.setup("medium"); home = d.site_xpos[sid].copy()
    small = [r[-1] for r in T.guaranteed(rays, home, "medium")]
    rows = T.feedback(None, 1, "medium", T.MAPS, "maps")
    assert len(got) == 3 and [r["leg"] for r in rows[:4]] == [1, 2, 1, 2]
    for t in got:
        k = np.array([T.depth(small, home, home + x) for x in t])
        assert len(t) == 15 and k[::2].max() <= T.LEAD + 1e-9
        assert all(np.abs(goals.reshape(-1, 3) - home - x).max(1).min() < 1e-12 for x in t[1::2])
    r1 = [r["depth"] for r in rows if r["leg"] == 1]
    assert max(r1) < T.FENCE[0], "a leg 1 goal is well inside the rest cage too"
