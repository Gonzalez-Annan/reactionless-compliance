"""Maps study: what a participant gets (blocks, practice, training) and how displays and goal sets go round over 20 ids.
free_drive is stubbed, nothing is simulated."""
import sys; sys.path.insert(0, "src")
import builtins, collections, numpy as np, teleop as T


def run(monkeypatch, pid, ok=True):
    """-> the rows of one participant and the targets of each free_drive call. ok: the stub reaches and gives up everything."""
    got = []
    monkeypatch.setattr(T, "free_drive", lambda axes, size, sc, mode, targets, aid: got.append(targets) or
                        [{"time_s": 1.0, "reached": ok, "parked": ok} for _ in targets])
    monkeypatch.setattr(builtins, "input", lambda q="": "50 40 30 20 10 0" if "six" in q else "5" if "trust" in q else "")
    monkeypatch.setattr(T.pd.DataFrame, "to_csv", lambda *a, **k: None)
    return T.pd.DataFrame(T.maps_study(None, pid, "medium")), got


def test_blocks(monkeypatch):
    df, got = run(monkeypatch, 1, ok=False)
    assert [len(t) for t in got] == [8, 8, 24, 24, 24, 24]      # training twice (this stub never passes it), then 4 blocks of 12 pairs
    assert len(run(monkeypatch, 1)[1]) == 5                     # training once when it is passed
    pr = T.pd.read_csv("data/pairs_medium.csv").set_index("cand")
    assert pr.index.is_unique                                   # no pair in two sets
    for b in (1, 2, 3, 4):
        q = df[(df.block == b) & (df.leg == 2)]
        assert list(q.use) == ["block"] * 2 + ["scored"] * 10 and sorted(q.cls[:2]) == ["edge", "trap"]
        assert q[q.use == "scored"].cls.value_counts().to_dict() == dict(edge=4, trap=2, inner=2, outer=2)
        assert q.set.nunique() == 1 and (q.tlx == 25).all() and (q.trust == 5).all()
    assert (df[df.block == 0].use == "train").all() and df[df.leg == 1].depth.isna().all() and df[df.leg == 2].depth.notna().all()
    b1 = df[df.block == 1]                                      # the targets are the pairs of the file, lead goal first
    assert np.allclose(got[2], np.where((b1.leg == 1).to_numpy()[:, None], pr.loc[b1.pair, ["lx", "ly", "lz"]], pr.loc[b1.pair, ["tx", "ty", "tz"]]))


def test_balance(monkeypatch):
    place, after, at, meet = (collections.Counter() for _ in range(4))
    for pid in range(1, 21):
        b = run(monkeypatch, pid)[0].query("block > 0").drop_duplicates("block")
        cues, sets = list(b.cue), list(b.set)
        assert sorted(cues) == sorted(T.MAPS) and sorted(sets) == [0, 1, 2, 3]      # every display and every set once
        place.update(enumerate(cues)); after.update(zip(cues, cues[1:])); at.update(enumerate(sets)); meet.update(zip(cues, sets))
    assert len(place) == 16 and set(place.values()) == {5}      # every display 5 times in every place
    assert len(after) == 12 and set(after.values()) == {5}      # and 5 times right after every other
    for x in (at, meet):                                        # a set in a place, a display on a set: 4 to 6 times each
        assert len(x) == 16 and 4 <= min(x.values()) and max(x.values()) <= 6
