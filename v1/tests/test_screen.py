"""Our screen (D78), drawn without a window: the cage kept from the last frame gives the same picture as drawn fresh,
the cards and the read screen change the picture, and the pacing skips frames, not sim time."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import teleop as T
import view


def test_screen():
    m, d, sid, _ = T.setup("medium")
    v = view.Screen(m, d, window=False)
    p = d.site_xpos[sid].copy()
    ref = (p + [.2, .1, .05],)
    s = {"hand": (p, T.GREEN), "text": "current", "label": "map", "link": (p, ref[0]), "rgba": T.WHITE,
         "cage": (T.TRIS, T.EDGES), "rays": list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"])}
    bare = np.asarray(v.frame(d, {}, None))
    first = np.asarray(v.frame(d, s, ref))
    s["link"] = (p, p + [.1, 0, 0])                     # a mark moves, the cage stays
    kept = np.asarray(v.frame(d, s, ref))
    v.kept.clear()
    fresh = np.asarray(v.frame(d, s, ref))
    assert first.shape == (view.H, view.W, 3)
    assert np.array_equal(kept, fresh), "the kept cage draws what a fresh one draws"
    assert not np.array_equal(first, kept), "the moved mark shows"
    assert (np.abs(first.astype(int) - bare).sum(2) > 30).mean() > 0.05, "cage and cards change the picture"
    s["say"] = ("Block 1 of 4", "Cross or Enter to start.")
    assert np.asarray(v.frame(d, s, ref)).mean() < fresh.mean(), "the scene dims behind a screen to read"
    v.close()


def test_pacing(monkeypatch):
    m, d, _, _ = T.setup("small")
    clock, posts = [0.0], []
    monkeypatch.setattr(view, "time", SimpleNamespace(time=lambda: clock[0],   # view's clock only: threads left by
                        sleep=lambda t: clock.__setitem__(0, clock[0] + max(t, 0))))   # other tests sleep on the real one

    class Fake:                                          # the screen thread, counting the states the sim posts
        def __init__(self, m, d): self.new = self
        def set(self): posts.append(self.post[0])
        def is_running(self): return True

    monkeypatch.setattr(view, "Live", Fake)
    s, hook = view.viewer_hook()
    for i in range(300):                                 # 3 s of sim
        d.time = i * view.DECIM * m.opt.timestep
        if i == 150:
            clock[0] += 1.0                              # a rehearsal holds one step for 1 s
        hook(m, d, None)
    assert abs(clock[0] - 4) < .05, ("real time, and the hitch slips the clock: no running fast after it", clock[0])
    assert 2 * view.FPS < len(posts) <= 4 * view.FPS + 1, len(posts)
    assert posts[0] is not d and posts[0].time < posts[-1].time, "each post is a copy, the thread never reads mid-step"


def test_depth():
    """depth solves all faces at once (D78); the old loop, face by face, gives the same number to the bit."""
    def old(tips, p0, g):
        for t in T.TRIS:
            A = np.column_stack([tips[i] - p0 for i in t])
            if abs(np.linalg.det(A)) > 1e-12:
                c = np.linalg.solve(A, g - p0)
                if (c >= -1e-9).all():
                    return c.sum()
        return np.inf
    rays, rng = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"]), np.random.default_rng(1)
    for sh in range(len(rays[0])):
        tips = [r[sh] for r in rays]
        p0 = np.mean(tips, 0) + rng.normal(0, .1, 3)
        for g in list(p0 + rng.normal(0, .4, (100, 3))) + [p0 + (t - p0) * f for t in tips for f in (1, 1.5)]:
            assert old(tips, p0, g) == T.depth(tips, p0, g)
