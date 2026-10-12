"""Headless run of the maps study's drive loop with a scripted operator: two goals per block, the first from rest and
the second from where the first left the arm.
gate: a near goal is sent and reached, a far one is refused by the rehearsal and never moves the arm.
guaranteed: the dot cannot leave the small cage, so a goal outside it is given up without a send.
live: after the first move the cage is probed again from where the hand stopped and swapped in.
lookup: the same swap with the cage looked up from the stored states, and the lookup itself against learn.py safe.
band: two looked-up edges and the rehearsal on every go. A trap is refused, a goal past the outer edge is never sent.
Run from the repo root."""
import sys; sys.path.insert(0, "src")
import types, numpy as np, teleop as T


def drive(aid, G, steps=30000):
    n, at, s = [0], [None, 0, False], {"v": types.SimpleNamespace(cam=types.SimpleNamespace(azimuth=90.0, elevation=-45.0), close=lambda: None)}
    seen = []

    def hook(m, d, pts):
        n[0] += 1
        if not seen or s["rays"] is not seen[-1]:
            seen.append(s["rays"])                      # every cage the operator was shown
        if s.get("far") is not None:
            s["refusal"] = (list(s["rgba"]), s["far"])  # band: the dot and the far ball as shown on a refused go
        return n[0] < steps

    def axes():                                         # sx, sy, dz, cx, cy, go, snap, fine
        if "tgt" not in s or "link" not in s:
            return 0, 0, 0, 0, 0, False, False, False
        if s["tgt"] is not at[0]:
            at[:] = [s["tgt"], n[0], False]
        e = s["tgt"] - s["link"][1]
        az, el = np.radians(90.0), np.radians(-45.0)    # the inverse of T.move at the stub camera
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        right = np.array([np.sin(az), -np.cos(az), 0])
        c = np.clip(np.array([e @ right, e @ np.cross(right, fwd), e @ fwd]) / 0.0015, -1, 1)
        go = np.linalg.norm(e) < 1e-3 and not at[2]
        at[2] = at[2] or go
        return c[0], c[1], c[2], 0, 0, go, False, False

    # give a goal up 30 s in; live: as soon as the new cage is in
    axes.park = lambda: len(seen) > 1 if aid == "live" else n[0] - at[1] > 3000
    T.viewer_hook = lambda: (s, hook)
    rows = T.free_drive(axes, "medium", 5, "3d", targets=G - home, aid=aid)
    for r in rows:
        print(aid, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    s["seen"] = seen
    return rows, s, len(seen) > 1


def test_maps():
    global home
    _, goals, dep, _, truth = T.pool("medium")
    G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
    m, d, sid, _ = T.setup("medium")
    home = d.site_xpos[sid].copy()
    near, far = np.flatnonzero((D < 0.6) & TR)[0], np.flatnonzero(D > 1.3)[0]

    (a, b), s, _ = drive("gate", G[[near, far]])
    assert a["reached"] and a["true_go"] and a["sends"] == 1 and a["refused"] == 0 and a["trips"] == 0
    assert b["parked"] and not b["true_go"] and b["refused"] == 1 and b["trips"] == 0 and b["path_m"] < 0.01
    assert "cage" not in s

    rays = T.pool("medium")[0]
    small = [r[-1] for r in T.guaranteed(rays, home, "medium")]
    kg = np.array([T.depth(small, home, g) for g in G])
    assert max(T.depth([r[-1] for r in rays], home, t) for t in small) <= 1 + 1e-9, "the small cage lies inside the rest cage"
    out = np.flatnonzero((kg > 1.2) & TR)[0]            # reachable from rest, outside the small cage
    (a, b), s, _ = drive("guaranteed", G[[np.flatnonzero(kg < 0.8)[0], out]])
    assert a["reached"] and a["trips"] == 0
    assert b["parked"] and b["sends"] == 0, "the dot stops at the small cage, so nothing is sent"

    (a, b), s, _ = drive("rest", G[[near, far]])
    assert a["reached"] and b["sends"] == 0 and len(s["rays"][0]) == 3, "rest: the full cage, dot stopped at its edge"

    (a, b), s, swapped = drive("live", G[[near, far]], 300000)
    assert a["reached"] and swapped, "the cage is probed again once the hand has stopped away from rest"


def test_dry_run_is_kept_out_of_the_participants():
    """--dry: the file a session would write is under data/pilot. No keys here, so it stops at the first prompt."""
    import subprocess
    out = [subprocess.run([sys.executable, "src/teleop.py", "--maps", "99", *dry], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True).stdout for dry in ([], ["--dry"])]
    assert "participants" in out[0] and "pilot" not in out[0]
    assert "pilot" in out[1] and "participants" not in out[1]


if __name__ == "__main__":
    test_maps()
    test_dry_run_is_kept_out_of_the_participants()
    print("ok")


def test_live_line_starts_at_the_hand():
    """A line probed by a pool worker is the line probed here, from a hand that is not at rest."""
    import mujoco, livemap
    m, d, sid, Rb0 = T.setup("medium"); g = d.site_xpos[sid] + [0.15, 0.1, 0.1]
    for i in range(int(8 / m.opt.timestep)):
        if i % T.DECIM == 0:
            T.ctrl(m, d, sid, Rb0, g, 5, "3d")
        mujoco.mj_step(m, d)
    livemap.init(); u = T.DIRS[0]
    assert np.allclose(livemap.probe((d.qpos.copy(), d.qvel.copy(), u)), T.ray(m, d, sid, Rb0, u, 5, "3d"), atol=1e-6)


def test_lookup(monkeypatch):
    """The looked-up cage at rest and one move out: 98 tips, the ones learn.py safe gives, none past the stored reach.
    No reach with the base over budget. In the tool it is swapped in when the hand stops, with no pool."""
    global home
    import livemap
    sys.path.insert(0, "analysis/envelope"); import learn
    a, b = (np.load(T.OUT.parent / f"learn{n}_medium.npz") for n in ("", 2))
    X, Y = np.r_[a["X"], b["X"]], np.r_[a["Y"], b["Y"]]
    g = learn.safe(X, Y, 10, 0.9)
    for legs in ((), ([0.15, 0.1, 0.1],)):
        m, d, sid, Rb0 = learn.state(legs); p, x = d.site_xpos[sid].copy(), learn.feat(d, Rb0)
        tips = np.array(livemap.lookup(m, d, sid, Rb0)); r = np.linalg.norm(tips - p, axis=1)
        assert tips.shape == (98, 3) and np.allclose(tips, p + g(x)[:, None] * np.array(T.DIRS), atol=1e-12)
        near = Y[np.argmin(np.linalg.norm((X - x) / X.std(0), axis=1))]     # the reaches stored for the nearest state
        assert r.min() > 0 and (r <= 0.9 * near + 1e-5).all(), "no tip further than 0.9 of what the nearest stored state reached"
        print("lookup", len(legs), "move(s): reach %.2f to %.2f m, mean %.2f" % (r.min(), r.max(), r.mean()))
    for dv in (-0.01, 0.01):
        d.qpos[3:7] = (Rb0 * T.Rot.from_rotvec([np.radians(T.BASE_MAX + dv), 0, 0])).as_quat()[[3, 0, 1, 2]]
        assert (livemap.lookup(m, d, sid, Rb0) == d.site_xpos[sid]).all() == (dv > 0), "base over budget: no reach at all, under it: some"

    _, goals, dep, _, truth = T.pool("medium")
    G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
    m, d, sid, _ = T.setup("medium"); home = d.site_xpos[sid].copy()
    monkeypatch.setattr(T.multiprocessing, "Pool", None)            # a pool started for this map would fail here
    (a, b), s, swapped = drive("lookup", G[[np.flatnonzero((D < 0.6) & TR)[0], np.flatnonzero(D > 1.3)[0]]])
    assert a["reached"] and swapped, "the cage is looked up once the hand has stopped away from rest"
    assert b["parked"] and b["sends"] == 0, "the dot stops at the looked-up cage, so nothing is sent"
    assert "hand" not in s, "nothing to wait for: no hand ball"


def test_band(monkeypatch):
    """The band (D70): two edges from the same ten stored states. The inner one is lookup's, the outer one is never
    inside it, and with the base over budget there is no reach on either. In the tool every go is rehearsed:
    a trap (inside the rest cage, no longer reachable after the lead move) is sent from the band, refused, shown red
    with the far ball, and does not move the arm. A goal past the outer edge is held there and never sent.
    The cage probed from the same place moves the inner edge to it (livemap.sharpen), the outer edge only outwards.
    The pool and the probe are stand-ins here. With a probe that is wrong by metres the trap turns green, is sent,
    and is still refused: a wrong inner edge costs a refusal, not a tilt."""
    global home
    import livemap
    sys.path.insert(0, "analysis/envelope"); import learn
    for legs in ((), ([0.15, 0.1, 0.1],)):
        m, d, sid, Rb0 = learn.state(legs); p = d.site_xpos[sid].copy()
        B = np.array(livemap.band(m, d, sid, Rb0)); lo, hi = (np.linalg.norm(B[:, i] - p, axis=1) for i in (0, 2))
        assert B.shape == (98, 3, 3) and np.allclose(B[:, 0], livemap.lookup(m, d, sid, Rb0), atol=1e-12) and (B[:, 1] == B[:, 2]).all()
        assert (hi >= lo).all() and (hi > lo + 0.01).any(), "the outer edge is never inside the inner one"
        print("band", len(legs), "move(s): inner edge %.2f m, outer edge %.2f m (means)" % (lo.mean(), hi.mean()))
    d.qpos[3:7] = (Rb0 * T.Rot.from_rotvec([np.radians(T.BASE_MAX + 0.01), 0, 0])).as_quat()[[3, 0, 1, 2]]
    assert (np.array(livemap.band(m, d, sid, Rb0)) == d.site_xpos[sid]).all(), "base over budget: no reach on either edge"

    _, goals, dep, _, truth = T.pool("medium")
    G, D, TR = goals.reshape(-1, 3), dep.ravel(), truth.ravel()
    m, d, sid, _ = T.setup("medium"); home = d.site_xpos[sid].copy()
    S = livemap.sharpen([np.array([[0, 0, 1.0], [0, 0, 2.0], [0, 0, 2.0]])] * 3, np.array([[0, 0, 1.5], [0, 0, 2.5], [0, 0, 0.5]]), np.zeros(3))
    assert np.array_equal(S, [[[0, 0, 1.5], [0, 0, 2], [0, 0, 2]], [[0, 0, 2.5]] * 3, [[0, 0, 0.5], [0, 0, 2], [0, 0, 2]]]), "inner edge to the probed tip, out or in. Outer edge only out"

    import multiprocessing.pool
    from unittest.mock import create_autospec
    ready, pools = [False], []                                      # the probe is not in yet: the band as looked up

    def ask(pl, d):                                                 # a probe that says 3 m on every line
        job = create_autospec(multiprocessing.pool.AsyncResult, instance=True)
        job.ready.return_value, job.get.return_value = ready[0], [d.site_xpos[sid] + 3.0 * np.array(u) for u in T.DIRS]
        return job
    monkeypatch.setattr(T.multiprocessing, "Pool", lambda *a: pools.append(create_autospec(multiprocessing.pool.Pool, instance=True)) or pools[-1])
    monkeypatch.setattr(livemap, "ask", ask)
    import pandas as pd
    t = pd.read_csv(T.OUT.parent / "pairs_medium.csv").query("cls == 'trap'").iloc[0]   # a study trap: inside the rest cage, out of reach after its lead move
    trap = home + t[["lx", "ly", "lz", "tx", "ty", "tz"]].to_numpy(float).reshape(2, 3)
    (a, b), s, swapped = drive("band", trap)
    assert a["reached"] and a["zone"] == 0 and swapped, "from rest both edges are the measured cage: a green go, then the band is looked up"
    assert not b["true_go"] and b["sends"] == b["refused"] == 1 and b["zone"] == 1, "the trap is inside the outer edge, so it is sent, from the band, and refused"
    assert b["trips"] == 0 and b["path_m"] < 0.01 and not b["reached"], "a refused go does not move the arm"
    rgba, far = s["refusal"]
    assert rgba == T.RED and np.linalg.norm(far - trap[1]) > 0.04, "red dot, and the far ball short of the trap"
    assert a["early"] == 0 and b["early"] == 1 and np.isnan(b["map_s"]), "the trap was sent before the probe was in"
    assert len(pools) == 1 and pools[0].terminate.call_count == 1, "one pool, stopped at the end"

    ready[0] = True                                                 # the same pair, and now the probe comes in before the send
    (a, b), s, _ = drive("band", trap)
    assert any(np.allclose(np.linalg.norm(np.array(c)[:, i] - trap[0], axis=1), 3.0, atol=0.02) for c in s["seen"] for i in (0, 2)), "the probed cage was shown, both edges on it where it is the further one"
    assert b["zone"] == 0 and b["early"] == 0, "the trap is green now and sent after the probe"
    assert b["sends"] == b["refused"] == 1 and b["trips"] == 0 and b["path_m"] < 0.01, "and still refused, the arm does not move"

    ready[0] = False
    (a, b), s, _ = drive("band", G[[np.flatnonzero((D < 0.6) & TR)[0], np.flatnonzero(D > 1.3)[0]]])
    assert a["reached"] and b["parked"] and b["sends"] == 0 and "refusal" not in s, "the dot stops at the outer edge, so nothing is sent"
