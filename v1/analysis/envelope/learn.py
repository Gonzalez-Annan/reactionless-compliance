"""Can the live map be looked up instead of computed? Run from the repo root.
  learn.py gen   200 states (100 one move from rest, 100 two moves), from each the 98 reaches with livemap.fast
                 -> data/learn_medium.npz. About 40 min on 10 processes.
  learn.py gen more   800 more states, other seeds -> data/learn2_medium.npz, saved every 50. About 2.7 h.
  learn.py sim   the 200 states `safe` scores on, their 98 reaches again from the full simulation (T.ray)
                 -> data/learnsim_medium.npz, saved every 10. `safe` then scores against those too. (D55)
  learn.py       fit on those, test on states left out, then on the 5 states of fastmap.py that have a simulated map.
  learn.py safe  the looked-up map made careful: shortest of the k nearest states, pulled in by a factor. How often
                 does it still promise too much, and how much reach does it give away? (D53)
Input: 7 joint angles + base tilt as a rotation vector. Output: reach along each of the 98 lines, m.
Models, numpy only: mean of the training maps (what a fixed map amounts to), ridge, 5 nearest states.
ponytail: labels come from livemap.fast (8 mm median off the simulation, D44), not from the simulation itself."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "analysis/envelope")
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T, livemap
from wide import dirs

N = 200


def state(legs):
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    for s in legs:
        for k in range(int(12 / m.opt.timestep)):
            if k % T.DECIM == 0:
                T.ctrl(m, d, sid, Rb0, p0 + s, 5, "3d")
            mujoco.mj_step(m, d)
    d.qvel[:] = 0; mujoco.mj_forward(m, d)
    return m, d, sid, Rb0


def feat(d, Rb0):
    return np.r_[d.qpos[7:], (Rb0.inv() * T.Rot.from_quat(d.qpos[[4, 5, 6, 3]])).as_rotvec()]


def one(legs):
    m, d, sid, Rb0 = state(legs); p = d.site_xpos[sid].copy()
    return np.r_[feat(d, Rb0), [np.linalg.norm(livemap.fast(m, d, sid, Rb0, u) - p) for u in T.DIRS]]


def legs(more):
    """The moves that lead to each stored state, in the order of the file. more: the 800 of learn2, else the first 200."""
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1])
    n, s0 = (800, 31) if more else (N, 21)
    A, B = [dirs(s, n, tips, p0) * np.random.default_rng(s + 1).uniform(0.3, 0.8, n)[:, None] for s in (s0, s0 + 2)]
    L = [(a,) if i < n // 2 else (a, b) for i, (a, b) in enumerate(zip(A, B))]
    return [L[i] for i in np.random.default_rng(0).permutation(n)] if more else L    # the 800 mixed, so a partial file is usable


def sim(l):
    """-> features and the 98 reaches from the full simulation, not the fast estimate."""
    m, d, sid, Rb0 = state(l); p = d.site_xpos[sid].copy()
    return np.r_[feat(d, Rb0), [np.linalg.norm(T.ray(m, d, sid, Rb0, u, 5, "3d") - p) for u in T.DIRS]]


def fit(X, Y):
    """-> {name: predict(x)}"""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    Wt = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ Y)

    def knn(x, k=5):
        dist = np.linalg.norm(Z[:, :-1] - (x - mu) / sd, axis=1); i = np.argsort(dist)[:k]
        w = 1 / (dist[i] + 1e-6)
        return w @ Y[i] / w.sum()
    return {"mean map": lambda x: Y.mean(0), "ridge": lambda x: np.r_[(x - mu) / sd, 1] @ Wt, "5 nearest": knn}


def safe(X, Y, k, pull):
    """-> predict(x): per line the shortest reach among the k nearest stored states, times pull.
    No reach at all if the base is already over the tilt budget at x: that much is read off the state, not looked up."""
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    return lambda x: (np.linalg.norm(x[7:]) < np.radians(T.BASE_MAX)) * pull * Y[np.argsort(np.linalg.norm(Z - (x - mu) / sd, axis=1))[:k]].min(0)


def row(name, E):
    print("  %-10s error median %.3f, 95%% %.3f, worst %.3f m, over by 0.05: %.1f%%, short by 0.05: %.1f%%" % (
        name, np.median(abs(E)), np.percentile(abs(E), 95), abs(E).max(), 100 * (E > 0.05).mean(), 100 * (E < -0.05).mean()))


if __name__ == "__main__":
    f = T.OUT.parent / "learn_medium.npz"
    if "gen" in sys.argv:
        more = "more" in sys.argv; L = legs(more); f = f.with_name("learn2_medium.npz") if more else f
        R = []
        with Pool(10) as pl:
            for r in pl.imap(one, L, chunksize=1):
                R.append(r)
                if len(R) % 50 == 0 or len(R) == len(L):
                    np.savez(f, X=np.array(R)[:, :10], Y=np.array(R)[:, 10:]); print(len(R), flush=True)
        sys.exit()
    if "sim" in sys.argv:
        Xs = np.r_[np.load(f)["X"], np.load(f.with_name("learn2_medium.npz"))["X"]]
        I = np.random.default_rng(0).permutation(len(Xs))[len(Xs) // 5:2 * len(Xs) // 5]   # = sc in `safe`: scored on, never picked on
        L = legs(False) + legs(True); fs = f.with_name("learnsim_medium.npz")
        R = list(np.c_[np.load(fs)["X"], np.load(fs)["Y"]]) if fs.exists() else []     # a killed run picks up where it stopped
        with Pool(10) as pl:
            for r in pl.imap(sim, [L[i] for i in I[len(R):]], chunksize=1):
                R.append(r)
                if len(R) % 10 == 0 or len(R) == len(I):
                    A = np.array(R); np.savez(fs, idx=I[:len(R)], X=A[:, :10], Y=A[:, 10:])
                    print(len(R), "same states as stored: features off by at most %.1e" % abs(A[:, :10] - Xs[I[:len(R)]]).max(), flush=True)
        sys.exit()
    z = np.load(f); X, Y = z["X"], z["Y"]
    if "safe" in sys.argv and f.with_name("learn2_medium.npz").exists():
        z = np.load(f.with_name("learn2_medium.npz")); X, Y = np.r_[X, z["X"]], np.r_[Y, z["Y"]]
    if "safe" in sys.argv:
        fm = np.load(T.OUT.parent / "fastmap_medium.npz"); S = np.load(T.OUT.parent / "wide98_medium.npz")["S"]
        Xt = np.array([feat(*state(l)[1::2]) for l in [()] + [(s,) for s in S[:4]]])
        Yt = np.array([fm["true%d" % i] for i in range(5)])
        print("left-out states (5 folds, %d states x 98 lines) | simulated map (5 states x 98 lines). k = %d: every stored state, a fixed map" % (len(X), len(X)))
        print("  k pull   over at all  over by 0.02  over by 0.05  reach kept |  over at all  over by 0.05  reach kept")
        for k in (1, 5, 10, 20, len(X)):
            for pull in (1.0, 0.9, 0.8, 0.7):
                P = np.zeros_like(Y)
                for j in range(5):
                    te = np.arange(len(X))[j::5]; tr = np.setdiff1d(np.arange(len(X)), te); g = safe(X[tr], Y[tr], k, pull)
                    P[te] = [g(x) for x in X[te]]
                g = safe(X, Y, k, pull); Pt = np.array([g(x) for x in Xt]); E, Et = P - Y, Pt - Yt
                print("%4d  %.1f     %5.1f%%       %5.1f%%        %5.1f%%       %.2f   |     %5.1f%%        %5.1f%%       %.2f" % (
                    k, pull, 100 * (E > 0).mean(), 100 * (E > 0.02).mean(), 100 * (E > 0.05).mean(), P.sum() / Y.sum(),
                    100 * (Et > 0).mean(), 100 * (Et > 0.05).mean(), Pt.sum() / Yt.sum()))
        print("one fixed map, per line the q-th percentile of the stored reaches (no reach over the tilt budget, as above):")
        lvl = np.linalg.norm(X[:, 7:], axis=1) < np.radians(T.BASE_MAX)
        for q in (0.5, 1, 2, 5, 10):
            P = np.zeros_like(Y)
            for j in range(5):
                te = np.arange(len(X))[j::5]; tr = np.setdiff1d(np.arange(len(X)), te)
                P[te] = np.percentile(Y[tr], q, 0) * lvl[te, None]
            E = P - Y
            print("  q %4.1f     %5.1f%%       %5.1f%%        %5.1f%%       %.2f" % (
                q, 100 * (E > 0).mean(), 100 * (E > 0.02).mean(), 100 * (E > 0.05).mean(), P.sum() / Y.sum()))
        # the clean score: settings picked on states the final score never sees. 60% stored, 20% to pick on, 20% to score
        i = np.random.default_rng(0).permutation(len(X)); n = len(X) // 5
        # a level state with no reach on any line starts with the arm inside the bus clearance: state() drives there with
        # no clearance check and the tool would refuse that move. Not a state an operator can be in, so it is left out
        ok = ~(lvl & (Y.max(1) == 0)); st, pk, sc = (a[ok[a]] for a in (i[2 * n:], i[:n], i[n:2 * n]))
        print("left out, arm inside the bus clearance at the start: rows", np.flatnonzero(~ok))
        left = np.clip(1 - np.linalg.norm(X[:, 7:], axis=1) / np.radians(T.BASE_MAX), 0.05, 1)[:, None]   # share of the tilt budget left
        rng = np.random.default_rng(1)
        def score(P, Yt):
            """-> share of pairs over by 0.05, reach kept, and per state the share of its lines over by 0.05."""
            return (P - Yt > 0.05).mean(), P.sum() / Yt.sum(), (P - Yt > 0.05).mean(1)
        def say(P, Yt):
            # the lines of one state fail together, so the interval resamples states, not pairs
            a, kept, v = score(P, Yt); lo, hi = np.percentile(v[rng.integers(0, len(v), (4000, len(v)))].mean(1), [2.5, 97.5])
            return "over by 0.05 on %.1f%% of pairs (95%% over states: %.1f to %.1f), in %d of %d states, worst %.2f m, reach kept %.2f" % (
                100 * a, 100 * lo, 100 * hi, (v > 0).sum(), len(v), (P - Yt).max(), kept)
        look = lambda k, pull, on: np.array([safe(X[st], Y[st], k, pull)(x) for x in X[on]])
        fixed = lambda q, on: np.percentile(Y[st], q, 0) * lvl[on, None]
        # the fair fixed map: one map that shrinks with the tilt already spent (the lookup's neighbours carry that for free)
        s2 = st[lvl[st]]; shrunk = lambda q, pull, on: pull * np.percentile(Y[s2] / left[s2], q, 0) * left[on] * lvl[on, None]
        fs = f.with_name("learnsim_medium.npz")
        if fs.exists():
            z = np.load(fs); I, Ys = z["idx"], z["Y"]; assert np.isin(I, i[n:2 * n]).all(); I, Ys = I[ok[I]], Ys[ok[I]]
            print("the fast labels against the full simulation, %d of the scoring states:" % len(I)); row("fast label", Y[I] - Ys)
        for name, g, grid in (("looked up", look, [(k, p) for k in (1, 5, 10, 20) for p in (1.0, 0.9, 0.8, 0.7)]),
                              ("one fixed map", fixed, [(q,) for q in (0.5, 1, 2, 3, 5)]),
                              ("fixed map x tilt left", shrunk, [(q, p) for q in (1, 2, 5, 10, 20) for p in (1.0, 0.9, 0.8, 0.7)])):
            best = max((a for a in grid if score(g(*a, pk), Y[pk])[0] < 0.01), key=lambda a: score(g(*a, pk), Y[pk])[1], default=None)
            if best is None:
                print("clean, %s: no setting stays under 1%% on the picking states" % name); continue
            print("clean, %s: picked %s on %d states. on %d others: %s" % (name, best, len(pk), len(sc), say(g(*best, sc), Y[sc])))
            if fs.exists():
                P = g(*best, I)
                print("   against the full simulation, %d states: over at all %.1f%%, %s" % (len(I), 100 * (P > Ys).mean(), say(P, Ys)))
        sys.exit()
    tilt = np.degrees(np.linalg.norm(X[:, 7:], axis=1))
    print("%d states, tilt there median %.2f, most %.2f deg. reach median %.2f m, %d states with no reach at all" % (
        len(X), np.median(tilt), tilt.max(), np.median(Y), (Y.max(1) < 0.01).sum()))
    for label, pick in (("one move from rest", np.arange(len(X)) < N // 2), ("two moves", np.arange(len(X)) >= N // 2), ("all", np.ones(len(X), bool))):
        idx = np.flatnonzero(pick); E = {}
        for k in range(5):                                            # 5 folds, trained on all the other states
            te = idx[k::5]; tr = np.setdiff1d(np.arange(len(X)), te)
            for name, g in fit(X[tr], Y[tr]).items():
                E.setdefault(name, []).append(np.array([g(x) for x in X[te]]) - Y[te])
        print("left-out states, %s:" % label)
        for name, e in E.items():
            row(name, np.concatenate(e))
    fm = np.load(T.OUT.parent / "fastmap_medium.npz"); S = np.load(T.OUT.parent / "wide98_medium.npz")["S"]
    Xt = np.array([feat(*state(l)[1::2]) for l in [()] + [(s,) for s in S[:4]]])
    Yt = np.array([fm["true%d" % i] for i in range(5)])
    print("against the simulated map, rest + 4 starts:")
    for name, g in fit(X, Y).items():
        row(name, np.array([g(x) for x in Xt]) - Yt)
