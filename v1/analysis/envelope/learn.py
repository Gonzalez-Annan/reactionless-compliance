"""Can the live map be looked up instead of computed? Run from the repo root.
  learn.py gen   200 states (100 one move from rest, 100 two moves), from each the 98 reaches with livemap.fast
                 -> data/learn_medium.npz. About 40 min on 10 processes.
  learn.py       fit on those, test on states left out, then on the 5 states of fastmap.py that have a simulated map.
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


def row(name, E):
    print("  %-10s error median %.3f, 95%% %.3f, worst %.3f m, over by 0.05: %.1f%%, short by 0.05: %.1f%%" % (
        name, np.median(abs(E)), np.percentile(abs(E), 95), abs(E).max(), 100 * (E > 0.05).mean(), 100 * (E < -0.05).mean()))


if __name__ == "__main__":
    f = T.OUT.parent / "learn_medium.npz"
    if "gen" in sys.argv:
        m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
        tips = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1])
        A, B = [dirs(s, N, tips, p0) * np.random.default_rng(s + 1).uniform(0.3, 0.8, N)[:, None] for s in (21, 23)]
        legs = [(a,) if i < N // 2 else (a, b) for i, (a, b) in enumerate(zip(A, B))]
        with Pool(10) as pl:
            R = np.array(pl.map(one, legs, chunksize=1))
        np.savez(f, X=R[:, :10], Y=R[:, 10:])
        sys.exit()
    z = np.load(f); X, Y = z["X"], z["Y"]
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
