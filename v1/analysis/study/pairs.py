"""Goal sets of the maps study: pairs of a lead goal and a test goal, drawn with a fixed seed, rehearsed and sorted
into the four classes of the design. Simulation only, no people. Writes data/pairs_<size>.csv, which teleop.py --maps reads.
Run from v1: python analysis/study/pairs.py     (10 processes. The rehearsals are kept in data/pairs_cand_<size>.npz,
                                                 so a second run only picks and takes seconds)

A pair: the hand goes from rest to the lead goal (LEAD of the way out in the map that held, so every map allows it),
then straight to the test goal. The classes, by where the test goal lies and whether that second move can be made:
  trap    inside the rest map, outside the map that held, cannot be reached
  edge    inside the rest map, outside the map that held, can be reached
  inner   inside the map that held, can be reached
  outer   at 1.2 to 1.4 of the rest edge, cannot be reached
A pair is kept only if it is in the same class when the lead move ends 2 cm off the lead goal in each of six directions.
"Cannot be reached" means the hand ends GAP or more short: the tool counts a goal as reached within 2 cm, so a goal
missed by a centimetre would be no trap. "Outside the map that held" means GAP or more outside it, for the same reason.
A set has NEED scored pairs and 2 more that start its block and are not scored (1 edge, 1 trap). 4 pairs are for
training. The traps come from three bands of depth, SETS traps next to each other in each, so every set gets traps of
nearly the same depth: the low and the high band are scored, the middle one starts the blocks.
Every trap of a set has an edge goal within 0.03 of its depth, and the other edge goals are the next nearest.
The map from the current state is not used to sort. For the traps and edge goals that were picked it is probed and
printed: between its 98 lines it is a straight join, so it can allow a trap or stop short of an edge goal."""
import sys; sys.path.insert(0, "src")
import multiprocessing, time, numpy as np, pandas as pd, mujoco
import teleop as T, livemap

SIZE, SEED, N_IN, N_OUT, SETS = "medium", 0, 2000, 80, 4
NEED = dict(trap=2, edge=4, inner=2, outer=2)       # scored pairs of a set. The fallback of the design is edge=3, inner=1
TRAIN = dict(edge=1, inner=2, outer=1)              # the training pairs, in no set
GAP = 0.04                                          # m, twice T.NEAR
OFF = 0.02 * np.r_[np.zeros((1, 3)), np.eye(3), -np.eye(3)]    # where the lead move ends: on the lead goal, and 2 cm off it
F = T.OUT.parent
CAND = F / f"pairs_cand_{SIZE}.npz"


def lead(m, d, sid, Rb0, L):
    """From rest to the lead goal and 1 s more, so the hand has settled as it has when a participant sends.
    -> the state there, or None."""
    mujoco.mj_resetData(m, d)
    d.qpos[7:] = T.Q_REST
    mujoco.mj_forward(m, d)
    d1 = T.reachable(m, d, sid, Rb0, L, 5, "3d")
    if d1 is None:
        return None
    for i in range(int(1 / m.opt.timestep)):
        if i % T.DECIM == 0:
            T.ctrl(m, d1, sid, Rb0, L, 5, "3d")
        mujoco.mj_step(m, d1)
    mujoco.mj_forward(m, d1)
    return d1


def one(a):
    """The pair (lead goal, test goal), rehearsed: the second move as teleop.check makes it.
    -> (m the hand ends short of the test goal, what ended the move). -1 if the lead goal was not reached."""
    L, G = a; m, d, sid, Rb0 = livemap.G
    d1 = lead(m, d, sid, Rb0, L)
    if d1 is None:
        return -1.0, "lead"
    e, w = G - d1.site_xpos[sid], []
    far = T.ray(m, d1, sid, Rb0, e / np.linalg.norm(e), 5, "3d", np.linalg.norm(e), T.STALL, why=w)
    return np.linalg.norm(far - G), (w or ["ok"])[0]


def load():
    """The candidates and what has been rehearsed of them so far. Drawn once per (SEED, N_IN, N_OUT)."""
    key = np.array([SEED, N_IN, N_OUT])
    if CAND.exists():
        c = dict(np.load(CAND))
        if (c["key"] == key).all():
            return c
    rays = list(np.load(F / f"decide4_{SIZE}.npz")["rays"])
    m, d, sid, _ = T.setup(SIZE); home = d.site_xpos[sid].copy()
    rest, held = [r[-1] for r in rays], [r[-1] for r in T.guaranteed(rays, home, SIZE)]
    rng, L, G = np.random.default_rng(SEED), [], []
    while len(L) < N_IN + N_OUT:
        u, v = (x / np.linalg.norm(x) for x in rng.normal(size=(2, 3)))
        k = rng.uniform(1.2, 1.4) if len(L) >= N_IN else rng.uniform(0.5, 1.0)
        a, b = home + u * T.LEAD / T.depth(held, home, home + u), home + v * k / T.depth(rest, home, home + v)
        if np.linalg.norm(a - home) >= 0.08 and np.linalg.norm(b - home) >= 0.15 and np.linalg.norm(b - a) >= 0.10:
            L.append(a); G.append(b)       # a lead goal that moves the arm, a test goal that is a move of its own
    n = len(L)
    return dict(key=key, home=home, L=np.array(L), G=np.array(G), kr=np.array([T.depth(rest, home, g) for g in G]),
                ks=np.array([T.depth(held, home, g) for g in G]), S=np.full((n, len(OFF)), np.nan),
                why=np.full(n, "", "<U5"), live=np.full(n, np.nan))


def sim(pl, c, jobs):
    """Rehearses the (candidate, where the lead move ends) jobs that have no result yet and keeps them."""
    jobs, t = [(i, j) for i, j in jobs if np.isnan(c["S"][i, j])], time.time()
    for n, ((i, j), (s, w)) in enumerate(zip(jobs, pl.imap(one, [(c["L"][i] + OFF[j], c["G"][i]) for i, j in jobs], chunksize=4)), 1):
        c["S"][i, j] = s
        if j == 0:
            c["why"][i] = w
        if n % 200 == 0:
            print(f"  {n}/{len(jobs)} rehearsals, {time.time() - t:.0f} s", flush=True)
    np.savez(CAND, **c)


def classes(c):
    """-> class name: (candidates, 7) is the pair in that class with the lead move ending there. Not rehearsed = no."""
    S, r = c["S"], np.linalg.norm(c["G"] - c["home"], axis=1)
    hit, miss = (S >= 0) & (S < T.TOL), S >= GAP
    between = ((c["kr"] < 1) & (r * (1 - 1 / c["ks"]) >= GAP))[:, None]
    return {"trap": between & miss, "edge": between & hit, "inner": (c["ks"] <= 0.9)[:, None] & hit, "outer": (c["kr"] >= 1.2)[:, None] & miss}


def spread(ix, key, n, what):
    """n of the candidates ix, evenly over their order by key."""
    if len(ix) < n:
        raise SystemExit(f"only {len(ix)} {what} pairs hold in all 7 rehearsals, {n} needed: more candidates (N_IN, N_OUT) or fewer pairs")
    ix = ix[np.argsort(key[ix], kind="stable")]
    return ix[np.round(np.linspace(0, len(ix) - 1, n)).astype(int)]


if __name__ == "__main__":
    c = load()
    kr, N, rows, gaps = c["kr"], len(c["kr"]), [], []
    snake = lambda j: (j % SETS, SETS - 1 - j % SETS)[j // SETS % 2]       # deals a sorted list so every set gets low and high
    take = lambda i, s, use, cls: rows.append(dict(cand=int(i), set=s, use=use, cls=cls))
    m, d, sid, Rb0 = T.setup(SIZE)
    with multiprocessing.Pool(T.PROCS, livemap.init) as pl:
        sim(pl, c, [(i, 0) for i in range(N)])
        k = classes(c)
        print(N, "candidates. In a class with the lead move ending on the lead goal:", {x: int(v[:, 0].sum()) for x, v in k.items()}, flush=True)
        sim(pl, c, [(i, j) for i in np.flatnonzero(k["trap"][:, 0]) for j in range(1, len(OFF))])
        tr, nb = np.flatnonzero(classes(c)["trap"].all(1)), NEED["trap"] + 1
        if len(tr) < 2 * nb * SETS:
            raise SystemExit(f"only {len(tr)} trap pairs hold in all 7 rehearsals: more candidates (N_IN)")
        tr = tr[np.argsort(kr[tr], kind="stable")]      # bands of depth around 1/6, 3/6, 5/6 of the traps, SETS next to each other in each
        band = [tr[a:a + SETS] for a in np.round((2 * np.arange(nb) + 1) / (2 * nb) * len(tr) - SETS / 2).astype(int)]
        traps = np.concatenate(band[:nb // 2] + band[nb // 2 + 1:] + [band[nb // 2]])      # the middle band last: it starts the blocks
        e0 = np.flatnonzero(k["edge"][:, 0])
        want = [e0[np.argsort(abs(kr[e0] - kr[t]), kind="stable")[:12]] for t in traps]     # the edge goals nearest in depth to a trap
        want += [np.flatnonzero(k[x][:, 0])[:5 * (SETS * NEED[x] + TRAIN[x])] for x in ("inner", "outer")]
        sim(pl, c, [(i, j) for i in np.unique(np.concatenate(want)) for j in range(1, len(OFF))])
        k = classes(c)
        print("of those, in the same class in all 7:", {x: int(v.all(1).sum()) for x, v in k.items()}, "(edge, inner, outer: only as many as needed were tried)", flush=True)

        for j, t in enumerate(traps):       # one of every band to every set, the scored bands dealt against each other
            take(t, snake(j), "block" if j // SETS == NEED["trap"] else "scored", "trap")
        eg = np.flatnonzero(k["edge"].all(1))

        def edge(t, s, use):
            free = [i for i in eg if i not in {r["cand"] for r in rows}]
            if not free:
                raise SystemExit("not enough edge pairs hold in all 7 rehearsals: more candidates (N_IN)")
            i = min(free, key=lambda i: abs(kr[i] - kr[t]))
            take(i, s, use, "edge")
            return abs(kr[i] - kr[t])
        mine = {s: [r["cand"] for r in rows if r["set"] == s and r["use"] == "scored"] for s in range(SETS)}
        for n in range(NEED["edge"]):       # one for every trap first, then the next nearest
            for s in range(SETS):
                gaps += [edge(mine[s][n % len(mine[s])], s, "scored")][:n < len(mine[s])]
        for r in [r for r in rows if r["use"] == "block"]:
            edge(r["cand"], r["set"], "block")
        edge(traps[-1], -1, "train")
        for cls, key in (("inner", c["ks"]), ("outer", kr)):
            ix = spread(np.flatnonzero(k[cls].all(1)), key, SETS * NEED[cls] + TRAIN[cls], cls)
            mid, j = range((len(ix) - TRAIN[cls]) // 2, (len(ix) + TRAIN[cls]) // 2), 0
            for q, i in enumerate(ix):      # the middle of the depths trains, the rest is dealt
                take(i, -1 if q in mid else snake(j), "train" if q in mid else "scored", cls)
                j += q not in mid

        for i in [r["cand"] for r in rows if r["cls"] in ("trap", "edge") and np.isnan(c["live"][r["cand"]])]:
            d1 = lead(m, d, sid, Rb0, c["L"][i])    # the map a participant who waits would be shown
            c["live"][i] = T.depth(livemap.ask(pl, d1).get(), d1.site_xpos[sid].copy(), c["G"][i])
        np.savez(CAND, **c)

    df = pd.DataFrame(rows); i = df.cand.to_numpy()
    for n, a in (("l", c["L"]), ("t", c["G"])):     # as teleop.free_drive takes its targets: from the hand at rest
        for j, x in enumerate("xyz"):
            df[n + x] = a[i, j] - c["home"][j]
    df["depth"], df["depth_held"], df["short_m"], df["why"], df["depth_live"] = kr[i], c["ks"][i], c["S"][i, 0], c["why"][i], c["live"][i]
    df = df.sort_values(["set", "use", "cls", "depth"], kind="stable")
    sc = df[df.use == "scored"]
    assert df.cand.is_unique, "no pair twice"
    assert sc.groupby(["set", "cls"]).size().unstack()[list(NEED)].eq(list(NEED.values())).all().all() and len(sc) == SETS * sum(NEED.values())
    assert (df[df.use == "block"].groupby("set").cls.apply(sorted).map(lambda x: x == ["edge", "trap"])).all()
    assert df[df.use == "train"].cls.value_counts().to_dict() == TRAIN and max(gaps) <= 0.03, max(gaps)
    assert all(classes(c)[x][j].all() for x, j in zip(df.cls, df.cand)), "every pair in its class in all 7 rehearsals"    # df is sorted by now, i is not
    df.to_csv(F / f"pairs_{SIZE}.csv", index=False)

    print(f"\n{len(df)} pairs in data/pairs_{SIZE}.csv: {len(sc)} scored in {SETS} sets, {(df.use == 'block').sum()} to start the blocks, {(df.use == 'train').sum()} for training")
    print("depth of the test goals on the rest map (inner: on the map that held), set -1 = training:")
    df["k"] = np.where(df.cls == "inner", df.depth_held, df.depth)
    print(df.groupby(["set", "use", "cls"]).k.apply(lambda x: " ".join(f"{v:.2f}" for v in x)).unstack().fillna("").to_string())
    print(f"each scored trap has an edge goal within {max(gaps):.3f} of its depth (0.03 allowed)")
    print("scored traps, mean depth by set:", " ".join(f"{v:.3f}" for v in sc[sc.cls == "trap"].groupby("set").depth.mean()))
    tr, ed = df[df.cls == "trap"], df[df.cls == "edge"]
    print("what ends the move on a trap:", tr.why.value_counts().to_dict(), " hand ends %.2f to %.2f m short" % (tr.short_m.min(), tr.short_m.max()))
    print(f"map from the current state, probed after the lead move: it allows {(tr.depth_live < 1).sum()} of the {len(tr)} traps "
          f"and stops short of {(ed.depth_live > 1).sum()} of the {len(ed)} edge goals")
    print("  scored only:", f"{(sc[sc.cls == 'trap'].depth_live < 1).sum()} of {(sc.cls == 'trap').sum()} traps, "
          f"{(sc[sc.cls == 'edge'].depth_live > 1).sum()} of {(sc.cls == 'edge').sum()} edge goals")
