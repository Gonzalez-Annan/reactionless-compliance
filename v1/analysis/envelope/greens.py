"""Why does the band show goals green that are refused? Run from the repo root. About 5 min, 10 processes.
python analysis/envelope/greens.py
The 700 checks of static.py after a move, with the probed cage in (livemap.sharpen). A wrong green is a goal
inside the inner edge whose rehearsal is refused. For each one: how the rehearsal ended and how far short, and how
far a probe line gets along the goal's own direction (full reach, the probe's own clock).
Written down before the run, and it was wrong: the straight join between two probed lines cuts the corner (D71).
-> the wrong greens one by one, and what pulling the green edge in to a share of the probed cage would cost."""
import sys; sys.path[:0] = ["src", "analysis/envelope"]
import numpy as np
from multiprocessing import Pool
import teleop as T, livemap, static as S

if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    rest = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    U = np.random.default_rng(0).normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]    # the 20 directions of static.py
    E = [u / T.depth(list(rest), p0, p0 + u) for u in U]
    A = S.FRACS + S.OUTER
    N = range(len(S.DETS)), range(len(E)), range(len(A))
    with Pool(10) as pl:
        R = pl.map(S.one, [(e, det, "medium", T.BASE_MAX) for det in S.DETS for e in E])
    R = [R[i:i + len(E)] for i in range(0, len(R), len(E))]
    with Pool(10, livemap.init) as pl:                        # the cage probed from where each move ended, as in static.py
        P = [(d_.site_xpos[sid_].copy(), list(np.array(livemap.ask(pl, d_).get())), livemap.band(m_, d_, sid_, Rb0_))
             for m_, d_, sid_, Rb0_, _, _ in map(S.moved, S.DETS)]
    S2 = [livemap.sharpen(B, tips, c) for c, tips, B in P]
    why = np.array([[[R[i][k][1][j] for j in N[2]] for k in N[1]] for i in N[0]])
    z0 = np.array([[[R[i][k][2][j] for j in N[2]] for k in N[1]] for i in N[0]])                  # zone as looked up
    dP = np.array([[[T.depth(P[i][1], P[i][0], p0 + A[j] * E[k]) for j in N[2]] for k in N[1]] for i in N[0]])   # 1 is the probed cage's edge
    z1 = np.array([[[sum(T.depth([s[x] for s in S2[i]], P[i][0], p0 + A[j] * E[k]) > 1 for x in (0, 2)) for j in N[2]] for k in N[1]] for i in N[0]])
    ok = why == ""
    print("checks", ok.size, "reachable", ok.sum(), "not", (~ok).sum())
    print("probed cage in. reachable by zone:", [int((ok & (z1 == z)).sum()) for z in range(3)], " not reachable:", [int((~ok & (z1 == z)).sum()) for z in range(3)])
    print("green and inside the probed cage agree on", int(((z1 == 0) == (dP <= 1 + 1e-6)).sum()), "of", ok.size)
    print("\nif the inner edge is pulled in to a share of the probed cage:")
    for f in (1.0, 0.98, 0.95, 0.9, 0.85, 0.8):
        print("  %.2f: green %d of %d reachable, %d of %d not reachable" % (f, (ok & (dP <= f)).sum(), ok.sum(), (~ok & (dP <= f)).sum(), (~ok).sum()))
    print("\nthe wrong greens:")
    for i, k, j in np.argwhere((z1 == 0) & ~ok):
        m, d, sid, Rb0, _, base = S.moved(S.DETS[i])
        c = d.site_xpos[sid].copy(); g = p0 + A[j] * E[k]; dist = np.linalg.norm(g - c); v = (g - c) / dist
        w1, w2 = [], []
        far = T.ray(m, d, sid, Rb0, v, 5, "3d", dist, T.STALL, why=w1)          # the rehearsal
        own = T.ray(m, d, sid, Rb0, v, 5, "3d", why=w2)                           # a probe line along the goal's own direction
        cos = np.array(T.DIRS) @ v; near = np.argsort(-cos)[:3]
        tips = np.array(P[i][1])
        print("  move %s dir %2d goal %.2f | looked-up zone %d | goal %.3f m from the hand, depth on the probed cage %.3f"
              % (S.DETS[i], k, A[j], z0[i, k, j], dist, dP[i, k, j]))
        print("      rehearsal: %s, got %.3f m, %.1f cm short. own probe line: %s, got %.3f m (%.1f cm %s the goal)"
              % (w1[0] if w1 else "time", (far - c) @ v, 100 * np.linalg.norm(far - g), w2[0] if w2 else "time", (own - c) @ v,
                 100 * abs((own - c) @ v - dist), "past" if (own - c) @ v > dist else "short of"))
        print("      nearest probed lines: " + ", ".join("%.0f deg off, reach %.3f m" % (np.degrees(np.arccos(min(1, cos[q]))), np.linalg.norm(tips[q] - c)) for q in near))
