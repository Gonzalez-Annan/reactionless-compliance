"""Is the cage static? Run from the repo root. Exits 1 if it is not.
The claim under test: a cage drawn once with the arm at rest stays right wherever the hand has been.
The cage is the saved rest map and is never redrawn here. The goals are fixed in the world on 20 random directions:
inside the cage or on its edge (0.6 to 1.0 of the way out) and just outside it (1.05 to 1.2). Each one is rehearsed
from rest and again after one 0.3 m move, with the base tilt that move leaves kept, as it would be in use.
A static cage gives the same answer both times. It can be wrong two ways:
  a wrong go: GO from rest, REFUSED after the move. The cage still shows the goal inside.
  a wrong no: REFUSED from rest, GO after the move. The cage still shows the goal outside.
-> the goals where the answer changes, and what stopped the hand.
Last, what the band display (D70) shows for the same goals after the same moves. It has two edges, looked up from
where the arm is. It is wrong when a goal the arm can reach is past its outer edge, and when a goal the arm can
not reach is inside its inner edge. The second kind costs a refused command, because every go is checked.
Then the same band once the cage probed from where the move ended is in (livemap.sharpen, what the tool draws
after 11 to 20 s), and what would happen if the outer edge were pulled in to that probed cage as well."""
import sys; sys.path.insert(0, "src")
from collections import Counter
import numpy as np, mujoco
from multiprocessing import Pool
import teleop as T, livemap

FRACS = (0.6, 0.8, 0.9, 1.0)
OUTER = (1.05, 1.1, 1.2)
DETS = ((0, 0, 0.3), (0, 0, -0.3), (-0.3, 0, 0), (0.3, 0, 0), (0, 0.3, 0))    # the five moves of start.py far
WHY = dict(tilt="base tilt passed 2 deg", bus="arm would hit the bus", stall="hand stalled short of the goal", time="ran out of time")


def moved(det):
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy(); base = 0.0
    if det is not None:
        for k in range(int(6 / m.opt.timestep)):              # 6 s: 3 s of travel at V_GO, the rest to settle
            if k % T.DECIM == 0:
                p, base = T.ctrl(m, d, sid, Rb0, p0 + np.array(det), 5, "3d")
            mujoco.mj_step(m, d)
        assert np.linalg.norm(p - p0 - det) < T.TOL, "the move itself did not arrive"
    return m, d, sid, Rb0, p0, base


def one(job):
    edge, det = job                                           # edge: the cage's 2 deg edge along one direction, from the rest hand position
    m, d, sid, Rb0, p0, base = moved(det)
    c = d.site_xpos[sid].copy(); B = livemap.band(m, d, sid, Rb0)    # the band's two edges from here: looked up, nothing simulated
    zone = [sum(T.depth([b[i] for b in B], c, p0 + f * edge) > 1 for i in (0, 2)) for f in FRACS + OUTER]   # 0 inside the inner edge, 1 between the two, 2 past the outer one
    out = []
    for f in FRACS + OUTER:                                   # T.check, opened up to read why a line ended
        e = p0 + f * edge - d.site_xpos[sid]; dist = np.linalg.norm(e); why = []
        far = T.ray(m, d, sid, Rb0, e / dist, 5, "3d", dist, T.STALL, why=why)
        out.append("" if np.linalg.norm(far - p0 - f * edge) < T.TOL else why[0] if why else "time")
    return base, out, zone


if __name__ == "__main__":
    m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
    tips = np.load(T.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
    U = np.random.default_rng(0).normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]    # the 20 directions of start.py hard
    E = [u / T.depth(list(tips), p0, p0 + u) for u in U]      # depth 1 is the cage's edge
    with Pool(10) as pl:
        R = pl.map(one, [(e, det) for det in (None,) + DETS for e in E])
    R = [R[i:i + len(E)] for i in range(0, len(R), len(E))]   # [start][direction] -> (base tilt left, [what stopped each goal, "" if it arrived])
    A = FRACS + OUTER; nin = len(FRACS)
    go = [(k, j) for k in range(len(E)) for j in range(nin) if R[0][k][1][j] == ""]
    no = [(k, j) for k in range(len(E)) for j in range(nin, len(A)) if R[0][k][1][j]]
    print("CLAIM UNDER TEST: the cage is static. Drawn once at rest, it is right wherever the hand has been.")
    print("The cage is the saved rest map. It is not redrawn anywhere in this test.")
    print("%d goals fixed in the world: %d directions, each at %s of the way to the cage edge." % (len(E) * nin, len(E), FRACS))
    print("%d more just outside the cage, at %s of the way to its edge." % (len(E) * len(OUTER), OUTER))
    print("\nFrom rest: %d of %d goals GO." % (len(go), len(E) * nin))
    print("From rest: %d of %d goals outside the cage REFUSED.\n" % (len(no), len(E) * len(OUTER)))
    bad, won = [], []
    for det, rows in zip(DETS, R[1:]):
        flips = [(k, j, rows[k][1][j]) for k, j in go if rows[k][1][j]]
        gains = [(k, j) for k, j in no if rows[k][1][j] == ""]
        bad += [(j, k, det, w) for k, j, w in flips]
        won += [(j, k, det) for k, j in gains]
        print("After one move of %s m (base left at %.2f deg, cage not redrawn): %d of %d now REFUSED. By depth %s. %s"
              % (det, rows[0][0], len(flips), len(go), [sum(j == x for _, j, _ in flips) for x in range(nin)],
                 dict(Counter(w for _, _, w in flips))))
        print("    and %d of %d goals outside the cage now GO. By depth %s." % (len(gains), len(no), [sum(j == x for _, j in gains) for x in range(nin, len(A))]))
    print("\nEvery goal that is GO from rest and REFUSED after one move, deepest inside the cage first:")
    for j, k, det, w in sorted(bad):
        print("  direction %2d, goal %.2f of the way out (%4.1f cm inside the edge), after move %s: %s"
              % (k, A[j], 100 * (1 - A[j]) * np.linalg.norm(E[k]), det, WHY[w]))
    print("\nEvery goal that is REFUSED from rest and GO after one move, furthest outside the cage first:")
    for j, k, det in sorted(won, reverse=True):
        print("  direction %2d, goal %.2f of the way out (%4.1f cm outside the edge), after move %s" % (k, A[j], 100 * (A[j] - 1) * np.linalg.norm(E[k]), det))
    print("\nRESULT: %d of %d checks on goals that are GO from rest are REFUSED after one 0.3 m move." % (len(bad), len(go) * len(DETS)))
    print("RESULT: %d of %d checks on goals that are REFUSED from rest are GO after one 0.3 m move." % (len(won), len(no) * len(DETS)))
    print("CLAIM BROKEN: the cage is not static." if bad or won else "CLAIM HOLDS on every goal tested.")
    Z = ("inside the inner edge", "between the two edges", "past the outer edge")
    split = lambda ks: ", ".join("%d %s" % (sum(z == i for z in ks), Z[i]) for i in range(3))
    print("\nTHE BAND DISPLAY on the same goals, after the same moves (two looked-up edges, D70):")
    print("  the %d wrong goes of the rest cage: %s." % (len(bad), split([R[1 + DETS.index(det)][k][2][j] for j, k, det, _ in bad])))
    print("  the %d wrong noes of the rest cage: %s." % (len(won), split([R[1 + DETS.index(det)][k][2][j] for j, k, det in won])))
    def table(zs):                                            # zs[move][direction][goal]
        a = [(w == "", z) for rows, zr in zip(R[1:], zs) for (_, ws, _), zk in zip(rows, zr) for w, z in zip(ws, zk)]
        print("  all %d checks after a move. Can be reached (%d): %s." % (len(a), sum(g for g, _ in a), split([z for g, z in a if g])))
        print("  Can not be reached (%d): %s." % (sum(not g for g, _ in a), split([z for g, z in a if not g])))
    table([[r[2] for r in rows] for rows in R[1:]])
    print("  The band is wrong on the first line where it says past the outer edge, on the second where it says inside the inner edge.")
    with Pool(10, livemap.init) as pl:                        # the cage probed from where each move ended: 98 lines, what the live display shows after its wait
        P = [(d.site_xpos[sid].copy(), list(np.array(livemap.ask(pl, d).get())), livemap.band(m, d, sid, Rb0)) for m, d, sid, Rb0, _, _ in map(moved, DETS)]
    print()
    print("THE SAME BAND ONCE THE PROBED CAGE IS IN (livemap.sharpen, as the tool draws it): the inner edge is moved to the cage")
    print("probed from where the move ended. The outer edge stays, or goes out to the probed cage where that is further.")
    N = range(len(DETS)), range(len(E)), range(len(A))
    dP = [[[T.depth(P[i][1], P[i][0], p0 + A[j] * E[k]) for j in N[2]] for k in N[1]] for i in N[0]]     # 1 is the probed cage's edge
    S = [livemap.sharpen(B, tips, c) for c, tips, B in P]
    table([[[sum(T.depth([s[x] for s in S[i]], P[i][0], p0 + A[j] * E[k]) > 1 for x in (0, 2)) for j in N[2]] for k in N[1]] for i in N[0]])
    print("  And if the outer edge is pulled in to a multiple of the probed cage (never past the looked-up one):")
    for f in (1.0, 1.05, 1.1, 1.2, 1.3):
        a = [(R[1 + i][k][1][j] == "", dP[i][k][j] > f or R[1 + i][k][2][j] == 2) for i in N[0] for k in N[1] for j in N[2]]
        print("    at %.2f: %d that can be reached are past it, and %d of the %d that can not" % (f, sum(g and z for g, z in a), sum(z and not g for g, z in a), sum(not g for g, _ in a)))
    sys.exit(1 if bad or won else 0)
