"""Does the gap of static.py hold when the set-up changes? Run from the repo root. About 30 min, 10 processes.
python analysis/envelope/robust.py           every set-up below
python analysis/envelope/robust.py 0 4       only these, by place in SETUPS
python analysis/envelope/robust.py --clock 4 the rehearsals get 4 times their length at V_GO and not 2 (D72)
One thing is changed at a time against static.py (medium servicer, 2 deg, 0.3 m first move): the size of the
first move, the servicer, the tilt limit. For each set-up the cage is drawn from rest with that servicer and that
limit, 98 lines. The 20 directions and the 7 depths of static.py give the goals, and each goal is rehearsed from
rest and again after each first move, as there.
A first move is left out, and named, if the hand does not arrive or the base passes the limit on the way.
Written down before the run. Each of these can fail:
  the gap grows with the move, 0.1 < 0.2 < 0.3 < 0.4 m
  it is close to nothing on the large servicer, where the base hardly tilts
  it is bigger under a 1 deg limit and smaller under 3 deg
Known from a pre-check: the small servicer (30 kg) can not make four of the five 0.3 m moves. So it is only run
at 0.1 m, where three of the five arrive, next to the large one at 0.1 m.
-> the counts per first move, and one line per set-up at the end. The first set-up must give the counts of static.py."""
import sys; sys.path[:0] = ["src", "analysis/envelope"]
from collections import Counter
import numpy as np
from multiprocessing import Pool
import teleop as T, static as S

SETUPS = (("as static.py", "medium", 2.0, 0.3), ("move 0.1 m", "medium", 2.0, 0.1), ("move 0.2 m", "medium", 2.0, 0.2),
          ("move 0.4 m", "medium", 2.0, 0.4), ("large servicer", "large", 2.0, 0.3), ("large, move 0.1 m", "large", 2.0, 0.1),
          ("small, move 0.1 m", "small", 2.0, 0.1), ("limit 1 deg", "medium", 1.0, 0.3), ("limit 3 deg", "medium", 3.0, 0.3))


K = [T.SLACK, T.SLACK]                                        # the clock of the cage lines, and of the rehearsals: --clock sets that one (D72)


def clock(k):
    K[1] = k                                                  # per process


def one(job):
    T.SLACK = K[1]
    return S.one(job)


def tip(job):
    T.SLACK = K[0]                                            # the cage keeps its own clock
    size, u, T.BASE_MAX = job                                 # the tilt limit is set per process, as in static.one
    m, d, sid, Rb0 = T.setup(size)
    return T.ray(m, d, sid, Rb0, u, 5, "3d")


def run(pl, size, lim, L, U):
    """-> moves kept, wrong goes, checks they are out of, wrong noes, checks they are out of, deepest wrong go and
    furthest wrong no in cm. The band zone that static.one also returns is not read: its lookup is medium at 2 deg."""
    m, d, sid, Rb0 = T.setup(size); p0 = d.site_xpos[sid].copy()
    tips = pl.map(tip, [(size, u, lim) for u in T.DIRS])      # the cage from rest for this servicer and this limit
    E = [u / T.depth(tips, p0, p0 + u) for u in U]
    dets = []
    for det in S.DETS:
        det, pk = tuple(float(x) for x in np.sign(det) * L), []
        try:
            S.moved(det, size, pk)
        except AssertionError:
            print("    move %s left out: the hand did not arrive (base up to %.2f deg)" % (det, pk[0])); continue
        if pk[0] > lim:
            print("    move %s left out: the base went to %.2f deg on the way" % (det, pk[0])); continue
        dets.append((det, pk[0]))
    R = pl.map(one, [(e, det, size, lim) for det in [None] + [x for x, _ in dets] for e in E])
    n, A, nin = len(E), S.FRACS + S.OUTER, len(S.FRACS)
    R = [R[i:i + n] for i in range(0, len(R), n)]
    go = [(k, j) for k in range(n) for j in range(nin) if R[0][k][1][j] == ""]
    no = [(k, j) for k in range(n) for j in range(nin, len(A)) if R[0][k][1][j]]
    print("    cage edge %.2f m from the hand on average. From rest %d of %d goals inside go, %d of %d outside are refused."
          % (np.mean([np.linalg.norm(e) for e in E]), len(go), n * nin, len(no), n * len(S.OUTER)))
    bad, won = [], []
    for (det, pk), rows in zip(dets, R[1:]):
        f = [(k, j, rows[k][1][j]) for k, j in go if rows[k][1][j]]
        g = [(k, j) for k, j in no if rows[k][1][j] == ""]
        bad += f; won += g
        print("    after %s (base up to %.2f deg, left at %.2f): wrong go %d of %d %s, wrong no %d of %d"
              % (det, pk, rows[0][0], len(f), len(go), dict(Counter(w for _, _, w in f)), len(g), len(no)), flush=True)
    return (len(dets), len(bad), len(go) * len(dets), len(won), len(no) * len(dets),
            max([100 * (1 - A[j]) * np.linalg.norm(E[k]) for k, j, _ in bad], default=0),
            max([100 * (A[j] - 1) * np.linalg.norm(E[k]) for k, j in won], default=0))


if __name__ == "__main__":
    U = np.random.default_rng(0).normal(size=(20, 3)); U /= np.linalg.norm(U, axis=1)[:, None]    # the 20 directions of static.py
    a = sys.argv[1:]; k = float(a.pop(a.index("--clock") + 1)) if "--clock" in a else T.SLACK
    want = {2.0: (36, 400, 30, 270), 4.0: (28, 400, 32, 270)}     # the counts of static.py at these two clocks
    rows = []
    print("A rehearsal may take %g times its length at %.2f m/s, plus 1 s." % (k, T.V_GO))
    with Pool(10, clock, (k,)) as pl:
        for name, size, lim, L in [SETUPS[int(x)] for x in a if x != "--clock"] or SETUPS:
            print("\n%s: %s servicer, %g deg, %g m" % (name, size, lim, L), flush=True)
            rows.append((name,) + run(pl, size, lim, L, U))
            assert (size, lim, L) != SETUPS[0][1:] or k not in want or rows[-1][2:6] == want[k], "not the counts of static.py"
    print("\n%-19s moves  wrong go             wrong no             deepest wrong go  furthest wrong no" % "set-up")
    for name, k, b, nb, w, nw, deep, far in rows:
        print("%-19s %d      %3d of %3d (%4.1f%%)   %3d of %3d (%4.1f%%)   %4.1f cm           %4.1f cm"
              % (name, k, b, nb, 100 * b / max(nb, 1), w, nw, 100 * w / max(nw, 1), deep, far))
