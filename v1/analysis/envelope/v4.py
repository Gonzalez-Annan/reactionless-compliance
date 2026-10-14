"""The envelope as a detector: probe-line count sets how sharp it is, a shrink factor sets how cautious.
300 goals near the edge (medium bus), truth by rehearsal. For 26 and 98 lines, sweep the factor s
(aid says go if depth < s) and count false go (unreachable called go) and false no-go (reachable called no-go)."""
import sys; sys.path.insert(0, "src")
import pathlib, numpy as np
from scipy.spatial import ConvexHull
import teleop
here = pathlib.Path(__file__).parent
rays = np.load(teleop.OUT.parent / "decide4_medium.npz")["rays"]
m, d, sid, Rb0 = teleop.setup("medium"); p0 = d.site_xpos[sid].copy()
D = np.array(teleop.DIRS); edge = rays[:, -1]

def depth(tips, tris, g):
    for t in tris:
        A = np.column_stack([tips[i] - p0 for i in t])
        if abs(np.linalg.det(A)) > 1e-12:
            c = np.linalg.solve(A, g - p0)
            if (c >= -1e-9).all():
                return c.sum()
    return np.inf

f = here / "v4.npz"
if f.exists():
    z = np.load(f); g, truth = z["g"], z["truth"]
else:
    rng = np.random.default_rng(4); n = 300
    U = rng.normal(size=(n, 3)); U /= np.linalg.norm(U, axis=1)[:, None]
    g = np.array([p0 + u * k / depth(edge, teleop.TRIS, p0 + u) for u, k in zip(U, rng.uniform(0.75, 1.25, n))])
    truth = np.array([teleop.check(m, d, sid, Rb0, x, 5, "3d")[0] for x in g])
    np.savez(f, g=g, truth=truth)
few = np.array([len(set(np.round(np.abs(u[np.abs(u) > 1e-9]), 4))) == 1 for u in D])   # the 26 of the 98
print("goals %d, reachable %d" % (len(g), truth.sum()))
for name, k in (("26 lines", few), ("98 lines", np.ones(len(D), bool))):
    assert k.sum() == int(name.split()[0])
    tips, tris = edge[k], ConvexHull(D[k]).simplices
    dep = np.array([depth(tips, tris, x) for x in g])
    print(name)
    for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
        go = dep < s
        print("  factor %.2f: false go %2d of %d unreachable, false no-go %3d of %d reachable, right %d of %d"
              % (s, (go & ~truth).sum(), (~truth).sum(), (~go & truth).sum(), truth.sum(), (go == truth).sum(), len(g)))
