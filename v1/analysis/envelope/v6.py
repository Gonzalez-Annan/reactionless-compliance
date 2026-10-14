"""Robust envelope. The map is drawn for the nominal bus; reality may be 10% lighter or heavier.
Core = per line the shortest 2 deg reach over the three masses, shell = the longest.
Test on goals with truth rehearsed under each mass: nominal map alone vs go-in-core / no-go-outside-shell."""
import sys; sys.path.insert(0, "src")
import pathlib, numpy as np, mujoco, teleop
here = pathlib.Path(__file__).parent; N = 120; KS = (0.9, 1.0, 1.1)
z = np.load(here / "v4.npz"); g, t_nom = z["g"][:N], z["truth"][:N]
nom = np.load(teleop.OUT.parent / "decide4_medium.npz")["rays"][:, -1]
D = np.array(teleop.DIRS)

def world(k):
    m, d, sid, Rb0 = teleop.setup("medium")
    b = m.jnt_bodyid[0]; m.body_mass[b] *= k; m.body_inertia[b] *= k; mujoco.mj_forward(m, d)   # not mj_setConst: it leaves d at qpos0 and broke the first run
    return m, d, sid, Rb0

m, d, sid, Rb0 = world(1.0); p0 = d.site_xpos[sid].copy()
tips, truth = {1.0: nom}, {1.0: t_nom}
for i in (0, 40, 97):          # the unperturbed world must reproduce the cached map, else the harness is wrong
    mk = []; best = teleop.ray(m, d, sid, Rb0, D[i], 5, '3d', marks=mk)
    assert np.linalg.norm((mk + [best] * 3)[2] - nom[i]) < 1e-3, (i, (mk + [best] * 3)[2], nom[i])
print('harness reproduces the cached map', flush=True)
for k in (0.9, 1.1):
    f = here / ("v6_%g.npz" % k)
    if not f.exists():
        w = world(k); r = []
        for u in D:
            mk = []; best = teleop.ray(*w, u, 5, "3d", marks=mk); r.append((mk + [best] * 3)[2])
        print("rays done x%g" % k, flush=True)
        np.savez(f, tips=np.array(r), truth=np.array([teleop.check(*w, x, 5, "3d")[0] for x in g]))
        print("truth done x%g" % k, flush=True)
    q = np.load(f); tips[k], truth[k] = q["tips"], q["truth"]
R = np.array([np.linalg.norm(tips[k] - p0, axis=1) for k in KS])            # (3, 98) reach per mass per line
odd = np.flatnonzero((R.max(0) / R.min(0)) > 1.3)
print("lines whose reach changes over 30%% across masses: %d of 98" % len(odd))
for i in odd[:8]:
    print("  dir", np.round(D[i], 2), "reach", np.round(R[:, i], 3))
U = (nom - p0) / R[1][:, None]
core, shell = p0 + U * R.min(0)[:, None], p0 + U * R.max(0)[:, None]
dep = {n: np.array([teleop.depth(list(t), p0, x) for x in g]) for n, t in (("nom", nom), ("core", core), ("shell", shell))}
for k in KS:
    t = truth[k]; go = dep["nom"] < 1
    cg, sn = dep["core"] < 1, dep["shell"] >= 1
    print("reality x%g (%d of %d reachable): nominal map false go %d, false no-go %d | core false go %d of %d, shell false no-go %d of %d, between %d (reachable %d)"
          % (k, t.sum(), N, (go & ~t).sum(), (~go & t).sum(), (cg & ~t).sum(), cg.sum(), (sn & t).sum(), sn.sum(), (~cg & ~sn).sum(), (t & ~cg & ~sn).sum()))
print("rehearsal on the nominal model disagrees with reality: x0.9 %d, x1.1 %d of %d goals" % ((truth[0.9] != t_nom).sum(), (truth[1.1] != t_nom).sum(), N))
for n, r in (("nominal", R[1]), ("core", R.min(0)), ("shell", R.max(0))):
    print("%s: reach min %.3f, median %.3f, max %.3f m" % (n, r.min(), np.median(r), r.max()))
