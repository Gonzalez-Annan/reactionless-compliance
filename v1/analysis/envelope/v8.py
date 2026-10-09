"""Crossover chart data: bus mass against tilt limit. The medium (300 kg) model with its bus scaled by k.
Per mass and limit: median reach, lines under 0.25 m, and lines where the tilt limit stops the hand before the arm does.
ponytail: inertia scaled by the same k as mass (a real bus grows roughly as m^(5/3)); the 30 and 3000 kg rows
are comparable to, not identical with, the ff_small / ff_large models. Upgrade: scale inertia by k**(5/3)."""
import sys; sys.path.insert(0, "src")
import pathlib, numpy as np, mujoco, teleop
here = pathlib.Path(__file__).parent; D = np.array(teleop.DIRS)
for k in (0.1, 0.2, 0.33, 0.5, 1.0, 2.0, 3.3, 10.0):
    m, d, sid, Rb0 = teleop.setup("medium")
    b = m.jnt_bodyid[0]; m.body_mass[b] *= k; m.body_inertia[b] *= k; mujoco.mj_forward(m, d)
    p0 = d.site_xpos[sid].copy(); f = here / ("v8_%g.npz" % k)
    if f.exists():
        z = np.load(f); rays, bind = z["rays"], z["bind"]
    else:
        rays, bind = [], []
        for u in D:
            mk = []; best = teleop.ray(m, d, sid, Rb0, u, 5, "3d", marks=mk)
            rays.append((mk + [best] * 3)[:3]); bind.append(len(mk))      # len(mk) = how many of the three limits were hit
        rays, bind = np.array(rays), np.array(bind); np.savez(f, rays=rays, bind=bind)
    if k == 1.0:                                                         # the unscaled world must reproduce the cached map
        assert np.abs(rays - np.load(teleop.OUT.parent / "decide4_medium.npz")["rays"]).max() < 1e-3
    R = np.linalg.norm(rays - p0, axis=2)
    print("%5.0f kg " % (300 * k) + " | ".join("%g deg: median %.3f m, under 0.25 m %2d, tilt binds %2d" % (lim, np.median(R[:, j]), (R[:, j] < 0.25).sum(), (bind > j).sum())
                                             for j, lim in enumerate(teleop.LIMITS)), flush=True)
