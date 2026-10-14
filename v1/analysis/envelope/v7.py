"""Parking box: the shortest reach over 98 directions (a sphere of that radius around the rest hand position is
reachable everywhere) per bus size and tilt limit, against a 0.25 m parking error.
ponytail: sphere centred on the rest hand position; a better-centred aim point could be a little larger."""
import sys; sys.path.insert(0, "src")
import pathlib, numpy as np, teleop
here = pathlib.Path(__file__).parent; D = np.array(teleop.DIRS)
for size in ("small", "medium", "large"):
    m, d, sid, Rb0 = teleop.setup(size); p0 = d.site_xpos[sid].copy()
    f = here / ("v7_%s.npy" % size)
    if size == "medium":
        rays = np.load(teleop.OUT.parent / "decide4_medium.npz")["rays"]
    elif f.exists():
        rays = np.load(f)
    else:
        rays = []
        for u in D:
            mk = []; best = teleop.ray(m, d, sid, Rb0, u, 5, "3d", marks=mk); rays.append((mk + [best] * 3)[:3])
        rays = np.array(rays); np.save(f, rays)
    R = np.linalg.norm(rays - p0, axis=2)                       # (98, 3)
    print(size, " ".join("%g deg: min %.3f median %.3f m, lines under 0.25 m: %d of 98;" % (lim, R[:, k].min(), np.median(R[:, k]), (R[:, k] < 0.25).sum())
                         for k, lim in enumerate(teleop.LIMITS)), flush=True)
