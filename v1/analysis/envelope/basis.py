"""Two checks on what the envelope rests on. Run from the repo root, after the arm-bus clearance went into ray().

1. Collision: how much of the old envelope (data/decide4_medium_nocollision.npz) had the arm inside the bus.
2. Physics: does momentum conservation at the rest pose predict the measured reach per direction, with no simulation?
   a. reactionless speed  s(u) = 1 / |pinv(J N) u|   hand speed along u per unit joint speed with the base dead level
   b. tilt per metre      c(u) = |W pinv(J) u|       base tilt per metre along u if the arm takes the shortest joint path
   Spearman rank correlation of each with the measured reach over the 98 probe directions."""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
from scipy.linalg import null_space
from scipy.stats import spearmanr
import teleop as T
from dynamics import kin

old = np.load("data/decide4_medium_nocollision.npz")
rays, goals, dep, band, truth = T.pool("medium")           # regenerated with the clearance limit if the cache was removed
m, d, sid, _ = T.setup("medium")
p0 = d.site_xpos[sid].copy()
print(f"rest pose: arm {T.clear(m, d) * 100:.0f} cm from the bus, hand {(p0[2] - 0.5) * 100:.0f} cm above its top face")
ro, rn = (np.linalg.norm(r - p0, axis=2) * 100 for r in (old["rays"], rays))     # (98, 3) cm
for sh, lim in enumerate(T.LIMITS):
    cutn = (ro[:, sh] - rn[:, sh] > 1).sum()
    print(f"{lim:g} deg shell: reach was {ro[:, sh].min():.0f}-{ro[:, sh].max():.0f} cm (median {np.median(ro[:, sh]):.0f}), "
          f"now {rn[:, sh].min():.0f}-{rn[:, sh].max():.0f} (median {np.median(rn[:, sh]):.0f}); {cutn} of 98 directions cut short by the bus")
print(f"goals a rehearsal passes: was {old['truth'].sum()} of {old['truth'].size}, now {truth.sum()} of {truth.size}")

k = kin(m, d)
J, W, U = k.Jstar[:3], k.W, np.array(T.DIRS)
N = null_space(W)
s = 1 / np.linalg.norm(np.linalg.pinv(J @ N) @ U.T, axis=0)
c = np.linalg.norm(W @ np.linalg.pinv(J) @ U.T, axis=0)
print("singular values of J on the reactionless joint motions:", np.round(np.linalg.svd(J @ N)[1], 2))
free = ro[:, 2] - rn[:, 2] < 1                              # directions the bus does not cut
for name, x in (("reactionless speed s(u)", s), ("1 / tilt per metre c(u)", 1 / c)):
    print(name + ": " + "; ".join(f"{lim:g} deg rho = {spearmanr(x[free], rn[free, sh])[0]:+.2f}" for sh, lim in enumerate(T.LIMITS))
          + f"  ({free.sum()} directions the bus does not cut)")
pred = np.degrees(c) * rn[:, 0] / 100                       # first-order tilt at the measured 0.5 deg reach: 0.5 if the linear model held
print(f"first-order tilt predicted at the measured 0.5 deg edge: median {np.median(pred[free]):.2f} deg, "
      f"10th-90th pct {np.percentile(pred[free], 10):.2f}-{np.percentile(pred[free], 90):.2f} (0.5 = the linear model is exact)")
