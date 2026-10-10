"""Band chart from the v8 data: where the tilt limit, not arm length, sets the reach, and where reach falls under
the parking error. Run from the repo root after v8.py. -> paper/figs/fig_band.{pdf,png}"""
import sys; sys.path.insert(0, "src")
import pathlib, numpy as np, mujoco, teleop
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
here = pathlib.Path(__file__).parent; PARK = 0.25           # m, ASSUMED parking error (no source yet)
ks = (0.1, 0.2, 0.33, 0.5, 1.0, 2.0, 3.3, 10.0); bind, short = [], []
for k in ks:
    m, d, sid, Rb0 = teleop.setup("medium")
    b = m.jnt_bodyid[0]; m.body_mass[b] *= k; mujoco.mj_forward(m, d)   # hand rest position does not depend on the bus mass
    z = np.load(here / ("v8_%g.npz" % k)); R = np.linalg.norm(z["rays"] - d.site_xpos[sid], axis=2)
    bind.append([(z["bind"] > j).sum() for j in range(3)]); short.append((R < PARK).sum(axis=0))
bind, short, mass = np.array(bind), np.array(short), 300 * np.array(ks)
assert list(bind[4]) == [97, 93, 63] and list(short[4]) == [45, 25, 4]     # the 300 kg row printed by v8.py
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6), sharex=True)
for j, lim in enumerate(teleop.LIMITS):
    ax[0].semilogx(mass, bind[:, j], "o-", label="%g deg" % lim); ax[1].semilogx(mass, short[:, j], "o-")
ax[0].set_ylabel("directions where tilt\nstops the hand (of 98)"); ax[1].set_ylabel("directions with reach\nunder %.2f m (of 98)" % PARK)
for a in ax:
    a.set_xlabel("bus mass (kg), 12.5 kg arm"); a.grid(alpha=.3)
ax[0].legend(title="tilt limit", fontsize=7)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig("paper/figs/fig_band." + e, dpi=200)
print("ok")
