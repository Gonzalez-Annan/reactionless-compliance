"""The work envelope as three flat cuts through the starting hand position, to read it without a 3-D view.
Rings: base tilt 0.5, 1 and 2 degrees. Run from the repo root -> paper/figs/fig_slices.png/.pdf, and reach per shell."""
import sys; sys.path.insert(0, "src")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import teleop as T

rays = T.pool("medium")[0]
m, d, sid, _ = T.setup("medium")
p0 = d.site_xpos[sid].copy()
COL, AX = ("tab:green", "tab:orange", "tab:blue"), "xyz"
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, k in zip(axs, (2, 1, 0)):                      # the plane's normal: z (seen from above), y, x
    i, j = [a for a in range(3) if a != k]
    for sh in (2, 1, 0):
        seg = (T.cut([r[sh] for r in rays], p0, np.eye(3)[k]) - p0) * 100
        for a, b in seg:
            ax.plot([a[i], b[i]], [a[j], b[j]], color=COL[sh], lw=1.6)
        ax.plot([], [], color=COL[sh], label=f"base tilt {T.LIMITS[sh]:g}$^\\circ$")
    ax.plot(0, 0, "k+", ms=10, label="hand at rest")
    ax.set(xlabel=f"{AX[i]} (cm)", ylabel=f"{AX[j]} (cm)", title=f"cut at {AX[k]} = hand", aspect="equal")
    ax.grid(alpha=0.3)
axs[0].legend(fontsize=7, loc="best")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"paper/figs/fig_slices.{e}", dpi=200)
for sh, lim in enumerate(T.LIMITS):
    r = np.array([np.linalg.norm(x[sh] - p0) for x in rays]) * 100
    print(f"{lim:g} deg shell: reach {r.min():.0f} to {r.max():.0f} cm, median {np.median(r):.0f}; "
          f"longest along {np.round(T.DIRS[r.argmax()], 2)}, shortest along {np.round(T.DIRS[r.argmin()], 2)}")
