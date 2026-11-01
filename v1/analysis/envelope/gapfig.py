"""Where the rest map is wrong after one move, from what static.py 4 saved. Run from the repo root. Scripted runs, no participants.
One panel per move. x: how far out the goal is, 1 = the edge of the map drawn at rest. y: the 20 directions.
A cross is a wrong go (inside the map, the arm can no longer reach it), a circle a wrong no (outside the map, the arm
now reaches it), a dot a goal the map still gets right. Goals the map already gets wrong from rest are left out.
python analysis/envelope/gapfig.py -> paper/figs/fig_gap.pdf and .png"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

z = np.load("data/static_medium_4.npz")
why, A, edge = z["why"], z["A"], z["edge"]                     # why[rest + moves][direction][goal], "" = arrived
inside = A <= 1
go = (why[0] == "") & inside                                  # the goals the rest map lets through, and the arm reaches
no = (why[0] != "") & ~inside                                 # the goals the rest map refuses, and the arm can not reach
wgo, wno = (why[1:] != "") & go, (why[1:] == "") & no         # [move][direction][goal]
assert (wgo.sum(), go.sum() * 5, wno.sum(), no.sum() * 5) == (28, 400, 32, 270), (wgo.sum(), go.sum(), wno.sum(), no.sum())
names = {(0, 0, .3): "+z (up)", (0, 0, -.3): "-z (down)", (-.3, 0, 0): "-x", (.3, 0, 0): "+x", (0, .3, 0): "+y"}
fig, ax = plt.subplots(1, 5, figsize=(7, 3.4), sharey=True)
y, x = np.mgrid[:len(edge), :len(A)]
for i, a in enumerate(ax):
    ok = (go | no) & ~wgo[i] & ~wno[i]
    a.plot(A[x[ok]], y[ok], ".", color="0.75", ms=3)
    a.plot(A[x[wgo[i]]], y[wgo[i]], "x", color="#D55E00", ms=6, mew=1.5)
    a.plot(A[x[wno[i]]], y[wno[i]], "o", mfc="none", mec="#0072B2", ms=6, mew=1.5)
    a.axvline(1, color="k", lw=0.8, ls=":")
    a.set_title("move %s\n%d wrong go\n%d wrong no" % (names[tuple(np.round(z["dets"][i], 1))], wgo[i].sum(), wno[i].sum()), fontsize=7)
    a.set(xlim=(0.55, 1.25), xticks=(0.6, 0.8, 1.0, 1.2), ylim=(-0.7, len(edge) - 0.3))
    a.tick_params(labelsize=7)
ax[0].set_ylabel("direction", fontsize=8); ax[0].set_yticks(range(0, len(edge), 5))
ax[2].set_xlabel("goal distance / edge of the rest map", fontsize=8)
h = [Line2D([], [], ls="", marker=m, ms=6, mew=1.5, color=c, mfc=f) for m, c, f in
     (("x", "#D55E00", None), ("o", "#0072B2", "none"), (".", "0.75", None))]
fig.legend(h, ("wrong go (inside, not reachable)", "wrong no (outside, reachable)", "map right"),
           loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for e in ("pdf", "png"):
    fig.savefig(f"paper/figs/fig_gap.{e}", dpi=200)
d = edge[:, None] * (1 - A)                                   # cm inside the edge, negative outside
print("wrong go %d of %d, wrong no %d of %d; per move %s / %s; deepest wrong go %.1f cm inside, furthest wrong no %.1f cm outside"
      % (wgo.sum(), go.sum() * 5, wno.sum(), no.sum() * 5, wgo.sum((1, 2)).tolist(), wno.sum((1, 2)).tolist(),
         100 * (d * wgo).max(), -100 * (d * wno).min()))
