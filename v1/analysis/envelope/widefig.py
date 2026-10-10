"""Two pictures of the moving edge, from the sweep wide.py saved. Run from the repo root. Scripted runs, no participants.
Left: every (start, direction) pair, its edge as a fraction of the edge from rest. Most sit at 1, a tail collapses.
Right: the guaranteed edge (worst over the starts used so far) against the number of starts: has it stopped shrinking?
python analysis/envelope/widefig.py        the 20 random directions
python analysis/envelope/widefig.py --98   the 98 lines of the cage
-> paper/figs/fig_wide.pdf / fig_wide98.pdf and .png"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = "wide98" if "--98" in sys.argv else "wide"
z = np.load(f"data/{tag}_medium.npz")
K = z["R"][~np.isnan(z["R"][:, 1]), 1:]                       # the starts that arrived
r = (K / z["rest"]).ravel(); r = r[~np.isnan(r)]
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6))
a.hist(r, bins=np.arange(0.3, 1.35, 0.05), color="0.35")
a.axvline(1, color="k", lw=0.8, ls=":")
a.set(xlabel="edge after a move / edge from rest", ylabel="start-direction pairs", yscale="log")
n = np.arange(1, len(K) + 1)
g = np.array([np.nanmin(K[:i], 0) for i in n])                # (starts used, direction)
b.plot(n, np.median(g, 1), "k-", label="median direction")
b.plot(n, g.min(1), "k--", label="worst direction")
b.plot(n, np.mean(g ** 3, 1) / np.mean(z["rest"] ** 3), color="0.5", label="volume kept")
b.set(xlabel="starts used", ylabel="guaranteed edge (1 = rest map)", ylim=(0, 1.1))
b.legend(frameon=False, fontsize=7)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"paper/figs/fig_{tag}.{e}", dpi=200)
print("pairs %d, within 0.05 of rest %.0f%%, under 0.8 %.0f%%; guaranteed at %d starts: median %.2f worst %.2f volume %.0f%%"
      % (len(r), 100 * np.mean(abs(r - 1) <= 0.05), 100 * np.mean(r < 0.8), len(K), np.median(g[-1]), g[-1].min(),
         100 * np.mean(g[-1] ** 3) / np.mean(z["rest"] ** 3)))
