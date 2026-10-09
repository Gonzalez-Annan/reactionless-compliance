"""Figures 1, 2, 3, 5 from data/sweep.csv. Figure 4 (operator ratings) needs participant data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from schemes import NAMES  # noqa: E402
from task import MODELS, Q_REST  # noqa: E402
from safety import vmax_map  # noqa: E402

OUT = ROOT / "paper" / "figs"
SIZES, MASS = ["small", "medium", "large"], [30, 300, 3000]
COL = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868", 4: "#c44e52", 5: "#8172b3"}
MARK = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "pdf.fonttype": 42})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)


def main():
    df = pd.read_csv(ROOT / "data" / "sweep.csv")
    base, sw = df[~df.sweep], df[df.sweep]

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for a, size, mass in zip(ax, SIZES, MASS):
        for s in (4, 5):
            c = sw[(sw["size"] == size) & (sw.scheme == s)].sort_values("beta")
            a.plot(c.max_base_deg, c.rms_pos_mm, "-", marker=MARK[s], ms=3, color=COL[s], label=NAMES[s])
        for s in (1, 2, 3):
            p = base[(base["size"] == size) & (base.scheme == s) & (base["mode"] == "6d")]
            a.plot(p.max_base_deg, p.rms_pos_mm, MARK[s], ms=5, color=COL[s], label=NAMES[s])
        a.axvline(5, color="k", ls="--", lw=.8)
        a.set(xscale="log", yscale="log", title=f"{size} ({mass} kg base)", xlabel="max base attitude drift (deg)")
    ax[0].set_ylabel("RMS hand position error (mm)")
    ax[0].legend(fontsize=6)
    save(fig, "fig1_tradeoff")

    fig, a = plt.subplots(figsize=(3.5, 2.6))
    for s in range(1, 6):
        p = base[(base.scheme == s) & (base["mode"] == "6d")].set_index("size").loc[SIZES]
        a.plot(MASS, p.max_base_deg, "-", marker=MARK[s], ms=4, color=COL[s], label=NAMES[s])
    a.axhline(5, color="k", ls="--", lw=.8)
    a.set(xscale="log", yscale="log", xlabel="base mass (kg), arm 12.5 kg", ylabel="max base attitude drift (deg)")
    a.legend(fontsize=6)
    save(fig, "fig2_size")

    fig, a = plt.subplots(figsize=(3.5, 2.4))
    med = base[base["size"] == "medium"]
    for j, mode in enumerate(["3d", "6d"]):
        v = med[med["mode"] == mode].sort_values("scheme").q_return_rad
        a.bar(np.arange(5) + (j - .5) * .38, v, .38, label=mode.upper() + " task", color=["#999", "#444"][j])
    a.set(yscale="log", xticks=range(5), ylabel=r"joint return error $\|q_f-q_0\|$ (rad)")
    a.set_xticklabels(NAMES.values(), fontsize=6)
    a.legend(fontsize=6)
    save(fig, "fig3_jointdrift")

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    maps = [vmax_map(mujoco.MjModel.from_xml_path(str(MODELS / f"ff_{s}.xml")), Q_REST) for s in SIZES]
    lo, hi = min(z.min() for *_, z in maps), max(z.max() for *_, z in maps)
    for a, size, (q2, q4, z) in zip(ax, SIZES, maps):
        im = a.pcolormesh(q2, q4, z, vmin=lo, vmax=hi, cmap="viridis", shading="auto")
        a.grid(False)
        a.set(title=f"{size}: {z.min():.2f}-{z.max():.2f} m/s", xlabel="shoulder $q_2$ (rad)")
    ax[0].set_ylabel("elbow $q_4$ (rad)")
    fig.colorbar(im, ax=ax, label=r"$v_{max}$ (m/s)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig5_vmax.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_vmax.png", dpi=200, bbox_inches="tight")
    for size, (_, _, z) in zip(SIZES, maps):
        print(f"v_max {size}: {z.min():.3f} to {z.max():.3f} m/s")


if __name__ == "__main__":
    main()
