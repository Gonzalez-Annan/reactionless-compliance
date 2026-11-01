"""All paper figures, from data/ with no manual steps. Owner: P5 (first version by P3).

    python -m analysis.figures                 # every figure that has data -> figures/
    python -m analysis.figures --synthetic     # Fig. 4 from fake ratings (marked SYNTHETIC)

Plan, Week 10:
  Fig. 1  trade-off (6-D): hand error vs base drift; beta curves for schemes 4-5, points 1-3
          (+ participant trials when data/participants has logs)      <- data/results/sweep.csv
  Fig. 2  base drift vs servicer size, where it crosses the +/-5 deg limit <- data/results/gate.csv
  Fig. 3  joint-configuration return error ||q_f - q_0||, both modes     <- data/results/gate.csv
  Fig. 4  predictability and raw NASA-TLX per scheme                     <- questionnaires
  Fig. 5  v_max over arm configuration, per servicer                     <- data/safety/vmax_*.csv
  Fig. S1 safety Monte Carlo + impact validation                         <- data/safety/*.csv
All figures 3.5 in wide (one IEEE column) or 7.16 in (two), vector PDF + PNG.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"
SCHEMES = ["dls", "grad_proj", "impedance", "rns", "impedance_rns"]
LABEL = {"dls": "1 DLS", "grad_proj": "2 Grad. proj.", "impedance": "3 Impedance",
         "rns": "4 RNS", "impedance_rns": "5 Imp.+RNS"}
COLOR = dict(zip(SCHEMES, ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]))
MARKER = dict(zip(SCHEMES, ["o", "s", "^", "D", "v"]))
SIZES = ["small", "medium", "large"]
RATIO = {"small": 6.49 / 21.7, "medium": 6.49 / 151.7, "large": 6.49 / 1501.7}   # Gen3 moving arm / servicer (incl. 1.7 kg arm base)
LIMIT = 5.0

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _plain_log(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    return OUT / f"{name}.pdf"


def _limit_line(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(LIMIT, color="r", lw=0.8, ls="--")


def fig1_tradeoff(sweep, human=None):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    for ax, size in zip(axs, SIZES):
        g = sweep[(sweep["size"] == size) & (sweep["mode"] == "6d")]
        for s in SCHEMES:
            h = g[g.scheme == s]
            if h.empty:
                continue
            if "gain_beta" in h and h["gain_beta"].notna().any():
                h = h.dropna(subset=["gain_beta"]).sort_values("gain_beta")
                ax.plot(h.peak_antenna_deg, h.rms_pos_err_mm, "-", marker=MARKER[s], ms=3, color=COLOR[s], label=LABEL[s])
                nd = h[~h.completed.astype(bool)]
                ax.plot(nd.peak_antenna_deg, nd.rms_pos_err_mm, "x", color="k", ms=4)
                for _, r in h.iloc[[0, -1]].iterrows():
                    ax.annotate(f"β={r.gain_beta:g}", (r.peak_antenna_deg, r.rms_pos_err_mm), fontsize=5,
                                xytext=(2, 2), textcoords="offset points", color=COLOR[s])
            else:
                ax.plot(h.peak_antenna_deg, h.rms_pos_err_mm, MARKER[s], ms=4, color=COLOR[s], label=LABEL[s])
        if human is not None and not human.empty:
            hh = human[(human["mode"] == "6d")]
            for s in SCHEMES:
                x = hh[hh.scheme == s]
                ax.plot(x.peak_antenna_deg, x.rms_pos_err_mm, MARKER[s], mfc="none", ms=3, color=COLOR[s], alpha=0.6)
        if g.peak_antenna_deg.max() > LIMIT * 0.6:
            _limit_line(ax)
        xs = g.peak_antenna_deg
        if xs.max() / max(xs.min(), 1e-9) > 4:     # log only when drift spans a wide range
            ax.set_xscale("log")
            _plain_log(ax)
        ax.set_title(f"ff_{size} (arm/base {RATIO[size]:.3g})")
        ax.set_xlabel("peak base attitude drift [deg]")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("rms hand error [mm]")
    axs[2].legend(loc="upper right", frameon=False)
    fig.suptitle("Fig. 1  6-D task: task accuracy vs base drift (scripted 2-loop ORU task; x = task not finished)",
                 fontsize=7, y=1.02)
    return save(fig, "fig1_tradeoff")


def fig2_size(gate):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
    for ax, mode in zip(axs, ("3d", "6d")):
        g = gate[gate["mode"] == mode]
        for s in SCHEMES:
            h = g[g.scheme == s].set_index("size").reindex(SIZES)
            x = [RATIO[z] for z in SIZES]
            ax.plot(x, h.peak_antenna_deg.clip(lower=1e-3), "-", marker=MARKER[s], ms=3, color=COLOR[s], label=LABEL[s])
            for xi, (_, r) in zip(x, h.iterrows()):
                if not r.completed:
                    ax.plot(xi, max(r.peak_antenna_deg, 1e-3), "x", color="k", ms=5)
        _limit_line(ax, "y")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("arm / base mass ratio")
        ax.set_title(f"{mode.upper()} task")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("peak base attitude drift [deg]")
    axs[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Fig. 2  Base drift vs servicer size (x = task not finished; red = ±5° antenna limit)", fontsize=7, y=1.02)
    return save(fig, "fig2_size")


def fig3_joint_return(gate):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    w = 0.38
    for ax, size in zip(axs, SIZES):
        g = gate[gate["size"] == size]
        for j, mode in enumerate(("3d", "6d")):
            h = g[g["mode"] == mode].set_index("scheme").reindex(SCHEMES)
            ax.bar(np.arange(5) + (j - 0.5) * w, h.joint_return_rad, w, color=[COLOR[s] for s in SCHEMES],
                   alpha=0.55 if mode == "3d" else 1.0, hatch="//" if mode == "3d" else None, edgecolor="k", lw=0.3,
                   label=mode.upper())
        ax.set_xticks(range(5), [LABEL[s].split(" ", 1)[0] for s in SCHEMES])
        ax.set_title(f"ff_{size}")
        ax.set_xlabel("scheme")
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel(r"$\|q_f - q_0\|$ [rad]")
    axs[0].legend(frameon=False)
    fig.suptitle("Fig. 3  Joint-configuration return error after 2 loops (hatched 3-D, solid 6-D)", fontsize=7, y=1.03)
    return save(fig, "fig3_joint_return")


def fig4_subjective(q, synthetic=False):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.0))
    for ax, v, lab in zip(axs, ("predictability", "tlx_raw"), ("predictability (1-7, higher better)", "raw NASA-TLX (0-100, lower better)")):
        data = [q[q.scheme == s][v] for s in SCHEMES]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], SCHEMES):
            patch.set_facecolor(COLOR[s])
            patch.set_alpha(0.7)
        n = q.participant.nunique()
        if n < 8:   # plan: below 8 participants show paired per-participant lines instead of tests
            for pid, g in q.groupby("participant"):
                y = g.groupby("scheme")[v].mean().reindex(SCHEMES)
                ax.plot(range(1, 6), y, "-", color="k", alpha=0.25, lw=0.5)
        ax.set_xticks(range(1, 6), [LABEL[s] for s in SCHEMES], rotation=20)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
    title = f"Fig. 4  Operator ratings per scheme (n = {q.participant.nunique()})"
    if synthetic:
        title += "  —  SYNTHETIC DATA, NOT RESULTS"
        for ax in axs:
            ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center", fontsize=20,
                    color="r", alpha=0.25, rotation=20)
    fig.suptitle(title, fontsize=7, y=1.02)
    return save(fig, "fig4_subjective")


def fig5_vmax():
    import mujoco
    from matplotlib.colors import LogNorm
    rows = {"rigid joints": "vmax_rigid_{}.csv", "real servos (±5 Nm)": "vmax_{}.csv"}
    V = {(r, s): DATA / "safety" / f.format(s) for r, f in rows.items() for s in SIZES}
    if not all(p.exists() for p in V.values()):
        return None
    V = {k: np.loadtxt(p, delimiter=",") for k, p in V.items()}
    lo = min(np.nanmin(v) for v in V.values())
    hi = max(np.nanmax(v) for v in V.values())
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "ff_medium.xml"))
    r2, r4 = np.degrees(m.jnt_range[2]), np.degrees(m.jnt_range[4])
    hq = np.degrees(m.key("home").qpos)
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True, sharey=True)
    for i, r in enumerate(rows):
        for j, s in enumerate(SIZES):
            ax = axs[i, j]
            v = V[(r, s)]
            im = ax.imshow(v, origin="lower", aspect="auto", cmap="viridis", norm=LogNorm(lo, hi),
                           extent=[r2[0], r2[1], r4[0], r4[1]])
            ax.plot(hq[8], hq[10], "w+", ms=6)
            ax.set_title(f"ff_{s}, {r}\n{np.nanmin(v):.3f} to {np.nanmax(v):.3f} m/s", fontsize=5.5)
            if i == 1:
                ax.set_xlabel("shoulder j2 [deg]")
        axs[i, 0].set_ylabel("elbow j4 [deg]")
    fig.subplots_adjust(hspace=0.45)
    fig.colorbar(im, ax=axs, label="v_max [m/s] (log)", shrink=0.9, pad=0.01)
    fig.suptitle("Fig. 5  Contact-speed limit over arm configuration (contact along the tool axis, worst ISO/TS 15066 "
                 "region and restraint; white = arm in the base; + = home)", fontsize=7, y=1.0)
    return save(fig, "fig5_vmax")


def figS1_safety():
    mc, imp = DATA / "safety" / "montecarlo.csv", DATA / "safety" / "impact.csv"
    if not (mc.exists() and imp.exists()):
        return None
    from src.safety import GOVERNOR_MARGIN
    mc, imp = pd.read_csv(mc), pd.read_csv(imp)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    ax = axs[0]
    pos, ticks, labels = 0, [], []
    for size in SIZES:
        for p in sorted(mc.payload_kg.unique()):
            g = mc[(mc["size"] == size) & (mc.payload_kg == p)]
            for r, c in (("free", "#4c72b0"), ("clamped", "#c44e52")):
                ax.boxplot([g[f"v_max_{r}"]], positions=[pos], widths=0.35, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=c, alpha=0.6), medianprops=dict(color="k"))
                pos += 0.4
            ticks.append(pos - 0.6)
            labels.append(f"{size[0].upper()} {p:g}kg")
            pos += 0.5
    ax.axhline(mc.v_cap.iloc[0], color="k", ls="--", lw=0.8)
    ax.text(0, mc.v_cap.iloc[0] * 1.05, "teleop speed cap", fontsize=5)
    ax.set_xticks(ticks, labels, rotation=30)
    ax.set_ylabel("design v_max [m/s]")
    ax.set_yscale("log")
    ax.set_title("Monte Carlo: pose × direction × payload (blue free, red clamped)")
    ax.grid(alpha=0.3, axis="y")
    ax = axs[1]
    for (reg, res), g in imp.groupby(["region", "restraint"]):
        ax.scatter(g.axial, g.F_at_gov / g.F_limit, s=6, label=f"{reg.replace('_', ' ')}/{res}")
    ax.axhline(1.0, color="r", ls="--", lw=0.8)
    ax.text(0.72, 1.025, "ISO/TS 15066 limit", color="r", fontsize=5)
    ax.set_xlabel("|cos| between push and arm axis")
    ax.set_ylabel("peak force at limiter speed / ISO limit")
    ax.set_ylim(0, 1.3)
    ax.set_title(f"Impacts re-run at the limiter speed (n = {int(imp.F_at_gov.notna().sum())})")
    ax.legend(frameon=False, fontsize=5, loc="upper left", ncol=3)
    ax.grid(alpha=0.3)
    fig.suptitle("Fig. S1  Contact safety: sampled limits and the simulated-impact check", fontsize=7, y=1.02)
    return save(fig, "figS1_safety")


def main():
    ap = argparse.ArgumentParser(description="Render every figure that has data")
    ap.add_argument("--synthetic", action="store_true", help="Fig. 4 from fake ratings if no real ones")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from analysis.metrics import collect, questionnaires, synthetic_questionnaires
    made = []
    human = collect(DATA / "participants") if (DATA / "participants").exists() else pd.DataFrame()
    sweep, gate = DATA / "results" / "sweep.csv", DATA / "results" / "gate.csv"
    if sweep.exists():
        made.append(fig1_tradeoff(pd.read_csv(sweep), human if not human.empty else None))
    if gate.exists():
        g = pd.read_csv(gate)
        made += [fig2_size(g), fig3_joint_return(g)]
    q = questionnaires(DATA / "participants")
    if not q.empty:
        made.append(fig4_subjective(q))
    elif a.synthetic:
        made.append(fig4_subjective(synthetic_questionnaires(), synthetic=True))
    made += [fig5_vmax(), figS1_safety()]
    for p in made:
        if p:
            print("saved", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
