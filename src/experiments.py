"""Scripted (no-human) experiments: gain tuning, the Week 7 gate and the beta-sweep. Owner: P3.

Every run is a full closed-loop ORU task driven by the scripted reference (src.task), through
the same Teleop control loop the participants use (speed limiter and haptic signals included),
and writes the same per-tick CSV log. analysis/metrics.py turns logs into metrics, so scripted
runs and participant sessions go through one pipeline.

    python -m src.experiments tune      # per-scheme gain grid, medium, 3-D + 6-D -> data/results/tuning.csv
    python -m src.experiments gate      # 5 schemes x 2 modes x 3 servicers       -> data/results/gate.csv
    python -m src.experiments sweep     # 6-D beta-sweep, schemes 4-5 x 3 servicers -> data/results/sweep.csv
    python -m src.experiments all       # the three, in that order

Logs go to data/scripted/logs/<kind>/ (git-ignored, ~3 MB each; rebuilt by the commands above).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "scripted" / "logs"
RESULTS = ROOT / "data" / "results"
SIZES = ("small", "medium", "large")
MODES = ("3d", "6d")

# Gain grids for tuning (plan, recess: "tune gains per scheme, no strawmen"). Each scheme's
# own knob; the selection rule is fixed before looking (see pick_gains).
TUNE_GRID = {
    "dls":           ("lam", [0.005, 0.01, 0.02, 0.05]),
    "grad_proj":     ("kg", [1.0, 2.0, 5.0, 10.0]),
    "impedance":     ("kp", [0.5, 1.0, 2.0, 4.0]),
    "rns":           ("ka", [1.0, 2.0, 5.0, 10.0]),
    "impedance_rns": ("kp", [0.5, 1.0, 2.0, 4.0]),
}
BETAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def run(size, mode, scheme, gains=None, loops=2, log_path=None, timeout_factor=1.5):
    """One scripted closed-loop task. Returns the log path (metrics come from the log)."""
    from scipy.spatial.transform import Rotation
    from src.schemes import make_scheme
    from src.task import ClosedLoopTask
    from src.teleop import Teleop, site_pose

    meta = {"kind": "scripted", "scheme": scheme, "loops": loops,
            **{f"gain_{k}": v for k, v in (gains or {}).items()}}
    tele = Teleop(size, mode, "hold", log_path=log_path, headless=True, meta=meta)
    tele.controller = make_scheme(scheme, **(gains or {}))
    tele.controller_name = scheme
    p, R = site_pose(tele.model, tele.data, "end_effector")
    task = ClosedLoopTask(p, Rotation.from_matrix(R), mode, loops)
    tele.task = task
    T, ref = task.reference()
    while not task.done and tele.data.time < timeout_factor * T:
        pos, rot, twist = ref(tele.data.time)
        if mode == "3d":
            twist[3:] = 0.0
        tele.target_pos, tele.target_rot = pos, rot
        tele.control_tick(twist)
    tele.close()
    return log_path


def _job(cfg):
    kind, size, mode, scheme, gains, loops = cfg
    tag = "_".join([size, mode, scheme] + [f"{k}{v:g}" for k, v in sorted((gains or {}).items())])
    path = LOGS / kind / f"{tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    run(size, mode, scheme, gains, loops, path)
    sys.path.insert(0, str(ROOT))
    from analysis.metrics import trial_metrics
    m = trial_metrics(path, loops=loops)
    return {"kind": kind, "size": size, "mode": mode, "scheme": scheme,
            **{f"gain_{k}": v for k, v in (gains or {}).items()}, **m,
            "wall_s": round(time.time() - t, 1), "log": str(path.relative_to(ROOT))}


def run_all(cfgs, procs=None, out=None):
    import pandas as pd
    procs = procs or max(1, (os.cpu_count() or 2))
    rows = []
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pl:
            for r in pl.imap_unordered(_job, cfgs):
                rows.append(r)
                print(_line(r), flush=True)
    else:
        for c in cfgs:
            rows.append(_job(c))
            print(_line(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values([c for c in ("size", "mode", "scheme") if c in df]).to_csv(out, index=False)
        print("saved", out.relative_to(ROOT))
    return df


def _line(r):
    gains = " ".join(f"{k[5:]}={v:g}" for k, v in r.items() if k.startswith("gain_"))
    return (f"{r['kind']:5s} {r['size']:6s} {r['mode']} {r['scheme']:14s} {gains:12s} "
            f"done={'yes' if r['completed'] else 'NO ':3s} rms={r['rms_pos_err_mm']:6.1f} mm "
            f"peak_ant={r['peak_antenna_deg']:6.3f} final_ant={r['final_antenna_deg']:6.3f} "
            f"|dq|={r['joint_return_rad']:.3f} limited={100 * r['speed_limited_frac']:.1f}%")


# --- the three experiments ---------------------------------------------------------------

def tune(procs=None):
    cfgs = [("tune", "medium", mode, s, {knob: v}, 2)
            for s, (knob, values) in TUNE_GRID.items() for v in values for mode in MODES]
    df = run_all(cfgs, procs, RESULTS / "tuning.csv")
    print("\nchosen gains:", pick_gains(df))
    return df


def pick_gains(df):
    """Selection rule (fixed before the runs): per scheme, among gains that complete the task in
    BOTH modes, the lowest summed rms hand error; values within 10 % of the best are tied and the
    tie goes to the lower peak base drift. Returns {scheme: {knob: value}}."""
    out = {}
    for s, (knob, _) in TUNE_GRID.items():
        g = df[df.scheme == s]
        by = g.groupby(f"gain_{knob}").agg(done=("completed", "all"), err=("rms_pos_err_mm", "sum"),
                                            drift=("peak_antenna_deg", "sum"))
        ok = by[by.done] if by.done.any() else by
        tied = ok[ok.err <= 1.1 * ok.err.min()]
        out[s] = {knob: float(tied.drift.idxmin())}
    return out


def gate(procs=None):
    from src.schemes import NAMES
    cfgs = [("gate", size, mode, s, None, 2) for size in SIZES for mode in MODES for s in NAMES]
    df = run_all(cfgs, procs, RESULTS / "gate.csv")
    print(f"\n{int(df.completed.sum())}/{len(df)} completed; medium {int(df[df['size'] == 'medium'].completed.sum())}/10")
    return df


def sweep(procs=None):
    from src.schemes import NAMES
    cfgs = [("sweep", size, "6d", s, {"beta": b}, 2) for size in SIZES for s in ("rns", "impedance_rns") for b in BETAS]
    # schemes 1-3 have no beta: their 6-D gate runs are the single points on the same axes
    cfgs += [("sweep", size, "6d", s, None, 2) for size in SIZES for s in NAMES[:3]]
    return run_all(cfgs, procs, RESULTS / "sweep.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["tune", "gate", "sweep", "all"])
    ap.add_argument("--procs", type=int, help="parallel runs (default: all cores)")
    a = ap.parse_args()
    t = time.time()
    if a.what in ("tune", "all"):
        tune(a.procs)
    if a.what in ("gate", "all"):
        gate(a.procs)
    if a.what in ("sweep", "all"):
        sweep(a.procs)
    print(f"wall {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
