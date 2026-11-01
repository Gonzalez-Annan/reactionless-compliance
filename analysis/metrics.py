"""Per-trial metrics from the teleop CSV logs, and the study statistics. Owner: P5.

First version by P3 so the pipeline runs end to end (plan, recess gate: "pilot data flows
end-to-end through figures.py with zero manual steps"). P5 owns the definitions.

    from analysis.metrics import trial_metrics, collect, friedman, pairwise_wilcoxon
    df = collect("data/participants")          # one row per trial log
    python -m analysis.metrics data/participants      # print the table and the tests

Works on participant logs (src.session) and scripted logs (src.experiments) alike.
Questionnaires: data/participants/<ID>/questionnaire.csv, columns as in
data/questionnaire_template.csv (predictability 1-7, raw NASA-TLX six subscales 0-100).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANTENNA_LIMIT_DEG = 5.0
TLX = ["tlx_mental", "tlx_physical", "tlx_temporal", "tlx_performance", "tlx_effort", "tlx_frustration"]


def read_log(path):
    """(meta dict from the '# key: value' header, DataFrame)."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            k, _, v = line[1:].partition(":")
            meta[k.strip()] = v.strip()
    return meta, pd.read_csv(path, comment="#")


def _cols(df, name, n):
    return df[[f"{name}_{i}" for i in range(n)]].to_numpy()


def _quat_angle_deg(q1, q2):
    dot = np.abs(np.sum(q1 * q2, axis=1)) / (np.linalg.norm(q1, axis=1) * np.linalg.norm(q2, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dot, -1, 1)))


def trial_metrics(path, loops=None):
    """One dict of metrics for one trial log. Base drift = antenna pointing error (R1)."""
    meta, df = read_log(path)
    loops = int(loops or meta.get("loops", 2))
    t = df["t_sim"].to_numpy()
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 0.01))
    ee, tgt = _cols(df, "ee_pos", 3), _cols(df, "target_pos", 3)
    q = _cols(df, "qpos", 14)
    ant = df["antenna_err_deg"].to_numpy()
    speed = np.r_[0.0, np.linalg.norm(np.diff(ee, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)]
    m = {
        "completed": bool(df["task_loop"].iloc[-1] >= loops) if "task_loop" in df else False,
        "duration_s": float(t[-1]),
        "rms_pos_err_mm": 1000 * float(np.sqrt(np.mean(np.sum((ee - tgt) ** 2, axis=1)))),
        "rms_rot_err_deg": float(np.sqrt(np.mean(_quat_angle_deg(_cols(df, "ee_quat", 4), _cols(df, "target_quat", 4)) ** 2))),
        "peak_antenna_deg": float(ant.max()),
        "final_antenna_deg": float(ant[-1]),
        "over_limit_s": float(dt[ant > ANTENNA_LIMIT_DEG].sum()),
        "joint_return_rad": float(np.linalg.norm(q[-1, 7:] - q[0, 7:])),
        "base_pos_drift_m": float(np.linalg.norm(q[-1, :3] - q[0, :3])),
        "path_length_m": float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()),
        "peak_hand_speed": float(speed.max()),
        "peak_qdot_cmd": float(np.abs(_cols(df, "qdot_cmd", 7)).max()),
    }
    # columns added later (haptics, limiter, camera) -- absent in older logs
    m["contact_frac"] = float((df["contact_force"] > 0).mean()) if "contact_force" in df else np.nan
    m["peak_contact_N"] = float(df["contact_force"].max()) if "contact_force" in df else np.nan
    m["min_joint_margin_deg"] = float(df["joint_margin_deg"].min()) if "joint_margin_deg" in df else np.nan
    m["speed_limited_frac"] = float((df["speed_scale"] < 1).mean()) if "speed_scale" in df else 0.0
    if "v_max_design" in df and df["v_max_design"].notna().any():
        ok = df["v_max_design"].notna().to_numpy()
        m["speed_over_vmax"] = float(np.max(speed[ok] / df["v_max_design"].to_numpy()[ok]))   # R2: must be <= 1
    else:
        m["speed_over_vmax"] = np.nan
    if "cam_0" in df and df["cam_0"].notna().any():
        az = np.unwrap(np.radians(df["cam_0"].to_numpy()))
        m["camera_turn_deg"] = float(np.degrees(np.abs(np.diff(az)).sum()) + np.abs(np.diff(df["cam_1"])).sum())
    else:
        m["camera_turn_deg"] = np.nan
    for k in ("participant", "condition", "scheme", "mode", "model", "pad_layout"):
        if k in meta:
            m[k] = meta[k]
    return m


def collect(root, pattern="*.csv"):
    """Metrics for every trial log under root (questionnaire / order files are skipped)."""
    rows = []
    for p in sorted(Path(root).rglob(pattern)):
        if p.name in ("order.csv", "questionnaire.csv") or p.name.endswith("_summary.csv"):
            continue
        try:
            rows.append({**trial_metrics(p), "log": str(p)})
        except (KeyError, ValueError, IndexError):
            continue          # not a trial log
    return pd.DataFrame(rows)


def questionnaires(root):
    """All data/participants/<ID>/questionnaire.csv files stacked, with raw TLX added."""
    files = sorted(Path(root).rglob("questionnaire.csv"))
    if not files:
        return pd.DataFrame()
    q = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    q["tlx_raw"] = q[TLX].mean(axis=1)
    return q


def synthetic_questionnaires(n=10, seed=0):
    """FAKE ratings for testing the figures before real data exist (plan W7: 'all figures
    render from fake data'). Never mix with real data; figures mark them SYNTHETIC."""
    from src.session import condition_order
    rng = np.random.default_rng(seed)
    effect = {"dls": 0.0, "grad_proj": -0.5, "impedance": 0.3, "rns": -0.3, "impedance_rns": 0.4}
    rows = []
    for k in range(1, n + 1):
        pid = f"P{k:02d}"
        for c, (mode, scheme) in enumerate(condition_order(pid), 1):
            base = effect[scheme] - (0.4 if mode == "6d" else 0.0)
            row = {"participant": pid, "condition": c, "mode": mode, "scheme": scheme,
                   "predictability": int(np.clip(np.round(5 + base + rng.normal(0, 0.8)), 1, 7))}
            for s in TLX:
                row[s] = int(np.clip(45 - 10 * base + rng.normal(0, 12), 0, 100))
            rows.append(row)
    q = pd.DataFrame(rows)
    q["tlx_raw"] = q[TLX].mean(axis=1)
    return q


# --- statistics (plan, Week 10) ---------------------------------------------------------

def friedman(df, value, subject="participant", condition="scheme"):
    """Friedman test across conditions + Kendall's W. df: one row per subject x condition
    (average first if there are repeats). Returns dict(chi2, p, W, n, k)."""
    from scipy.stats import friedmanchisquare
    wide = df.pivot_table(index=subject, columns=condition, values=value, aggfunc="mean").dropna()
    n, k = wide.shape
    chi2, p = friedmanchisquare(*[wide[c] for c in wide.columns])
    return {"chi2": float(chi2), "p": float(p), "W": float(chi2 / (n * (k - 1))), "n": n, "k": k}


def pairwise_wilcoxon(df, value, subject="participant", condition="scheme"):
    """All pairs, Wilcoxon signed-rank, Holm-corrected, effect size r = Z / sqrt(n)."""
    from itertools import combinations
    from scipy.stats import norm, wilcoxon
    wide = df.pivot_table(index=subject, columns=condition, values=value, aggfunc="mean").dropna()
    rows = []
    for a, b in combinations(wide.columns, 2):
        diff = wide[a] - wide[b]
        if (diff == 0).all():
            p = 1.0
        else:
            p = float(wilcoxon(wide[a], wide[b], zero_method="wilcox").pvalue)
        z = norm.isf(p / 2) if p < 1 else 0.0
        rows.append({"a": a, "b": b, "median_diff": float(diff.median()), "p": p,
                     "r": float(z / np.sqrt(len(diff)))})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(out)
    adj, running = [], 0.0
    for i, p in enumerate(out["p"]):           # Holm step-down
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    out["p_holm"] = adj
    return out


def main():
    ap = argparse.ArgumentParser(description="Trial metrics and statistics")
    ap.add_argument("root", nargs="?", default="data/participants")
    ap.add_argument("--out", help="CSV for the per-trial table")
    a = ap.parse_args()
    df = collect(a.root)
    if df.empty:
        print(f"no trial logs under {a.root}")
        return
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df.drop(columns=["log"]).round(3).to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
    if "participant" in df and df["participant"].nunique() >= 2:
        for mode, g in df.groupby("mode"):
            for v in ("rms_pos_err_mm", "peak_antenna_deg", "joint_return_rad", "duration_s"):
                print(mode, v, friedman(g, v))
    q = questionnaires(a.root)
    if not q.empty:
        for v in ("predictability", "tlx_raw"):
            print(v, friedman(q, v))
            print(pairwise_wilcoxon(q, v).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
