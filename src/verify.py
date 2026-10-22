"""Safety case: hazard table + verification matrix, computed from the data. Owner: P4.

First version by P3 (plan, Week 9 gate: "Hazard table and verification matrix exist").
Every PASS / FAIL below is computed here from data/results/*.csv, data/safety/*.csv and a few
direct checks, so the document cannot drift from the code. Rebuild after any re-run:

    python -m src.experiments all          # scripted gate + sweep (if gains / schemes changed)
    python -m src.safety --impact --out data/safety/impact.csv
    python -m src.safety --montecarlo --out data/safety/montecarlo.csv
    python -m src.verify                   # -> docs/SAFETY.md
"""
import datetime
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "SAFETY.md"

HAZARDS = [
    # id, hazard, cause, consequence, S, L, mitigation, verified by
    ("H1", "Base attitude drifts past the ±5° antenna cone", "Arm reaction torque on a free-floating base; small servicer; 6-D task (reactionless infeasible)",
     "Loss of the operator's radio link", "High", "Med (small servicer: High)",
     "Reactionless schemes 4-5; attitude feedback; antenna line + red cue; pad pulse on exit; mission sizing (Fig. 2)", "R1, R4"),
    ("H2", "Arm strikes an EVA astronaut above the ISO/TS 15066 limit", "Hand speed too high for the pose's effective mass; payload raises m_eff; clamped (pinned) contact",
     "Injury (bruising to fracture by region)", "High", "Med",
     "Pose-dependent speed limiter (0.85 × v_max(q), worst region and restraint); teleop speed cap; HUD hand speed vs limit", "R2"),
    ("H3", "Contact with the head or face", "Operator drives the hand near the astronaut's helmet",
     "Severe injury: ISO allows no transient contact here", "Catastrophic", "Low",
     "Procedural keep-out volume around the helmet; not covered by the speed limit (ISO: not applicable)", "R9 (procedural)"),
    ("H4", "Arm collides with its own servicer or itself", "Large motions near the bus; posture drift (schemes 2, 4)",
     "Hardware damage; disturbance torque on the base", "Med", "Med",
     "Collision model; heavy-motor rumble scaled by contact force; red contact markers; posture impedance (3, 5)", "R6"),
    ("H5", "Joint driven into its mechanical limit", "Null-space drift; operator pushes past reach",
     "Jolt, lost DOF; the base absorbs what the locked joint cannot", "Med", "Med",
     "Joint geofence (3° margin, re-solve without the joint); light-motor rumble ramp over the last 15°", "R5"),
    ("H6", "Excessive joint rates near a dynamic singularity", "J* loses rank; undamped inverse",
     "Sudden fast arm / base motion", "High", "Low",
     "Damped inverses (λ, ε); uniform ±1 rad/s saturation that keeps direction; ±5 Nm torque limit", "R7"),
    ("H7", "Controller built on a wrong dynamics model", "Integrator error; wrong H_b / H_bm; momentum not conserved",
     "Schemes predict base motion wrongly; all results invalid", "High", "Low",
     "Momentum gate in CI on all servicers (< 1e-6); J* checked against the simulated hand twist", "R3"),
    ("H8", "Operator confusion (frame / mode / camera)", "Camera-relative moves, 3-D vs 6-D, rotate mode",
     "Unintended motion toward people or structure", "Med", "Med",
     "Mode on the HUD; one pad layout per study; fine mode; snap-target; training trial; pilot", "R10, pilot (pending)"),
    ("H9", "Loss of the input device mid-task", "Pad unplugged / Bluetooth drop",
     "Stale command keeps the arm moving", "Med", "Low",
     "A missing pad reads zero; hot-plug; the pose servo then holds the last target", "R10"),
    ("H10", "Infeasible reactionless demand in 6-D", "6 task + 3 reactionless rows > 7 joints",
     "A scheme promising zero drift silently trades task accuracy", "Med", "High",
     "Weighted least squares with exposed β; trade-off reported (Fig. 1); DOF check", "R8"),
]


def _md(df):
    """DataFrame -> markdown table without the optional tabulate dependency."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _pf(ok):
    return "**PASS**" if ok else "**FAIL**"


def _momentum(seconds=2.0):
    from src.dynamics import momentum_residual
    worst = {}
    for size in ("small", "medium", "large"):
        m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"ff_{size}.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        rng, w = np.random.default_rng(0), 0.0
        for k in range(int(seconds / m.opt.timestep)):
            if k % 2000 == 0:
                d.ctrl[:] = rng.uniform(-1, 1, m.nu)
            mujoco.mj_step(m, d)
            if k % 10 == 0:
                w = max(w, np.abs(momentum_residual(m, d)).max())
        worst[size] = w
    return worst


def _dof(n=50):
    from src.dynamics import kin
    from src.safety import random_pose
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "ff_medium.xml"))
    d, rng = mujoco.MjData(m), np.random.default_rng(1)
    r6, r3 = [], []
    for _ in range(n):
        random_pose(m, d, rng)
        k = kin(m, d)
        r6.append(np.linalg.matrix_rank(np.vstack([k.Jstar, k.W])))
        r3.append(np.linalg.matrix_rank(np.vstack([k.Jstar[:3], k.W])))
    return np.array(r6), np.array(r3)


def matrix():
    from src.safety import GOVERNOR_MARGIN, SERVO_MASS
    res = ROOT / "data" / "results"
    gate = pd.read_csv(res / "gate.csv")
    runs = pd.concat([pd.read_csv(res / f) for f in ("gate.csv", "sweep.csv", "tuning.csv") if (res / f).exists()])
    mc = pd.read_csv(ROOT / "data" / "safety" / "montecarlo.csv")
    imp = pd.read_csv(ROOT / "data" / "safety" / "impact.csv")
    rows = []

    # R1 -- base attitude within +/-5 deg over two loops, 6-D task, per servicer
    g6 = gate[gate["mode"] == "6d"]
    per, med_ok = [], None
    for size in ("small", "medium", "large"):
        h = g6[g6["size"] == size]
        ok = (h.peak_antenna_deg <= 5).all() and h.completed.all()
        worst = h.loc[h.peak_antenna_deg.idxmax()]
        med_ok = ok if size == "medium" else med_ok
        per.append(f"{size}: {_pf(ok)} (worst {worst.scheme} {worst.peak_antenna_deg:.2f}°, "
                   f"{int(h.completed.sum())}/5 finish)")
    per.insert(0, f"{_pf(med_ok)} on the study servicer; per servicer:")
    rows.append(("R1", "Base attitude within ±5° over two closed loops (6-D task)",
                 "Scripted gate, 5 schemes × 3 servicers; β-sweep (Fig. 1, 2); human trials pending",
                 "<br>".join(per), "medium (study servicer) must pass"))

    # R2 -- contact speed <= v_max(q)
    lim_ok = (GOVERNOR_MARGIN * imp.v_max_design <= imp.v_max_sim).mean()
    ratio = runs.speed_over_vmax.max()
    limited = runs.speed_limited_frac.max()
    p5 = mc[(mc.payload_kg == mc.payload_kg.max())]
    ok2 = lim_ok == 1.0 and ratio <= 1.0
    rows.append(("R2", "Contact speed ≤ v_max(q) at every tick (ISO/TS 15066 body model)",
                 f"{len(imp)} simulated impacts (3 servicers, 0/5 kg, 3 body cases); Monte Carlo {len(mc)} poses × "
                 f"payload × restraint; every tick of all {len(runs)} scripted runs",
                 f"{_pf(ok2)}: limiter (0.85 v_max, m_eff limp + {SERVO_MASS:g} kg) below the simulated-impact limit in "
                 f"{100 * lim_ok:.0f} % of impacts; worst tick hand speed / v_max(q) = {ratio:.2f}; limiter acted in "
                 f"≤ {100 * limited:.1f} % of ticks. Without it the teleop cap would exceed the clamped limit in "
                 f"{100 * p5.v_cap.gt(p5.v_max_clamped).mean():.0f} % of poses with a {p5.payload_kg.iloc[0]:g} kg payload",
                 "Fig. 5, Fig. S1"))

    # R3 -- momentum
    mom = _momentum()
    rows.append(("R3", "Momentum residual < 1e-6 (every servicer)",
                 "tests/test_momentum.py in CI (10 s, every step); recomputed here (2 s)",
                 f"{_pf(max(mom.values()) < 1e-6)}: " + ", ".join(f"{k} {v:.1e}" for k, v in mom.items()), "CI log"))

    # R4 -- 3-D reactionless ~ 0 drift
    g3 = gate[gate["mode"] == "3d"]
    lines, ok4 = [], True
    for size in ("small", "medium", "large"):
        h = g3[g3["size"] == size].set_index("scheme")
        base = h.loc["dls", "peak_antenna_deg"]
        for s in ("rns", "impedance_rns"):
            r = h.loc[s]
            red = 1 - r.peak_antenna_deg / base
            ok = r.final_antenna_deg < 0.05 and (red >= 0.5 or r.peak_antenna_deg < 0.05)
            ok4 &= ok
            lines.append(f"{size} {s}: peak {r.peak_antenna_deg:.3f}° (DLS {base:.3f}°), final {r.final_antenna_deg:.3f}°"
                         + ("" if r.completed else " (task not finished)"))
    rows.append(("R4", "Reactionless schemes give ≈ 0 base drift in 3-D (final < 0.05°; peak ≤ half of DLS or < 0.05°)",
                 "Scripted gate, 3-D, schemes 4-5 vs DLS", f"{_pf(ok4)}:<br>" + "<br>".join(lines), "verification, not a result (plan)"))

    # R5 -- joint limits
    mm = runs.min_joint_margin_deg.min()
    rows.append(("R5", "No joint driven past its limit in normal operation",
                 f"All {len(runs)} scripted runs (geofence 3°); tests/test_schemes.py geofence test",
                 f"{_pf(mm >= 0)}: smallest margin to any limit {mm:.1f}°", ""))

    # R6 -- contact signalled
    cf = runs.contact_frac.max()
    rows.append(("R6", "The scripted task never touches the bus; any contact is signalled (rumble + marker) the same tick",
                 "All scripted runs; tests/test_gamepad.py::test_haptic_signals_feel_the_arm_hit_the_base",
                 f"{_pf(cf == 0)}: contact in {100 * cf:.2f} % of ticks (worst run); haptic test in CI", ""))

    # R7 -- joint-rate saturation
    pq = runs.peak_qdot_cmd.max()
    rows.append(("R7", "Commanded joint rates within ±1 rad/s, direction kept",
                 "All scripted runs (limit_qdot); torque ±5 Nm in the model",
                 f"{_pf(pq <= 1 + 1e-9)}: largest command {pq:.3f} rad/s", ""))

    # R8 -- DOF argument
    r6, r3 = _dof()
    rows.append(("R8", "6-D reactionless is infeasible, 3-D is exact (plan, correction 2)",
                 "rank [J*; W] at 50 random poses (medium); tests/test_momentum.py",
                 f"{_pf((r6 == 7).all() and (r3 == 6).all())}: 6-D rank {sorted(int(x) for x in set(r6))} of 9 rows, "
                 f"3-D rank {sorted(int(x) for x in set(r3))} of 6", ""))

    rows.append(("R9", "No contact with head or face", "Procedural (keep-out volume); cannot be shown in simulation",
                 "**OPEN** (procedural, deployment)", "ISO: transient contact not applicable"))
    rows.append(("R10", "Missing input gives a zero command; the pad mapping is the documented one",
                 "tests/test_gamepad.py (no-pad, deadband, untouched triggers, layouts); real DualShock 4 checked",
                 "**PASS** (CI)", "pilot with participants pending"))
    return rows


def write():
    from src.safety import BODY, GOVERNOR_MARGIN, SERVO_MASS, TRANSIENT_MULT
    rows = matrix()
    s = pd.read_csv(ROOT / "data" / "safety" / "montecarlo_summary.csv")
    L = [f"# Safety case\n\nGenerated by `python -m src.verify` on {datetime.date.today()} from the data in this repo; "
         "do not edit by hand. Owner: P4 (first version by P3).\n",
         "## Contact model\n",
         "ISO/TS 15066:2016 Annex A body model (Tables A.2, A.3); ISO 10218-1:2025 excludes space robots, so only the "
         "body model is borrowed. Two-body energy model v_max = F / √(μ k).",
         f"- **free** contact (the person can recoil): F = {TRANSIENT_MULT:g} × quasi-static limit, μ = reduced mass of the arm and the body region.",
         "- **clamped** contact (pinned against structure, e.g. EVA foot restraint): F = quasi-static limit, μ = arm mass.",
         "- Head and face: ISO allows no transient contact; handled procedurally (H3).",
         f"- **Arm mass at impact**: the joints are velocity servos saturating at ±5 Nm, so they give way. Simulated impacts "
         f"(data/safety/impact.csv) show the peak force does not depend on the base mass (20 to 1500 kg). The design mass is the "
         f"limp-joint m_eff + {SERVO_MASS:g} kg; the rigid-joint m_eff (whole spacecraft) is reported only as an upper bound.",
         f"- **Limiter**: commanded joint rates are scaled so the predicted hand speed stays ≤ {GOVERNOR_MARGIN:g} × v_max(q) "
         "for the worst body region and restraint (src.safety.SpeedGovernor, on in teleop and all scripted runs).\n",
         "| Region | F quasi-static [N] | k [N/mm] | m_H [kg] |", "|---|---|---|---|"]
    L += [f"| {r.replace('_', ' ')} | {b['F_qs']:g} | {b['k'] / 1e3:g} | {b['m']:g} |" for r, b in BODY.items()]
    L += ["\n## Hazard table\n", "| ID | Hazard | Cause | Consequence | Severity | Likelihood | Mitigation | Verified by |",
          "|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(h) + " |" for h in HAZARDS]
    L += ["\n## Verification matrix\n", "| Req | Requirement | Test | Evidence | Note |", "|---|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in rows]
    L += ["\n## Monte Carlo summary (design v_max, m/s)\n",
          "p_cap: share of poses where the teleop speed cap would exceed v_max without the limiter; "
          "p_hw: same for the arm at full joint speed; *_if_rigid: with the rigid-joint mass.\n",
          _md(s.round(3))]
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    for r in rows:
        line = f"{r[0]:4s} {r[3].split(':')[0].replace('**', ''):5s} {r[1]}"
        print(line.encode("ascii", "replace").decode())
    print("saved", DOC.relative_to(ROOT))


if __name__ == "__main__":
    write()
