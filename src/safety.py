"""Pose-dependent contact-velocity limit for the free-floating arm. Owner: P4.

First version written by P3 so the pipeline runs end to end; P4 owns the numbers and the
hazard table / verification matrix.

Plan v2, Section 3:
    Operational-space inertia   Lambda(q) = (J M^-1 J^T)^-1      (full floating M, 13 DOF)
    Effective mass along u      m_eff = 1 / (u^T Lambda_v^-1 u)  (point contact: linear rows)
    Contact impulse             J_imp = (1 + e) mu v,   mu = reduced mass of the two bodies
    Unrestrained human          dv_h  = J_imp / m_h
    Permissible velocity        v_max(q) = limit / ((1 + e) m_eff(q))

What gives way at impact:
    locked=True   joints held stiff by the servos, only the 6 base DOFs move. Conservative
                  (largest m_eff) -> the default.
    locked=False  every joint back-drivable (no servo stiffness): the lower bound.
Restraint: m_other=None means the struck body is restrained (wall, docked target), so the
reduced mass is m_eff itself. A number means an unrestrained body of that mass.

Two v_max models:
  * v_max_iso: ISO/TS 15066 transient-contact energy model, v = F_max / sqrt(mu k). This is
    the default for the map. ISO 10218-1:2025 excludes space robots; we borrow the body
    model only. BODY values are from ISO/TS 15066 Annex A as recalled -- P4: CHECK THEM
    against the standard before the paper cites any number.
  * v_max_impulse: the plan's impulse form, with the limit (N s) left to P4.

    python -m src.safety --model medium                     # numbers at the home pose
    python -m src.safety --map --out data/vmax_medium.png   # v_max over shoulder x elbow
"""
import argparse

import mujoco
import numpy as np

from src.dynamics import EE_SITE, mass_matrix, site_jacobian

# ISO/TS 15066 Annex A (P4: verify). F_max transient = 2 x quasi-static limit.
BODY = {
    "chest": dict(F_max=280.0, k=25e3, m=40.0),   # N, N/m, kg
    "hand":  dict(F_max=280.0, k=75e3, m=0.6),
}
RESTITUTION = 0.0   # e: plastic impact. P4 to justify (0 for soft tissue, ~0.3-0.5 metal on metal)


def op_space_inertia(model, data, locked=True, site=EE_SITE):
    """6 x 6 Lambda(q) at the hand, world frame, rows [linear; angular]."""
    M, J = mass_matrix(model, data), site_jacobian(model, data, site)
    n = 6 if locked else model.nv
    Linv = J[:, :n] @ np.linalg.solve(M[:n, :n], J[:, :n].T)
    return np.linalg.inv(Linv)


def m_eff(model, data, u=None, locked=True, site=EE_SITE):
    """Effective mass [kg] at the hand along unit direction u (world). Default u: the tool z-axis."""
    M, J = mass_matrix(model, data), site_jacobian(model, data, site)[:3]
    if u is None:
        u = data.site_xmat[model.site(site).id].reshape(3, 3)[:, 2]
    u = np.asarray(u, float) / np.linalg.norm(u)
    n = 6 if locked else model.nv
    return float(1.0 / (u @ J[:, :n] @ np.linalg.solve(M[:n, :n], J[:, :n].T) @ u))


def reduced_mass(meff, m_other=None):
    return meff if m_other is None else 1.0 / (1.0 / meff + 1.0 / m_other)


def impulse(meff, v, e=RESTITUTION, m_other=None):
    """Contact impulse [N s] for an approach speed v [m/s]."""
    return (1.0 + e) * reduced_mass(meff, m_other) * v


def delta_v_struck(meff, v, m_other, e=RESTITUTION):
    """Speed [m/s] given to an unrestrained body of mass m_other (e.g. an astronaut on EVA)."""
    return impulse(meff, v, e, m_other) / m_other


def v_max_impulse(meff, limit, e=RESTITUTION, m_other=None):
    """Plan form: the speed whose contact impulse equals `limit` [N s]."""
    return limit / ((1.0 + e) * reduced_mass(meff, m_other))


def v_max_iso(meff, region="chest", restrained=False):
    """ISO/TS 15066 energy model: v = F_max / sqrt(mu k). restrained=True -> mu = m_eff."""
    b = BODY[region]
    mu = reduced_mass(meff, None if restrained else b["m"])
    return float(b["F_max"] / np.sqrt(mu * b["k"]))


def hand_speed_margin(model, data, region="chest", locked=True, site=EE_SITE):
    """(hand speed, v_max along the direction of travel) for a live display. Uses the
    current hand velocity direction; at rest returns (0, v_max along the tool axis)."""
    J = site_jacobian(model, data, site)[:3]
    v = J @ data.qvel
    speed = float(np.linalg.norm(v))
    u = v / speed if speed > 1e-6 else None
    return speed, v_max_iso(m_eff(model, data, u, locked, site), region)


def vmax_map(model, n=41, region="chest", locked=True, q_base=None):
    """v_max over shoulder pitch (j2) x elbow (j4), other joints at the home pose; contact
    along the tool axis. Returns (q2, q4, V) with V[i4, i2]; NaN where the arm hits the base."""
    d = mujoco.MjData(model)
    q_home = model.key("home").qpos.copy() if q_base is None else np.asarray(q_base, float)
    lo, hi = model.jnt_range[2], model.jnt_range[4]
    q2, q4 = np.linspace(lo[0], lo[1], n), np.linspace(hi[0], hi[1], n)
    V = np.full((n, n), np.nan)
    for a, s in enumerate(q2):
        for b, e in enumerate(q4):
            d.qpos[:] = q_home
            d.qpos[8], d.qpos[10] = s, e
            mujoco.mj_forward(model, d)
            if d.ncon:
                continue
            V[b, a] = v_max_iso(m_eff(model, d, locked=locked), region)
    return q2, q4, V


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Contact-velocity limit (P4)")
    ap.add_argument("--model", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--region", choices=list(BODY), default="chest")
    ap.add_argument("--map", action="store_true", help="v_max heatmap over shoulder x elbow")
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--out", help="PNG path for the map (a CSV is written next to it)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    model = mujoco.MjModel.from_xml_path(str(root / "models" / f"ff_{a.model}.xml"))
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    for locked in (True, False):
        me = m_eff(model, d, locked=locked)
        print(f"home pose, {'joints locked' if locked else 'joints free  '}: m_eff = {me:6.2f} kg   "
              f"v_max ({a.region}, unrestrained) = {v_max_iso(me, a.region):.3f} m/s   "
              f"(restrained {v_max_iso(me, a.region, restrained=True):.3f} m/s)")
    if a.map:
        q2, q4, V = vmax_map(model, a.n, a.region)
        print(f"map: v_max {np.nanmin(V):.3f} .. {np.nanmax(V):.3f} m/s, {np.isnan(V).sum()} poses in collision")
        if a.out:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out.with_suffix(".csv"), V, delimiter=",",
                       header=f"rows: j4 {q4[0]:.3f}..{q4[-1]:.3f} rad, cols: j2 {q2[0]:.3f}..{q2[-1]:.3f} rad, {a.n} each")
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            im = ax.imshow(V, origin="lower", aspect="auto", cmap="viridis",
                           extent=np.degrees([q2[0], q2[-1], q4[0], q4[-1]]))
            hq = model.key("home").qpos
            ax.plot(np.degrees(hq[8]), np.degrees(hq[10]), "w+", ms=8)
            ax.set_xlabel("shoulder j2 [deg]")
            ax.set_ylabel("elbow j4 [deg]")
            ax.set_title(f"v_max, ff_{a.model} ({a.region})", fontsize=9)
            fig.colorbar(im, ax=ax, label="m/s")
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            print("saved", out)


if __name__ == "__main__":
    main()
