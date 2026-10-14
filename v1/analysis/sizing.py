"""Sizing chart: what a base mass buys in level-base reach, and what reaction wheels would have to absorb. [P2]

    python analysis/sizing.py     # about 5 minutes; prints the table, writes paper/figs/fig6_sizing

Wheel momentum: if wheels hold the base attitude still while the hand moves at v, they carry
h = I~ W J+ v, with I~ the system inertia about its centre of mass (Schur complement of H_b). K = largest
singular value of I~ W J+ (kg m): N m s per m/s of hand speed, and N m per m/s^2 of hand acceleration.
ponytail: K is for minimum-norm joint motion at given poses, rigid bodies, velocity-product terms ignored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import teleop
from dynamics import kin
from make_models import xml
from task import Q_LIM, Q_REST

U = np.r_[np.eye(3), -np.eye(3)]
LIMITS = (0.5, 1.0, 2.0)
V = 0.10    # hand speed of the teleop, m/s


def setup(mass):
    m = mujoco.MjModel.from_xml_string(xml("x", mass))
    m.jnt_range[1:], m.jnt_limited[1:] = Q_LIM, 1
    d = mujoco.MjData(m)
    d.qpos[7:] = Q_REST
    mujoco.mj_forward(m, d)
    return m, d, m.site("ee").id, Rot.from_quat(d.qpos[[4, 5, 6, 3]])


def wheel(k):
    """-> (K in kg m, base turn in deg per 10 cm of hand travel, worst direction)."""
    H, Jp = k.H_b, np.linalg.pinv(k.Jstar[:3])
    X = H[3:, :3] @ np.linalg.inv(H[:3, :3])
    It = H[3:, 3:] - X @ H[:3, 3:]
    assert np.allclose(It @ k.W, -(k.H_bm[3:] - X @ k.H_bm[:3]), atol=1e-8)   # momentum balance
    return np.linalg.svd(It @ k.W @ Jp, compute_uv=False)[0], np.degrees(np.linalg.svd(k.W @ Jp, compute_uv=False)[0]) / 10


def main():
    masses = np.round(np.logspace(np.log10(30), np.log10(3000), 9))
    rows = []
    for mass in masses:
        m, d, sid, Rb0 = setup(mass)
        K, turn = wheel(kin(m, d))
        reach = []
        for lim in LIMITS:
            teleop.BASE_MAX = lim
            m, d, sid, Rb0 = setup(mass)
            p0 = d.site_xpos[sid].copy()
            reach.append(np.mean([(teleop.ray(m, d, sid, Rb0, u, 5, "3d") - p0) @ u for u in U]))
        rows.append([mass, K, turn, *reach])
        print("base %5.0f kg  K %.2f kg m  turn %6.3f deg/10cm  reach at 0.5/1/2 deg: %.2f %.2f %.2f m" % tuple(rows[-1]), flush=True)
    R = np.array(rows)
    m, d, sid, Rb0 = setup(300.0)
    rng, Ks = np.random.default_rng(0), []
    for q in rng.uniform(Q_LIM[:, 0] + 0.3, Q_LIM[:, 1] - 0.3, (2000, 7)):
        d.qpos[7:] = q
        mujoco.mj_forward(m, d)
        Ks.append(wheel(kin(m, d))[0])
    print("K over 2000 poses, 300 kg: median %.2f, 90th pct %.2f, max %.1f kg m" % (np.median(Ks), np.percentile(Ks, 90), max(Ks)))
    print("wheel momentum at %.2f m/s hand speed, rest pose: %.2f to %.2f N m s across base masses" % (V, V * R[:, 1].min(), V * R[:, 1].max()))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6))
    for i, lim in enumerate(LIMITS):
        a.semilogx(R[:, 0], R[:, 3 + i], "o-", ms=3, label=f"{lim:g}$^\\circ$ allowed")
    a.set(xlabel="base mass (kg)", ylabel="mean straight-line reach (m)")
    a.legend(fontsize=7)
    b.loglog(R[:, 0], R[:, 2], "o-", ms=3, color="C3")
    b.set(xlabel="base mass (kg)", ylabel="base turn per 10 cm ($^\\circ$)")
    c = b.twinx()
    c.semilogx(R[:, 0], V * R[:, 1], "s--", ms=3, color="C4")
    c.set_ylabel("wheel momentum (N m s)")
    c.set_ylim(0, 1.5 * V * R[:, 1].max())
    fig.tight_layout()
    out = Path(__file__).parent.parent / "paper" / "figs"
    fig.savefig(out / "fig6_sizing.pdf", bbox_inches="tight")
    fig.savefig(out / "fig6_sizing.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
