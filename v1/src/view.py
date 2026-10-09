"""Watch trials in the MuJoCo viewer, in real time.

    python src/view.py                      # small servicer, position task: DLS, then Imp.+RNS
    python src/view.py medium 6d 1 4 5      # size, mode, schemes...
"""
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from schemes import NAMES
from task import run_trial, DECIM


SHELL = ([0.2, 1, 0.3, 0.22], [1, 0.55, 0.1, 0.22], [0.2, 0.5, 1, 0.08])   # faces: base tilt under 0.5, 1, 2 deg


def draw_envelope(scn, n, tips, tris, edges):
    """Teleop work envelope through the probe rays (tips[direction][shell]): a blue cage where the base reaches
    2 deg, and inside it a yellow and a green volume for 1 and 0.5 deg (filled, no lines: as wireframes the
    three were a tangle). tips grows while the rays are run; only finished corners are drawn. -> next free geom index."""
    eye, k = np.eye(3).flatten(), len(tips)
    for i, j in edges:
        if max(i, j) < k and n < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], tips[i][-1], eye, [0.4, 0.85, 1, 0.9])
            mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 2, tips[i][-1], tips[j][-1])
            n += 1
    for sh, rgba in enumerate(SHELL):
        for i, j, l in tris:
            if max(i, j, l) < k and n + 2 <= scn.maxgeom:
                a, b, c = tips[i][sh], tips[j][sh], tips[l][sh]
                for e1, e2 in ((b - a, c - a), (c - a, b - a)):     # both windings: seen from inside and outside
                    M = np.column_stack([e1, e2, np.cross(e1, e2)])
                    mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_TRIANGLE, [1, 1, 1], a, M.flatten(), rgba)
                    n += 1
    return n


def viewer_hook(show_ref=True):
    """on_step callback for run_trial: opens the viewer on the first step and paces to real time."""
    s = {}

    def on_step(m, d, ref):
        if "v" not in s:
            s["v"], s["t0"] = mujoco.viewer.launch_passive(m, d), time.time() - d.time
        v = s["v"]
        if show_ref:   # green = where the hand should be
            mujoco.mjv_initGeom(v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                                ref[0], np.eye(3).flatten(), s.get("rgba", [0, 1, 0, 0.6]))
            n = 1
            if s.get("far") is not None:        # teleop: how far the hand would get toward an unreachable dot
                mujoco.mjv_initGeom(v.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], s["far"],
                                    np.eye(3).flatten(), [0.2, 0.5, 1, 0.9])
                n += 1
            if s.get("tgt") is not None:        # pilot: the goal to hold the hand on
                mujoco.mjv_initGeom(v.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], s["tgt"],
                                    np.eye(3).flatten(), [1, 0.2, 0.9, 0.5])
                n += 1
            if "cage" in s and s.get("show", 2):   # teleop work envelope; show: 0 slices, 1 both, 2 shells
                n = draw_envelope(v.user_scn, n, list(s.get("rays", [])), *s["cage"])
            if s.get("show", 0) < 2:               # the shells cut by the three planes through the dot
                for seg, rgba in s.get("cuts", ()):
                    for a, b in seg:
                        if n + 8 < v.user_scn.maxgeom:
                            mujoco.mjv_initGeom(v.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], a, np.eye(3).flatten(), rgba)
                            mujoco.mjv_connector(v.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_LINE, 3, a, b)
                            n += 1
            if "hand" in s:                        # teleop: live singularity margin, a ball on the hand
                mujoco.mjv_initGeom(v.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], s["hand"][0],
                                    np.eye(3).flatten(), s["hand"][1])
                n += 1
                if hasattr(v, "set_texts"):
                    v.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, "margin (1 = ready pose)", s["text"]))
            if "link" in s:                        # white = hand to goal (teleop)
                g = v.user_scn.geoms[n]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0], s["link"][1], np.eye(3).flatten(), [1, 1, 1, 0.9])
                if np.linalg.norm(s["link"][1] - s["link"][0]) > 1e-4:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2, *s["link"])
                n += 1
            v.user_scn.ngeom = n
        v.sync()
        time.sleep(max(0, s["t0"] + d.time + DECIM * m.opt.timestep - time.time()))
        return v.is_running()

    return s, on_step


def summary(r):
    return (f"hand error {r['rms_pos_mm']:.1f} mm, peak base drift {r['max_base_deg']:.2f} deg, "
            f"a joint at its limit {r['limit_frac']:.0%} of the time")


if __name__ == "__main__":
    a = sys.argv[1:]
    for scheme in [int(x) for x in a[2:]] or [1, 5]:
        size, mode = a[0] if a else "small", a[1] if len(a) > 1 else "3d"
        print(f"{NAMES[scheme]} on the {size} servicer, {mode} task", flush=True)
        s, hook = viewer_hook()
        print("  " + summary(run_trial(size, scheme, mode, loops=1, on_step=hook)), flush=True)
        time.sleep(2)
        s["v"].close()
