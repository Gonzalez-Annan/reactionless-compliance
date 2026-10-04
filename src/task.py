"""Closed-loop ORU task, 3-D or 6-D. Owner: P1.

The operator moves the hand from its start pose to an orbital replacement
unit (ORU), extracts it, stows it, and returns to the start pose. Two loops
per trial; returning to the start is what makes base drift and joint drift
(||q_f - q_0||) measurable.

    start -> approach -> ORU -> approach -> stow -> start      (x loops)

Waypoints are offsets from the hand's start pose, in the world frame at
t = 0 (= base frame, the base starts at the origin). A waypoint is reached
when the actual hand -- not the commanded target -- stays within POS_TOL
(and ROT_TOL_DEG in 6-D mode) for DWELL seconds.

Two uses:
  * human trials: Teleop(task=ClosedLoopTask(...)) draws the waypoints and
    ends the trial when all loops are done;
  * scripted runs (no human, for P3's stability gate and the beta-sweep):

    python -m src.task --model medium --mode 6d --controller demo
    python -m src.task --model small --mode 3d --controller dls --log data/pilot/scripted.csv
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

POS_TOL = 0.02        # m
ROT_TOL_DEG = 10.0    # 6-D mode only
DWELL = 0.3           # s the hand must stay inside the tolerance

# (name, position offset [m], rotation offset as rotation vector [deg]) from the start pose
# The home pose is at ~85% reach with the elbow 20 deg from its limit, so the
# loop moves sideways and up/down, with a short forward insertion at the ORU.
# Waypoints are fixed in the world (the ORU sits on another spacecraft), so
# base drift moves them relative to the arm.
WAYPOINTS = [
    ("approach", (0.00, 0.15, 0.08), (0, 0, 30)),
    ("ORU",      (0.05, 0.15, 0.08), (0, 0, 30)),
    ("approach", (0.00, 0.15, 0.08), (0, 0, 30)),
    ("stow",     (0.00, -0.15, -0.08), (30, 0, 0)),
    ("start",    (0.00, 0.00, 0.00), (0, 0, 0)),
]

# scripted reference speeds: same as full stick deflection in teleop
REF_LIN_SPEED = 0.05  # m/s
REF_ANG_SPEED = 0.25  # rad/s
MIN_SEGMENT = 1.0     # s


class ClosedLoopTask:
    def __init__(self, start_pos, start_rot, mode, loops=2):
        self.mode = mode
        self.loops = loops
        self.start = (np.asarray(start_pos, float).copy(), start_rot)
        self.waypoints = []
        for name, dp, drv in WAYPOINTS:
            rot = Rotation.from_rotvec(np.radians(drv)) * start_rot if mode == "6d" else start_rot
            self.waypoints.append((name, self.start[0] + np.asarray(dp), rot))
        self.index = 0      # current waypoint
        self.loop = 0       # completed loops
        self.inside = 0.0   # time spent inside the tolerance so far
        self.done = False

    @property
    def current(self):
        return self.waypoints[self.index]

    def reached(self, ee_pos, ee_rot):
        _, pos, rot = self.current
        if np.linalg.norm(ee_pos - pos) > POS_TOL:
            return False
        if self.mode == "6d":
            return np.degrees((rot * ee_rot.inv()).magnitude()) <= ROT_TOL_DEG
        return True

    def update(self, ee_pos, ee_rot, dt):
        """Call once per control tick with the actual hand pose."""
        if self.done:
            return
        self.inside = self.inside + dt if self.reached(ee_pos, ee_rot) else 0.0
        if self.inside >= DWELL:
            self.inside = 0.0
            self.index += 1
            if self.index == len(self.waypoints):
                self.index = 0
                self.loop += 1
                self.done = self.loop >= self.loops

    def status(self):
        if self.done:
            return f"done ({self.loops} loops)"
        name = self.current[0]
        return f"waypoint {self.index + 1}/{len(self.waypoints)} ({name})   loop {self.loop + 1}/{self.loops}"

    # --- scripted reference -------------------------------------------------

    def reference(self):
        """Smooth reference through all waypoints for all loops.

        Returns (duration, f) where f(t) -> (pos, Rotation, twist). Each
        segment is a minimum-jerk move followed by a DWELL-long hold, so a
        controller that tracks it perfectly also satisfies the task.
        """
        poses = [self.start] + [(p, r) for _, p, r in self.waypoints]
        segments = []
        for _ in range(self.loops):
            for (p0, r0), (p1, r1) in zip(poses[:-1], poses[1:]):
                dp = p1 - p0
                drv = (r1 * r0.inv()).as_rotvec()
                T = max(np.linalg.norm(dp) / REF_LIN_SPEED,
                        np.linalg.norm(drv) / REF_ANG_SPEED, MIN_SEGMENT)
                segments.append((T, p0, r0, dp, drv))
                segments.append((DWELL + 0.2, p1, r1, np.zeros(3), np.zeros(3)))
        duration = sum(s[0] for s in segments)

        def f(t):
            for T, p0, r0, dp, drv in segments:
                if t <= T:
                    break
                t -= T
            tau = np.clip(t / T, 0.0, 1.0)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5          # min-jerk 0 -> 1
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T  # its time derivative
            pos = p0 + dp * s
            rot = Rotation.from_rotvec(drv * s) * r0
            twist = np.r_[dp * ds, drv * ds]
            return pos, rot, twist

        return duration, f


def run_scripted(model_name, mode, controller_name, loops=2, log_path=None, timeout_factor=1.5):
    """Drive the task with the scripted reference instead of a human.

    Returns a small dict of sanity numbers (not the paper metrics -- those
    are P5's, computed from the log)."""
    from src.teleop import CONTROL_HZ, Teleop, site_pose

    tele = Teleop(model_name, mode, controller_name, log_path=log_path, headless=True)
    ee_pos, ee_R = site_pose(tele.model, tele.data, "end_effector")
    task = ClosedLoopTask(ee_pos, Rotation.from_matrix(ee_R), mode, loops)
    tele.task = task
    duration, ref = task.reference()

    max_qdot = 0.0
    while not task.done and tele.data.time < timeout_factor * duration:
        pos, rot, twist = ref(tele.data.time)
        if mode == "3d":
            twist[3:] = 0.0
        tele.target_pos, tele.target_rot = pos, rot
        tele.control_tick(twist)
        max_qdot = max(max_qdot, np.abs(tele.data.ctrl).max())
    tele.close()

    return {
        "completed": task.done,
        "sim_time_s": round(tele.data.time, 2),
        "reference_time_s": round(duration, 2),
        "final_antenna_err_deg": round(tele.antenna_error_deg(), 3),
        "joint_return_err_rad": round(float(np.linalg.norm(tele.data.qpos[7:] - tele.model.key("home").qpos[7:])), 4),
        "max_qdot_cmd": round(max_qdot, 3),
        "control_hz": CONTROL_HZ,
    }


def main():
    ap = argparse.ArgumentParser(description="Scripted (no-human) run of the closed-loop task")
    ap.add_argument("--model", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--mode", choices=["3d", "6d"], default="6d")
    ap.add_argument("--controller", default="demo")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--log", help="CSV path")
    args = ap.parse_args()
    result = run_scripted(args.model, args.mode, args.controller, args.loops, args.log)
    for k, v in result.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
