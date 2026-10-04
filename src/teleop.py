"""Teleoperation loop: operator input -> task-space velocity -> controller. Owner: P1.

    python -m src.teleop --model medium --mode 6d
    python -m src.teleop --model small --log data/pilot/test.csv
    python -m src.teleop --task --controller demo      # with the closed-loop ORU task

One pygame OpenGL window: MuJoCo draws straight into it (mjr_render) and
pygame reads all input, so the MuJoCo viewer's own keyboard shortcuts cannot
interfere. Off-screen rendering + blitting gave black frames on a hybrid
NVIDIA/AMD Wayland laptop; drawing into the window's own GL context works.

Rates: physics at model.opt.timestep (4 kHz), control at CONTROL_HZ (100 Hz,
the command is held over the physics steps in between), display ~30 fps.

Input -> commanded end-effector twist xdot = [vx vy vz wx wy wz], world frame
(same row order as mj_jacSite's [jacp; jacr]):

    W / S        +x / -x          Q / E         roll  (wx) + / -
    A / D        +y / -y          mouse drag    yaw (wz) and pitch (wy)
    R / F        +z / -z          (hold left button)
    Shift        fine mode (x0.3)
    G            snap target back to the hand     P  pause
    V            show / hide the reachable workspace
    Space        start the trial (when waiting)
    Backspace    reset simulation                 Esc  quit

In 3-D mode the rotational part of xdot is zeroed.

Controller interface (P3's schemes plug in here):

    controller(model, data, xdot, mode) -> qdot    # shape (7,), rad/s

The returned qdot is written to data.ctrl (velocity servos). If any joint
exceeds its ctrlrange the whole vector is scaled down, preserving direction.
Built in: "hold" (qdot = 0, arm stays still, only the target moves) and
"demo" (a crude fixed-base follower for showing teammates the base reaction;
NOT one of the five benchmark schemes). P3's schemes come from
src/schemes.py: SCHEMES = {name: controller}.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import pygame
from scipy.spatial.transform import Rotation

from src.logger import CsvLogger
from src.task import ClosedLoopTask
from src.workspace import reachable_points, surface_voxels

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CONTROL_HZ = 100
RENDER_EVERY = 3            # control ticks per rendered frame (~33 fps)
WIDTH, HEIGHT = 1280, 720
ANTENNA_LIMIT_DEG = 5.0
WORKSPACE_VOXEL = 0.1       # m, cube size for the workspace display

LIN_SPEED = 0.05            # m/s at full input
ANG_SPEED = 0.25            # rad/s at full input
FINE_SCALE = 0.3
MOUSE_PX_FULL = 10          # mouse pixels per control tick that count as full input


def hold(model, data, xdot, mode):
    """Placeholder controller: keep the arm still."""
    return np.zeros(model.nu)


def make_demo(get_target, gain=2.0, damping=0.05):
    """DEMO ONLY -- not one of the five benchmark schemes (those are P3's).

    Drives the hand toward the target with damped least squares on the
    FIXED-BASE arm Jacobian, i.e. it ignores that the base floats. That makes
    the base reaction easy to see: the base turns as the arm moves.
    """
    def demo(model, data, xdot, mode):
        sid = model.site("end_effector").id
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, 6:]

        target_pos, target_rot = get_target()
        ee_rot = Rotation.from_matrix(data.site_xmat[sid].reshape(3, 3))
        err = np.r_[target_pos - data.site_xpos[sid], (target_rot * ee_rot.inv()).as_rotvec()]
        v = xdot + gain * err
        if mode == "3d":
            J, v = J[:3], v[:3]
        return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(v)), v)
    return demo


def load_controller(name):
    if name == "hold":
        return hold
    from src import schemes  # P3
    return schemes.SCHEMES[name]


class KeyboardMouse:
    """WASD + mouse -> normalised twist in [-1, 1]^6."""

    def read(self):
        keys = pygame.key.get_pressed()
        axis = lambda pos, neg: float(keys[pos]) - float(keys[neg])
        u = np.array([
            axis(pygame.K_w, pygame.K_s),
            axis(pygame.K_a, pygame.K_d),
            axis(pygame.K_r, pygame.K_f),
            axis(pygame.K_q, pygame.K_e),
            0.0,
            0.0,
        ])
        dx, dy = pygame.mouse.get_rel()
        if pygame.mouse.get_pressed()[0]:
            u[4] = np.clip(dy / MOUSE_PX_FULL, -1, 1)   # drag down = pitch +
            u[5] = np.clip(-dx / MOUSE_PX_FULL, -1, 1)  # drag left = yaw +
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            u *= FINE_SCALE
        return u


def limit_qdot(model, qdot):
    """Scale qdot uniformly so every joint is inside its ctrlrange."""
    hi = model.actuator_ctrlrange[:, 1]
    ratio = np.max(np.abs(qdot) / hi)
    return qdot / ratio if ratio > 1 else qdot


def site_pose(model, data, name):
    sid = model.site(name).id
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def add_line(scene, a, b, rgba, width=3):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, float))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def add_box(scene, pos, half, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.full(3, half), np.asarray(pos, float), mat.ravel(), np.asarray(rgba, float))
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, float))
    scene.ngeom += 1


class Teleop:
    def __init__(self, model_name, mode, controller_name, log_path=None, headless=False,
                 task_loops=None, banner=None, blind=False, meta=None):
        """task_loops: run the closed-loop ORU task with this many loops; the
        trial ends when it is done. banner: wait for Space before starting,
        showing this text. blind: hide the controller name (participants)."""
        self.model_name = f"ff_{model_name}"
        self.model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / f"{self.model_name}.xml"))
        self.data = mujoco.MjData(self.model)
        self.mode = mode
        self.controller_name = controller_name
        if controller_name == "demo":
            self.controller = make_demo(lambda: (self.target_pos, self.target_rot))
        else:
            self.controller = load_controller(controller_name)
        self.n_sub = round(1.0 / CONTROL_HZ / self.model.opt.timestep)
        self.headless = headless
        self.logger = None
        if log_path:
            self.logger = CsvLogger(log_path, meta={
                "model": self.model_name, "mode": mode, "controller": controller_name,
                "control_hz": CONTROL_HZ, "timestep": self.model.opt.timestep,
                **(meta or {}),
            })
        self.task_loops = task_loops
        self.blind = blind
        self.reset()
        self.banner = banner
        self.waiting = banner is not None

        if not headless:
            pygame.init()
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            pygame.display.set_caption("Reactionless Compliance - teleop")
            # MuJoCo renders into the GL context pygame just made current
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
            self.scene = mujoco.MjvScene(self.model, maxgeom=5000)
            self.vopt = mujoco.MjvOption()
            self.viewport = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.6, 0, 0]
            self.cam.distance = 3.0
            self.cam.azimuth = 135
            self.cam.elevation = -25
            self.input = KeyboardMouse()
            pygame.mouse.get_rel()  # discard the first, large, relative motion

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.antenna_z0 = site_pose(self.model, self.data, "antenna")[1][:, 2].copy()
        self.snap_target()
        self.overruns = 0
        self.paused = False
        self.show_workspace = False
        self.workspace = None  # surface voxel centres in the base frame, computed on first V
        self.task = None
        if self.task_loops:
            ee_pos, ee_R = site_pose(self.model, self.data, "end_effector")
            self.task = ClosedLoopTask(ee_pos, Rotation.from_matrix(ee_R), self.mode, self.task_loops)

    def snap_target(self):
        self.target_pos, R = site_pose(self.model, self.data, "end_effector")
        self.target_rot = Rotation.from_matrix(R)

    def antenna_error_deg(self):
        z = site_pose(self.model, self.data, "antenna")[1][:, 2]
        return float(np.degrees(np.arccos(np.clip(z @ self.antenna_z0, -1.0, 1.0))))

    def twist_from_input(self, u):
        xdot = np.r_[u[:3] * LIN_SPEED, u[3:] * ANG_SPEED]
        if self.mode == "3d":
            xdot[3:] = 0.0
        return xdot

    def control_tick(self, xdot):
        """One control period: controller -> ctrl -> n_sub physics steps -> log."""
        m, d = self.model, self.data
        qdot = limit_qdot(m, np.asarray(self.controller(m, d, xdot, self.mode), float))
        d.ctrl[:] = qdot
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)

        dt = 1.0 / CONTROL_HZ
        self.target_pos = self.target_pos + xdot[:3] * dt
        self.target_rot = Rotation.from_rotvec(xdot[3:] * dt) * self.target_rot

        ee_pos, ee_R = site_pose(m, d, "end_effector")
        if self.task:
            self.task.update(ee_pos, Rotation.from_matrix(ee_R), dt)

        if self.logger:
            self.logger.write(
                t_sim=d.time, xdot_cmd=xdot, qdot_cmd=qdot,
                qpos=d.qpos, qvel=d.qvel,
                ee_pos=ee_pos, ee_quat=Rotation.from_matrix(ee_R).as_quat(scalar_first=True),
                target_pos=self.target_pos, target_quat=self.target_rot.as_quat(scalar_first=True),
                antenna_err_deg=self.antenna_error_deg(),
                task_waypoint=self.task.index if self.task else -1,
                task_loop=self.task.loop if self.task else -1,
            )

    def draw(self):
        scn = self.scene
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scn)

        # antenna: boresight at start (green) and now (red once past the limit)
        a_pos, a_R = site_pose(self.model, self.data, "antenna")
        err = self.antenna_error_deg()
        add_line(scn, a_pos, a_pos + 0.8 * self.antenna_z0, [0.2, 0.9, 0.2, 1])
        add_line(scn, a_pos, a_pos + 0.8 * a_R[:, 2],
                 [0.95, 0.2, 0.2, 1] if err > ANTENNA_LIMIT_DEG else [0.95, 0.8, 0.1, 1])

        # reachable workspace, carried along with the base
        if self.show_workspace:
            if self.workspace is None:
                self.workspace = surface_voxels(reachable_points(self.model), WORKSPACE_VOXEL)
            bid = self.model.body("base").id
            b_pos, b_R = self.data.xpos[bid], self.data.xmat[bid].reshape(3, 3)
            for c in self.workspace:
                add_box(scn, b_pos + b_R @ c, 0.45 * WORKSPACE_VOXEL, b_R, [0.3, 0.8, 0.9, 0.12])

        # task: path (thin), other waypoints (small grey), current one (orange + axes)
        if self.task and not self.task.done:
            wps = self.task.waypoints
            for (_, p0, _), (_, p1, _) in zip([(None, *self.task.start)] + wps[:-1], wps):
                add_line(scn, p0, p1, [0.6, 0.6, 0.6, 0.5], width=1)
            for i, (_, p, r) in enumerate(wps):
                if i != self.task.index:
                    add_sphere(scn, p, 0.012, [0.7, 0.7, 0.7, 0.6])
            _, p, r = self.task.current
            add_sphere(scn, p, 0.035, [1.0, 0.55, 0.1, 0.45])
            if self.mode == "6d":
                R = r.as_matrix()
                for i, c in enumerate(([1, 0.4, 0.4, 1], [0.4, 1, 0.4, 1], [0.4, 0.4, 1, 1])):
                    add_line(scn, p, p + 0.1 * R[:, i], c, width=4)

        # ghost target: sphere + its axes (6-D mode)
        add_sphere(scn, self.target_pos, 0.03, [0.3, 0.7, 1.0, 0.8])
        if self.mode == "6d":
            R = self.target_rot.as_matrix()
            for i, c in enumerate(([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])):
                add_line(scn, self.target_pos, self.target_pos + 0.08 * R[:, i], c, width=2)

        # follow the window size; SDL2 keeps the GL context across resizes
        self.viewport.width, self.viewport.height = pygame.display.get_window_size()
        mujoco.mjr_render(self.viewport, scn, self.ctx)

        over = err > ANTENNA_LIMIT_DEG
        rows = [("model", self.model_name), ("mode", self.mode)]
        if not self.blind:
            rows.append(("controller", self.controller_name))
        rows += [
            ("time", f"{self.data.time:.1f} s" + ("   [PAUSED]" if self.paused else "")),
            ("late ticks", str(self.overruns)),
            ("antenna error", f"{err:.2f} deg" + ("   OVER LIMIT" if over else f"   (limit {ANTENNA_LIMIT_DEG:.0f})")),
        ]
        if self.task:
            rows.append(("task", self.task.status()))
        labels = "\n".join(r[0] for r in rows)
        values = "\n".join(r[1] for r in rows)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           self.viewport, labels, values, self.ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           self.viewport,
                           "WASD / RF\nQ / E\nleft-drag\nShift\nG   P\nV\nBackspace   Esc",
                           "move x y / z\nroll\npitch / yaw\nfine mode\nsnap target   pause\nreachable workspace\nreset   quit",
                           self.ctx)
        if self.waiting:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP,
                               self.viewport, self.banner + "\n\npress SPACE to start", "", self.ctx)
        pygame.display.flip()

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE and self.waiting:
                    self.waiting = False
                    pygame.mouse.get_rel()
                elif event.key == pygame.K_BACKSPACE and not self.task:  # no resets mid-trial
                    self.reset()
                elif event.key == pygame.K_g:
                    self.snap_target()
                elif event.key == pygame.K_v:
                    self.show_workspace = not self.show_workspace
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
            if event.type == pygame.MOUSEWHEEL:
                self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** event.y), 0.5, 15))
        return True

    def run(self, duration=None):
        period = 1.0 / CONTROL_HZ
        next_tick = time.perf_counter()
        tick = 0
        try:
            while self.handle_events():
                if duration is not None and self.data.time >= duration:
                    break
                if self.task and self.task.done:
                    break
                u = self.input.read()
                if not (self.paused or self.waiting):
                    self.control_tick(self.twist_from_input(u))
                if tick % RENDER_EVERY == 0:
                    self.draw()
                tick += 1

                next_tick += period
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:  # this tick took longer than its 10 ms budget
                    self.overruns += 1
                    if slack < -0.1:  # fell far behind (e.g. window dragged): don't try to catch up
                        next_tick = time.perf_counter()
        finally:
            self.close()

    def close(self):
        if self.logger:
            self.logger.close()
        if not self.headless:
            self.ctx.free()
            pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--mode", choices=["3d", "6d"], default="6d")
    ap.add_argument("--controller", default="hold", help='"hold", "demo" (crude fixed-base follower, for showing the base reaction) or a name from src.schemes.SCHEMES')
    ap.add_argument("--log", help="CSV path, e.g. data/pilot/test.csv")
    ap.add_argument("--duration", type=float, help="stop after this many simulated seconds")
    ap.add_argument("--task", action="store_true", help="run the closed-loop ORU task (2 loops)")
    args = ap.parse_args()
    Teleop(args.model, args.mode, args.controller, args.log,
           task_loops=2 if args.task else None,
           banner=f"{args.mode.upper()} task" if args.task else None).run(args.duration)


if __name__ == "__main__":
    main()
