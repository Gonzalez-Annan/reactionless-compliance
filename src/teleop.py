"""Teleoperation loop: operator input -> task-space velocity -> controller. Owner: P1.

    python -m src.teleop --model medium --mode 6d
    python -m src.teleop --model small --log data/pilot/test.csv

One pygame window: MuJoCo renders off-screen into it and pygame reads all
input, so the MuJoCo viewer's own keyboard shortcuts cannot interfere.

Rates: physics at model.opt.timestep (4 kHz), control at CONTROL_HZ (100 Hz,
the command is held over the physics steps in between), display ~30 fps.

Input -> commanded end-effector twist xdot = [vx vy vz wx wy wz], world frame
(same row order as mj_jacSite's [jacp; jacr]):

    W / S        +x / -x          Q / E         roll  (wx) + / -
    A / D        +y / -y          mouse drag    yaw (wz) and pitch (wy)
    R / F        +z / -z          (hold left button)
    Shift        fine mode (x0.3)
    G            snap target back to the hand     P  pause
    Backspace    reset simulation                 Esc  quit

In 3-D mode the rotational part of xdot is zeroed.

Controller interface (P3's schemes plug in here):

    controller(model, data, xdot, mode) -> qdot    # shape (7,), rad/s

The returned qdot is written to data.ctrl (velocity servos). If any joint
exceeds its ctrlrange the whole vector is scaled down, preserving direction.
Until src/schemes.py provides SCHEMES = {name: controller}, only "hold"
(qdot = 0) is available: the arm stays still and the ghost target shows
the commanded motion.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import pygame
from scipy.spatial.transform import Rotation

from src.logger import CsvLogger

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CONTROL_HZ = 100
RENDER_EVERY = 3            # control ticks per rendered frame (~33 fps)
WIDTH, HEIGHT = 1280, 720   # must fit models/common/arm_defs.xml <visual> offwidth/offheight
ANTENNA_LIMIT_DEG = 5.0

LIN_SPEED = 0.05            # m/s at full input
ANG_SPEED = 0.25            # rad/s at full input
FINE_SCALE = 0.3
MOUSE_PX_FULL = 10          # mouse pixels per control tick that count as full input


def hold(model, data, xdot, mode):
    """Placeholder controller: keep the arm still."""
    return np.zeros(model.nu)


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


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, float))
    scene.ngeom += 1


class Teleop:
    def __init__(self, model_name, mode, controller_name, log_path=None, headless=False):
        self.model_name = f"ff_{model_name}"
        self.model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / f"{self.model_name}.xml"))
        self.data = mujoco.MjData(self.model)
        self.mode = mode
        self.controller_name = controller_name
        self.controller = load_controller(controller_name)
        self.n_sub = round(1.0 / CONTROL_HZ / self.model.opt.timestep)
        self.headless = headless
        self.logger = None
        if log_path:
            self.logger = CsvLogger(log_path, meta={
                "model": self.model_name, "mode": mode, "controller": controller_name,
                "control_hz": CONTROL_HZ, "timestep": self.model.opt.timestep,
            })
        self.reset()

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Reactionless Compliance - teleop")
            self.font = pygame.font.SysFont("monospace", 18)
            self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.6, 0, 0]
            self.cam.distance = 3.0
            self.cam.azimuth = 135
            self.cam.elevation = -25
            self.input = KeyboardMouse()
            pygame.mouse.get_rel()  # discard the first, large, relative motion

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.antenna_z0 = site_pose(self.model, self.data, "antenna")[1][:, 2].copy()
        self.snap_target()
        self.overruns = 0
        self.paused = False

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

        if self.logger:
            ee_pos, ee_R = site_pose(m, d, "end_effector")
            self.logger.write(
                t_sim=d.time, xdot_cmd=xdot, qdot_cmd=qdot,
                qpos=d.qpos, qvel=d.qvel,
                ee_pos=ee_pos, ee_quat=Rotation.from_matrix(ee_R).as_quat(scalar_first=True),
                target_pos=self.target_pos, target_quat=self.target_rot.as_quat(scalar_first=True),
                antenna_err_deg=self.antenna_error_deg(),
            )

    def draw(self):
        r = self.renderer
        r.update_scene(self.data, camera=self.cam)
        scn = r.scene

        # antenna: boresight at start (green) and now (red once past the limit)
        a_pos, a_R = site_pose(self.model, self.data, "antenna")
        err = self.antenna_error_deg()
        add_line(scn, a_pos, a_pos + 0.8 * self.antenna_z0, [0.2, 0.9, 0.2, 1])
        add_line(scn, a_pos, a_pos + 0.8 * a_R[:, 2],
                 [0.95, 0.2, 0.2, 1] if err > ANTENNA_LIMIT_DEG else [0.95, 0.8, 0.1, 1])

        # ghost target: sphere + its axes (6-D mode)
        add_sphere(scn, self.target_pos, 0.03, [0.3, 0.7, 1.0, 0.8])
        if self.mode == "6d":
            R = self.target_rot.as_matrix()
            for i, c in enumerate(([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])):
                add_line(scn, self.target_pos, self.target_pos + 0.08 * R[:, i], c, width=2)

        frame = r.render()
        self.screen.blit(pygame.image.frombuffer(frame.tobytes(), (WIDTH, HEIGHT), "RGB"), (0, 0))

        lines = [
            f"model {self.model_name}   mode {self.mode}   controller {self.controller_name}"
            + ("   [PAUSED]" if self.paused else ""),
            f"t = {self.data.time:6.1f} s   loop overruns {self.overruns}",
            f"antenna error {err:5.2f} deg  (limit {ANTENNA_LIMIT_DEG:.0f})",
            "WASD/RF move  Q/E roll  drag: pitch/yaw  Shift fine  G snap  P pause  Bksp reset  Esc quit",
        ]
        for i, text in enumerate(lines):
            colour = (255, 80, 80) if i == 2 and err > ANTENNA_LIMIT_DEG else (240, 240, 240)
            self.screen.blit(self.font.render(text, True, colour), (12, 10 + 22 * i))
        pygame.display.flip()

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_BACKSPACE:
                    self.reset()
                elif event.key == pygame.K_g:
                    self.snap_target()
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
                u = self.input.read()
                if not self.paused:
                    self.control_tick(self.twist_from_input(u))
                if tick % RENDER_EVERY == 0:
                    self.draw()
                tick += 1

                next_tick += period
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    self.overruns += 1
                    if slack < -0.1:  # fell far behind (e.g. window dragged): don't try to catch up
                        next_tick = time.perf_counter()
        finally:
            self.close()

    def close(self):
        if self.logger:
            self.logger.close()
        if not self.headless:
            self.renderer.close()
            pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--mode", choices=["3d", "6d"], default="6d")
    ap.add_argument("--controller", default="hold", help='"hold" or a name from src.schemes.SCHEMES')
    ap.add_argument("--log", help="CSV path, e.g. data/pilot/test.csv")
    ap.add_argument("--duration", type=float, help="stop after this many simulated seconds")
    args = ap.parse_args()
    Teleop(args.model, args.mode, args.controller, args.log).run(args.duration)


if __name__ == "__main__":
    main()
