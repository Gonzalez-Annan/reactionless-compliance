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


HAPTIC_FORCE_FULL = 10.0           # N of arm contact force that gives full heavy-motor rumble
HAPTIC_JOINT_ZONE = np.radians(15)  # a joint this close to its limit starts the light motor


def haptic_signals(model, data):
    """What the pad should feel, from the current physics state. Added by P3.

    Returns (low, high, contact_force, joint_margin, joint):
      low   heavy motor 0..1 -- the arm touching the base or itself, scaled by the total
            normal contact force (0.35 at first touch, 1 at HAPTIC_FORCE_FULL);
      high  light motor 0..1 -- the joint nearest its limit, ramping from 0.25 at
            HAPTIC_JOINT_ZONE away to 1 at the limit;
      contact_force [N], joint_margin [rad] to the nearest limit, joint index 0..6.
    """
    force, f6 = 0.0, np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)
        force += abs(f6[0])
    q, lim = data.qpos[7:], model.jnt_range[1:]
    margins = np.minimum(q - lim[:, 0], lim[:, 1] - q)
    j = int(np.argmin(margins))
    margin = float(margins[j])
    low = 0.0 if data.ncon == 0 else min(1.0, 0.35 + 0.65 * force / HAPTIC_FORCE_FULL)
    ramp = float(np.clip(1.0 - margin / HAPTIC_JOINT_ZONE, 0.0, 1.0))
    high = 0.0 if ramp == 0.0 else 0.25 + 0.75 * ramp
    return low, high, force, margin, j


class Gamepad:
    """PS4 (DualShock 4) pad -> normalised world twist in [-1, 1]^6 (+ camera rates). Added by P3.

    SDL2 numbering for "PS4 Controller" on Windows, checked on a real pad (2026-10-14):
    axes 0 LX, 1 LY, 2 RX, 3 RY, 4 L2, 5 R2 (triggers rest at -1); buttons 0 Cross,
    1 Circle, 2 Square, 3 Triangle, 4 Share, 6 Options, 7 L3, 8 R3, 9 L1, 10 R1,
    11-14 D-pad up/down/left/right. If DS4Windows or Steam Input is running the pad shows
    up as an Xbox pad and these numbers are wrong.

    layout "game" (default): third-person-game style, moves are relative to the camera
        left stick             move the target across the screen (up = away from you)
        R2 / L2                target up / down
        right stick            orbit the camera
        R1 (hold)              rotate mode: left stick tilts / turns the tool, R2 / L2 roll
        L1 or Square (hold)    fine mode
        D-pad up / down        zoom in / out          R3 (click)   reset the camera
    layout "southpaw": the same with the sticks swapped (camera on the left stick).
    layout "classic": world-frame axes, fixed camera, mirrors the keyboard
        left stick +x/+y, R2/L2 z, R1/L1 roll, right stick pitch/yaw, Square fine.
    All layouts: Cross start, Circle snap target, Triangle workspace, Options pause, Share reset.
    """
    LAYOUTS = ("game", "southpaw", "classic")
    DEAD = 0.12
    CAM_RATE = 90.0          # deg/s at full stick
    ZOOM_RATE = 0.8          # distance x this per second (D-pad held): ~1.25x/s out
    AX_LX, AX_LY, AX_RX, AX_RY, AX_L2, AX_R2 = 0, 1, 2, 3, 4, 5
    BTN = dict(cross=0, circle=1, square=2, triangle=3, share=4, options=6, l3=7, r3=8,
               l1=9, r1=10, up=11, down=12, left=13, right=14)
    EVENTS = {0: pygame.K_SPACE, 1: pygame.K_g, 3: pygame.K_v, 6: pygame.K_p, 4: pygame.K_BACKSPACE,
              8: pygame.K_HOME}

    def __init__(self, layout="game"):
        if layout not in self.LAYOUTS:
            raise ValueError(f"pad layout must be one of {self.LAYOUTS}")
        self.layout = layout
        pygame.joystick.init()
        self.js = None
        self.trigger_seen = [False, False]
        if pygame.joystick.get_count():
            self.attach(0)

    def attach(self, index):
        self.js = pygame.joystick.Joystick(index)
        self.trigger_seen = [False, False]
        print(f"gamepad: {self.js.get_name()} ({self.js.get_numaxes()} axes, "
              f"{self.js.get_numbuttons()} buttons), layout {self.layout}")

    def handle(self, event):
        """Hot-plug, and turn pad button presses into the matching key code (or None)."""
        if event.type == pygame.JOYDEVICEADDED and self.js is None:
            self.attach(event.device_index)
        elif event.type == pygame.JOYDEVICEREMOVED and self.js is not None \
                and event.instance_id == self.js.get_instance_id():
            print("gamepad: disconnected")
            self.js = None
        elif event.type == pygame.JOYBUTTONDOWN and self.js is not None \
                and event.instance_id == self.js.get_instance_id():
            return self.EVENTS.get(event.button)
        return None

    def _axis(self, i):
        a = self.js.get_axis(i)
        return 0.0 if abs(a) < self.DEAD else float(np.sign(a) * (abs(a) - self.DEAD) / (1 - self.DEAD))

    def _trigger(self, k, i):
        # SDL reports 0 for a trigger until it is first touched; treat that as released
        a = self.js.get_axis(i)
        self.trigger_seen[k] |= a != 0.0
        return (a + 1) / 2 if self.trigger_seen[k] else 0.0

    def _btn(self, name):
        return float(self.js.get_button(self.BTN[name]))

    def _sticks(self):
        """(move x, move y, camera x, camera y), stick up = +y."""
        lx, ly, rx, ry = (self._axis(i) for i in (self.AX_LX, self.AX_LY, self.AX_RX, self.AX_RY))
        if self.layout == "southpaw":
            return rx, -ry, lx, -ly
        return lx, -ly, rx, -ry

    def read(self, azimuth=None):
        """World-frame twist intent. azimuth: camera azimuth in degrees (game / southpaw)."""
        if self.js is None:
            return np.zeros(6)
        b = self._btn
        z = self._trigger(1, self.AX_R2) - self._trigger(0, self.AX_L2)
        if self.layout == "classic":
            u = np.array([
                -self._axis(self.AX_LY),
                -self._axis(self.AX_LX),
                z,
                b("r1") - b("l1"),
                self._axis(self.AX_RY),       # stick down = pitch +, like dragging the mouse down
                -self._axis(self.AX_RX),      # stick left = yaw +
            ])
            fine = b("square")
        else:
            mx, my, _, _ = self._sticks()
            az = np.radians(135.0 if azimuth is None else azimuth)
            fwd = np.array([np.cos(az), np.sin(az), 0.0])     # away from you, on the ground plane
            right = np.array([np.sin(az), -np.cos(az), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            if b("r1"):     # rotate mode, flight-stick style: up = tip down, left = turn left, R2 = roll right
                u = np.r_[np.zeros(3), -my * right + -mx * up + z * fwd]
            else:
                u = np.r_[my * fwd + mx * right + z * up, np.zeros(3)]
            fine = max(b("l1"), b("square"))
        if fine:
            u = u * FINE_SCALE
        return np.clip(u, -1, 1)

    def camera(self):
        """(d azimuth, d elevation) in deg/s and zoom factor per second; zero in classic."""
        if self.js is None or self.layout == "classic":
            return 0.0, 0.0, 1.0
        _, _, cx, cy = self._sticks()
        zoom = self.ZOOM_RATE ** (self._btn("up") - self._btn("down"))
        return -cx * self.CAM_RATE, cy * self.CAM_RATE, zoom

    PULSE_MS = 250

    def _rumble(self, low, high, ms):
        try:
            self.js.rumble(low, high, ms)
        except pygame.error:
            pass

    def feel(self, low, high, now_ms=None):
        """Continuous haptics, called at the render rate: heavy motor = contact, light = joint limit.
        Each call lasts a little longer than a frame, so the buzz is smooth and stops by itself."""
        if self.js is None:
            return
        now_ms = pygame.time.get_ticks() if now_ms is None else now_ms
        if now_ms < getattr(self, "_pulse_until", 0):     # an event pulse is playing
            return
        if low > 0 or high > 0:
            self._rumble(low, high, 120)
            self._buzzing = True
        elif getattr(self, "_buzzing", False):
            try:
                self.js.stop_rumble()
            except pygame.error:
                pass
            self._buzzing = False

    def pulse(self, strength=0.9, now_ms=None):
        """One sharp both-motor pulse for an event (antenna leaving the cone); wins over feel()."""
        if self.js is None:
            return
        now_ms = pygame.time.get_ticks() if now_ms is None else now_ms
        self._pulse_until = now_ms + self.PULSE_MS
        self._rumble(strength, strength, self.PULSE_MS)

    rumble = pulse   # old name


class CombinedInput:
    """Keyboard/mouse and gamepad together; both can be used at any time."""

    def __init__(self, pad_layout="game"):
        self.keyboard, self.pad = KeyboardMouse(), Gamepad(pad_layout)

    def read(self, azimuth=None):
        return np.clip(self.keyboard.read() + self.pad.read(azimuth), -1, 1)

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
                 task_loops=None, banner=None, blind=False, meta=None, pad_layout="game",
                 haptics=True, speed_limit=True):
        """task_loops: run the closed-loop ORU task with this many loops; the
        trial ends when it is done. banner: wait for Space before starting,
        showing this text. blind: hide the controller name (participants).
        pad_layout: "game", "southpaw" or "classic" (see Gamepad).
        haptics: rumble on arm contact and near joint limits (see haptic_signals)."""
        self.haptics = haptics
        # P4 safety function: hand speed <= 0.85 x the pose-dependent contact limit (src.safety)
        if speed_limit:
            from src.safety import SpeedGovernor
            self.governor = SpeedGovernor()
        else:
            self.governor = None
        self.speed_scale = 1.0
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
            self.reset_camera()
            self.input = CombinedInput(pad_layout)
            self.over_limit = False
            pygame.mouse.get_rel()  # discard the first, large, relative motion

    def reset_camera(self):
        self.cam.lookat[:] = [0.6, 0, 0]
        self.cam.distance = 3.0
        self.cam.azimuth = 135
        self.cam.elevation = -25

    def camera_state(self):
        """[azimuth deg, elevation deg, distance m]; NaN when headless (no camera)."""
        if self.headless:
            return np.full(3, np.nan)
        return np.array([self.cam.azimuth, self.cam.elevation, self.cam.distance])

    def turn_camera(self, dt):
        daz, de, zoom = self.input.pad.camera()
        self.cam.azimuth = (self.cam.azimuth + daz * dt) % 360
        self.cam.elevation = float(np.clip(self.cam.elevation + de * dt, -89, 89))
        self.cam.distance = float(np.clip(self.cam.distance * zoom ** dt, 0.5, 15))

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.antenna_z0 = site_pose(self.model, self.data, "antenna")[1][:, 2].copy()
        self.snap_target()
        self.haptic = haptic_signals(self.model, self.data)
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
        if getattr(self.controller, "accepts_target", False):   # P3 schemes servo onto the blue sphere
            raw = self.controller(m, d, xdot, self.mode, target=(self.target_pos, self.target_rot))
        else:
            raw = self.controller(m, d, xdot, self.mode)
        qdot = limit_qdot(m, np.asarray(raw, float))
        if self.governor is not None:
            qdot, self.speed_scale = self.governor(m, d, qdot)
        d.ctrl[:] = qdot
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)

        dt = 1.0 / CONTROL_HZ
        self.target_pos = self.target_pos + xdot[:3] * dt
        self.target_rot = Rotation.from_rotvec(xdot[3:] * dt) * self.target_rot

        ee_pos, ee_R = site_pose(m, d, "end_effector")
        if self.task:
            self.task.update(ee_pos, Rotation.from_matrix(ee_R), dt)
        self.haptic = haptic_signals(m, d)

        if self.logger:
            self.logger.write(
                t_sim=d.time, xdot_cmd=xdot, qdot_cmd=qdot,
                qpos=d.qpos, qvel=d.qvel,
                ee_pos=ee_pos, ee_quat=Rotation.from_matrix(ee_R).as_quat(scalar_first=True),
                target_pos=self.target_pos, target_quat=self.target_rot.as_quat(scalar_first=True),
                antenna_err_deg=self.antenna_error_deg(),
                task_waypoint=self.task.index if self.task else -1,
                task_loop=self.task.loop if self.task else -1,
                cam=self.camera_state(),   # az, el, dist: how much the operator turned the view
                haptic=self.haptic[:2] if self.haptics else (0.0, 0.0),   # heavy (contact), light (joint limit)
                contact_force=self.haptic[2],
                joint_margin_deg=np.degrees(self.haptic[3]),
                speed_scale=self.speed_scale,   # < 1: the contact-speed limiter slowed the hand
                v_max_design=self.governor.last_vmax if self.governor else np.nan,   # R2 evidence
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

        # contacts: a red dot where the arm touches the base or itself
        for i in range(self.data.ncon):
            add_sphere(scn, self.data.contact[i].pos, 0.02, [1.0, 0.1, 0.1, 0.9])

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
        if self.haptics:
            if over and not self.over_limit:   # sharp pulse each time the antenna leaves the cone
                self.input.pad.pulse()
            self.input.pad.feel(*self.haptic[:2])
        self.over_limit = over
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
        _, _, force, margin, j = self.haptic
        labels, values = ["arm contact", "nearest limit"], [
            f"{force:.1f} N   TOUCHING" if self.data.ncon else "none",
            f"j{j + 1}  {np.degrees(margin):.0f} deg" + ("   CLOSE" if margin < HAPTIC_JOINT_ZONE else "")]
        if not self.blind:   # P4: hand speed against the pose-dependent contact limit
            from src.safety import hand_speed_margin
            speed, vmax, _ = hand_speed_margin(self.model, self.data)
            labels += ["hand speed", "contact limit"]
            values += [f"{speed:.3f} m/s" + ("   LIMITED" if self.speed_scale < 1 else ""),
                       f"{vmax:.2f} m/s" + ("   OVER" if speed > vmax else "")]
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                           self.viewport, "\n".join(labels), "\n".join(values), self.ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                           self.viewport,
                           "WASD / RF\nQ / E\nleft-drag\nShift\nG   P\nV\nBackspace   Esc",
                           "move x y / z\nroll\npitch / yaw\nfine mode\nsnap target   pause\nreachable workspace\nreset   quit",
                           self.ctx)
        if self.input.pad.js is not None:
            if self.input.pad.layout == "classic":
                keys = "PS4  L-stick / R2 L2\nR1 / L1\nR-stick\nSquare (hold)\nCircle   Options\nTriangle\nShare   Cross"
                does = "move x y / z\nroll\npitch / yaw\nfine mode\nsnap   pause\nworkspace\nreset   start"
            else:
                mv, cm = ("R", "L") if self.input.pad.layout == "southpaw" else ("L", "R")
                keys = (f"PS4  {mv}-stick   R2 / L2\n{cm}-stick   D-pad\nhold R1: {mv}-stick, R2/L2\n"
                        "hold L1\nCircle   R3\nTriangle   Options\nShare   Cross")
                does = ("move   up / down\ncamera   zoom\ntilt / turn, roll\nfine mode\nsnap   camera reset\n"
                        "workspace   pause\nreset   start")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                               self.viewport, keys, does, self.ctx)
        if self.waiting:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP,
                               self.viewport, self.banner + "\n\npress SPACE to start", "", self.ctx)
        pygame.display.flip()

    def handle_events(self):
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            pad_key = self.input.pad.handle(event)
            if pad_key is not None:   # a pad button acts as its keyboard twin
                event = pygame.event.Event(pygame.KEYDOWN, key=pad_key)
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
                elif event.key == pygame.K_HOME:   # Home key or R3: default view
                    self.reset_camera()
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
                self.turn_camera(1.0 / CONTROL_HZ)
                u = self.input.read(self.cam.azimuth)   # game layouts move relative to the view
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
    ap.add_argument("--no-haptics", action="store_true", help="no rumble on contact / joint limits")
    ap.add_argument("--no-speed-limit", action="store_true", help="switch off the contact-speed limiter (P4)")
    ap.add_argument("--pad-layout", choices=Gamepad.LAYOUTS, default="game",
                    help="game: camera on the right stick, moves follow the view; southpaw: sticks swapped; "
                         "classic: world axes, fixed camera")
    args = ap.parse_args()
    Teleop(args.model, args.mode, args.controller, args.log,
           task_loops=2 if args.task else None,
           banner=f"{args.mode.upper()} task" if args.task else None,
           pad_layout=args.pad_layout, haptics=not args.no_haptics,
           speed_limit=not args.no_speed_limit).run(args.duration)


if __name__ == "__main__":
    main()
