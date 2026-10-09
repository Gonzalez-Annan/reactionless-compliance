"""Drive the hand yourself: keyboard now, gamepad (PlayStation or any pygame joystick) with --pad. [P5]

    python src/teleop.py                           # set-and-go, keyboard, medium servicer, Imp.+RNS, position task
    python src/teleop.py --size large --scheme 4
    python src/teleop.py --session 3               # participant 3: all 10 conditions, shuffled, scheme hidden
    python src/teleop.py --delay 2 [--direct]      # 2 s command delay; --direct = hold the keys to drive the hand
    python src/teleop.py --pilot 1                 # delay pilot, participant 1: 4 blocks, five pink goals each
    python src/teleop.py --feedback 1              # feedback study, participant 1: 3 blocks (no cue / colour / hard stop), scored in seconds
    python src/teleop.py --pad ...                 # same, with a gamepad (needs: uv pip install pygame)

Set-and-go (default): move the dot anywhere in 3-D, press Enter (pad: Cross). If the hand can get there with
the base staying within 2 degrees of level it goes; if not, the dot is red and nothing moves.
The cage is the work envelope, drawn once (about a minute and a half) and fixed in space.
Inside the green volume the base tilts under 0.5 degrees, inside the orange under 1, inside the blue cage under 2.
The cage is a guide (right for about 9 goals in 10 near its edge, D20); the dot is the answer, half a second behind:
    green = the hand can get there in one straight move with the base within 2 degrees
    red   = it cannot; the blue ball on the line marks how far it would get
    yellow = still checking
Controls are designed for a PS4 pad; the keyboard mirrors it. Moves are relative to the camera:
"right" is right on your screen, whichever way you have turned the view.

    PS4                      keyboard            does
    left stick               arrows              dot left/right/up/down on screen
    R2 / L2                  PgUp / PgDn         dot away from / toward you
    right stick              Home/End or mouse   turn the camera (to judge depth)
    Cross                    Enter               send
    Circle                   Delete              put the dot back on the hand
    L1 (hold)                Shift (hold)        fine: quarter speed
    Square                   Insert              envelope view: slices through the dot / slices and shells / shells
    Triangle                 Right Ctrl          feedback study: give this goal up as out of reach
The slices are where the three shells cross the three world planes through the dot: nine rings that move with it.
The white line joins the hand to the dot. In --session the same sticks drive hand velocity in world axes.
Close the viewer window to end a run.

The keyboard path was checked without a human at the keys; the gamepad path is UNTESTED
(pygame is not installed here and no pad was attached). Pilot both before a real participant.
"""
import argparse
import copy
import collections
import ctypes
import itertools
import random
import threading
import time
from pathlib import Path
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot

from dynamics import kin
from schemes import NAMES, qdot
from task import run_trial, MODELS, Q_REST, Q_LIM, DECIM, QD_MAX
from schemes import MARGIN
from view import viewer_hook

V_MAX, W_MAX, DEAD = 0.08, 0.4, 0.1     # m/s, rad/s, stick deadband: tune in the pilot
G_MAX, V_GO, TOL = 0.15, 0.10, 0.005    # dot speed m/s, hand speed m/s, "arrived" m
TW_BASE, BASE_MAX = 2.0, 2.0            # hand first, base as level as the arm allows; deg of base drift that rejects a goal
# Measured, medium servicer from rest (D15): with the base held exactly level the hand has ONE good direction
# (singular values of J on the reactionless joint motions: 0.51, 0.07, 0.04), so reach depends on the drift allowed:
#   0.5 deg -> 0.14-0.34 m    1 deg -> 0.25-0.45 m    2 deg -> 0.32-0.85 m    no limit -> 0.32-0.96 m (joint limits)
FINE, CAM = 0.25, 60.0                  # L1 speed factor; camera turn deg/s
KX_GO, REACH = 20.0, 1.2                # 1/s pull near the goal (full speed until 5 mm out); m a probe ray pulls
STALL = 0.0003                          # m per 0.5 s: check() waits out the slow last stretch a probe ray gives up on
# 98 probe directions. Measured from rest, goals near the edge (D20): a 26-line cage agrees with the rehearsal
# on 31 of 42 goals, 98 lines on 37 of 42, and only 1 of its 5 misses says "in" for a goal that is refused.
DIRS = [np.array(u) for u in sorted({tuple(np.round(np.array(u) / np.linalg.norm(u), 6))
                                     for u in itertools.product((-2, -1, 0, 1, 2), repeat=3) if any(u)})]
TRIS = ConvexHull(np.array(DIRS)).simplices     # triangles joining neighbouring directions into a closed cage
EDGES = sorted({tuple(sorted((int(t[i]), int(t[i - 1])))) for t in TRIS for i in range(3)})
OUT = Path(__file__).parent.parent / "data" / "participants"
VK = dict(left=0x25, up=0x26, right=0x27, down=0x28, pgup=0x21, pgdn=0x22, home=0x24, end=0x23, enter=0x0D,
          delete=0x2E, shift=0x10, insert=0x2D, rctrl=0xA3)
LIMITS = (0.5, 1.0, 2.0)                # deg of base tilt: the three shells of the envelope
GREEN, YELLOW, RED, WHITE = [0, 1, 0, 0.6], [1, 1, 0, 0.6], [1, 0, 0, 0.8], [1, 1, 1, 0.6]
LINE = ([0.2, 1, 0.3, 1], [1, 0.55, 0.1, 1], [0.3, 0.6, 1, 1])     # slice rings: base tilt 0.5, 1, 2 deg
FENCE = (0.9, 1.1)                      # depth on the 2 deg cage: inside 0.9 the map is right 86 times in 89, past 1.1 nothing is reachable (D25)


def down(n):
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK[n]) & 0x8000)


def keyboard():
    """-> (sx, sy, depth, cam_x, cam_y, go, snap, fine), the same tuple as gamepad().
    ponytail: Windows-only and reads keys even when the viewer is not focused; use --pad elsewhere."""
    return (down("right") - down("left"), down("up") - down("down"), down("pgup") - down("pgdn"),
            down("home") - down("end"), 0, down("enter"), down("delete"), down("shift"))


keyboard.view, keyboard.park = (lambda: down("insert")), (lambda: down("rctrl"))


def gamepad():
    import pygame
    pygame.init()
    js = pygame.joystick.Joystick(0)

    def axes():   # pygame 2 / SDL2 DualShock 4 numbering: UNTESTED, print js.get_axis/get_button and fix here if off
        pygame.event.pump()
        a = [0 if abs(x) < DEAD else x for x in (js.get_axis(i) for i in range(6))]
        return a[0], -a[1], (a[5] - a[4]) / 2, a[2], a[3], js.get_button(0), js.get_button(1), js.get_button(9)
    axes.buzz = lambda x: js.rumble(x, x, 100)      # 0..1, UNTESTED like the numbering; pygame >= 2.0.2
    axes.view, axes.park = (lambda: js.get_button(2)), (lambda: js.get_button(3))     # Square, Triangle: UNTESTED
    return axes


def move(goal, sx, sy, depth, az, el, step):
    """Shift the goal by `step` metres per unit stick, in the camera's frame (MuJoCo azimuth/elevation, deg)."""
    az, el = np.radians(az), np.radians(el)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([np.sin(az), -np.cos(az), 0])
    return goal + step * (sx * right + sy * np.cross(right, fwd) + depth * fwd)


def setup(size):
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"ff_{size}.xml"))
    m.jnt_range[1:], m.jnt_limited[1:] = Q_LIM, 1
    d = mujoco.MjData(m)
    d.qpos[7:] = Q_REST
    mujoco.mj_forward(m, d)
    return m, d, m.site("ee").id, Rot.from_quat(d.qpos[[4, 5, 6, 3]])


def ctrl(m, d, sid, Rb0, target, scheme, mode):
    """One control tick toward target, base attitude first. -> hand position, base drift in deg."""
    mujoco.mj_forward(m, d)
    p = d.site_xpos[sid].copy()
    e = target - p
    v = e * min(KX_GO, V_GO / (np.linalg.norm(e) + 1e-9))
    att = (Rb0.inv() * Rot.from_quat(d.qpos[[4, 5, 6, 3]])).as_rotvec()
    qd = qdot(scheme, kin(m, d), np.r_[v, 0, 0, 0], d.qpos[7:].copy(), Q_REST, mode, 10.0, None, att, Q_LIM, TW_BASE)
    d.ctrl[:] = qd / max(1.0, np.abs(qd).max() / QD_MAX)
    return p, np.degrees(np.linalg.norm(att))


MARGIN_Y, MARGIN_R = 0.24, 0.05         # singularity margin where the hand ball turns yellow, red: the median at first tilt and the
                                        # 90th percentile at 0.5 deg over 98 probes from the ready pose (D30); checked from other poses in analysis/envelope/margin.py


def sig6(m, d):
    """How solvable 'move the hand, keep the base level' still is: the smallest singular value of S = [J; W]
    (6 equations) over the joints not at a limit. 0 = no reactionless move left. Needs no envelope and no rest pose."""
    k, q = kin(m, d), d.qpos[7:]
    free = np.minimum(q - Q_LIM[:, 0], Q_LIM[:, 1] - q) > MARGIN + 1e-3
    return np.linalg.svd(np.vstack([k.Jstar[:3], k.W])[:, free], compute_uv=False)[5] if free.sum() >= 6 else 0.0


CLEAR = 0.02                            # m the arm must keep from the bus. ponytail: the bus box only, no arm-on-arm check


def clear(m, d):
    """Least distance from the arm to the bus, m (negative: inside it). Call after mj_forward.
    Link 1 is bolted to the bus and left out; geom 0 is the bus box."""
    return min(mujoco.mj_geomDistance(m, d, 0, g, 1.0, None) for g in range(m.ngeom) if m.geom_bodyid[g] > 2)


def reachable(m, d, sid, Rb0, target, scheme, mode):
    """Rehearse the move on a copy of the simulation: the arrival state if the hand gets there with the
    base level, else None.
    ponytail: waits out a timeout on a far goal (a few seconds frozen); add stall detection if that annoys."""
    d2 = copy.copy(d)
    n = int((2 * np.linalg.norm(target - d.site_xpos[sid]) / V_GO + 3) / m.opt.timestep)
    for i in range(n):
        if i % DECIM == 0:
            p, base = ctrl(m, d2, sid, Rb0, target, scheme, mode)
            if base > BASE_MAX:
                return None
            if np.linalg.norm(target - p) < TOL:
                return d2
        mujoco.mj_step(m, d2)
    return None


def ray(m, d, sid, Rb0, u, scheme, mode, reach=REACH, stall=0.002, marks=None):
    """Pull the hand along u on a copy of the simulation. -> the furthest point it got to with the base level.
    marks: a list that gets the last point inside each of LIMITS, as the base passes them."""
    d2 = copy.copy(d)
    p0 = best = d.site_xpos[sid].copy()
    last = 0.0
    for i in range(int((2 * reach / V_GO + 1) / m.opt.timestep)):
        if i % DECIM == 0:
            p, base = ctrl(m, d2, sid, Rb0, p0 + reach * u, scheme, mode)
            while marks is not None and len(marks) < len(LIMITS) and base > LIMITS[len(marks)]:
                marks.append(best)
            if base > BASE_MAX or clear(m, d2) < CLEAR:    # the arm may not pass through the bus either
                break
            best = p
            if i and i % (50 * DECIM) == 0:          # every 0.5 s: stalled if under `stall` m of progress
                if (p - p0) @ u - last < stall:
                    break
                last = (p - p0) @ u
        mujoco.mj_step(m, d2)
    return best


def envelope(m, d0, sid, Rb0, scheme, mode, s):
    """Work envelope from state d0: along 26 directions, where the base passes each of LIMITS and where the hand
    stalls. s["rays"] gets one (3, 3) array per direction, inner shell first; the viewer joins them into three
    nested cages. Every point on a probe line was visited; between lines a cage is a straight join.
    A probe also ends where the arm would come within CLEAR of the bus.
    Drawn once, from rest: a guide, check() stays the authority (D19, D20).
    ponytail: a background thread, about 85 s for all 98; cache to disk if the wait annoys."""
    out = s["rays"] = []

    def work():
        for u in DIRS:
            mk = []
            best = ray(m, d0, sid, Rb0, u, scheme, mode, marks=mk)
            out.append(np.array((mk + [best] * len(LIMITS))[:len(LIMITS)]))
    threading.Thread(target=work, daemon=True).start()


def depth(tips, p0, g):
    """How far out g is: 0 at p0, 1 on the cage through tips (26 points), along the line from p0."""
    for t in TRIS:
        A = np.column_stack([tips[i] - p0 for i in t])
        if abs(np.linalg.det(A)) > 1e-12:
            c = np.linalg.solve(A, g - p0)      # g = p0 + c . (the three corners of this face)
            if (c >= -1e-9).all():
                return c.sum()
    return np.inf


def cut(tips, o, n):
    """Where the cage through tips crosses the plane through o with normal n. -> (k, 2, 3) line segments."""
    P = np.asarray(tips)[TRIS]
    h = (P - o) @ n
    h = np.where(h == 0, 1e-12, h)
    out = []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):      # corner a alone on its side of the plane
        i = (h[:, a] * h[:, b] < 0) & (h[:, a] * h[:, c] < 0)
        A, ha = P[i, a], h[i, a]
        out.append(np.stack([A + (P[i, x] - A) * (ha / (ha - h[i, x]))[:, None] for x in (b, c)], 1))
    return np.concatenate(out)


def zone(k):
    """Geofence zone of a goal from its depth on the 2 degree cage: 0 free, 1 caution, 2 past the fence."""
    return int(k >= FENCE[0]) + int(k > FENCE[1])


def tilt(rays, p0, g):
    """Base tilt a goal will cost, read off the shells before the move."""
    if len(rays) < len(DIRS):
        return "not known yet (envelope still being drawn)"
    for i, lim in enumerate(LIMITS):
        if depth([r[i] for r in rays], p0, g) < 1:
            return f"under {lim:g} deg"
    return f"over {LIMITS[-1]:g} deg"


def check(m, d, sid, Rb0, goal, scheme, mode):
    """Can the hand get from where it is to goal in one straight move with the base within 2 degrees?
    -> (yes/no, the furthest point it gets to on the way)."""
    e = goal - d.site_xpos[sid]
    dist = np.linalg.norm(e)
    far = ray(m, d, sid, Rb0, e / dist, scheme, mode, dist, STALL) if dist > TOL else goal
    return np.linalg.norm(far - goal) < TOL, far


def preview(m, sid, Rb0, scheme, mode, s):
    """Background thread: keeps answering check() for the newest dot position (s["ask"] -> s["ans"])."""
    while True:
        a = s.pop("ask", None)
        if a is None:
            time.sleep(0.02)
        else:
            s["ans"] = check(m, a[1], sid, Rb0, a[0], scheme, mode)


TARGETS = np.array([[0, .2, 0], [0, .2, .2], [.1, 0, .2], [0, -.15, .1], [0, 0, 0]])   # pilot goals, from the starting hand position
T_MOVE = 30.0                           # s a move may take before the fence calls it back. 41 reachable moves from rest took at most 8.8 s (analysis/envelope/fence.py)
                                        # ponytail: one number, 3x the slowest from-rest move, to leave room for a move across the cage; scale with distance if it trips on good moves
T_REST = 13.1                           # the same limit for the feedback study, where every move starts from rest: 1.5 x the slowest of those 41
NEAR, DWELL, T_OUT = 0.02, 0.5, 90.0    # a pilot goal is reached after DWELL s within NEAR m; given up after T_OUT s


def free_drive(axes, size, scheme, mode, delay=0.0, direct=False, targets=None, aid=None):
    """delay: seconds before a command reaches the robot (the round trip, lumped on the uplink).
    direct: rate control, the stick drives the hand at up to V_GO while held; no dot, envelope or rehearsal.
    targets: pilot goals to touch in turn. -> one row per goal.
    aid: feedback study, one of CUES. Every go is sent (no rehearsal), the runtime fence is the only gate, each goal
    starts from rest, and Right Ctrl (pad: Triangle) gives a goal up as out of reach.
    ponytail: with a delay the rehearsal is of the state at the key press, not at arrival of the command."""
    m, d, sid, Rb0 = setup(size)
    s, hook = viewer_hook()
    p = goal = home = d.site_xpos[sid].copy()
    s0 = sig6(m, d)
    target, was, peak, dt = goal.copy(), False, 0.0, DECIM * m.opt.timestep
    pend, vel, rows = collections.deque(), np.zeros(3), []
    todo = [goal + t for t in targets] if targets is not None else []
    t0, sends, refused, path, gpeak, dwell = 0.0, 0, 0, 0.0, 0.0, 0.0
    start, tgo, kz, buzz = home, 0.0, 0, getattr(axes, "buzz", lambda x: None)
    view, park, wasv, wasp, trips, cutat, stop = getattr(axes, "view", bool), getattr(axes, "park", bool), False, False, 0, None, aid in (None, "stop")
    if direct:
        s["rgba"] = [0, 0, 0, 0]
    else:
        s["rgba"], s["ask"], s["show"] = (WHITE if aid else YELLOW), (goal.copy(), copy.copy(d)), 0
        if aid != "none":
            s["cage"] = (TRIS, EDGES)
        f = OUT.parent / f"decide4_{size}.npz"
        if f.exists() and (scheme, mode) == (5, "3d"):     # the cached cage: the fence is live from the first second,
            s["rays"] = list(np.load(f)["rays"])            # not after the 85 s the probe moves take
        else:
            envelope(m, copy.copy(d), sid, Rb0, scheme, mode, s)
        if not aid:
            threading.Thread(target=preview, args=(m, sid, Rb0, scheme, mode, s), daemon=True).start()
    while True:
        sx, sy, dz, cx, cy, go, snap, fine = axes()
        cam = s["v"].cam if "v" in s else None      # the viewer opens on the first hook call
        if cam:
            cam.azimuth -= CAM * dt * cx
            cam.elevation = np.clip(cam.elevation - CAM * dt * cy, -89, 89)
        az, el = (cam.azimuth, cam.elevation) if cam else (90, -45)
        if direct:
            pend.append((d.time + delay, move(np.zeros(3), sx, sy, dz, az, el, FINE if fine else 1.0)))
        else:
            vw = view()                             # Insert: slices through the dot / slices and shells / shells
            if vw and not wasv:
                s["show"] = (s["show"] + 1) % 3
            wasv = vw
            if snap:
                goal = d.site_xpos[sid].copy()
            if sx or sy or dz:
                goal = move(goal, sx, sy, dz, az, el, G_MAX * dt * (FINE if fine else 1))
            if (snap or sx or sy or dz) and len(s["rays"]) == len(DIRS):    # the fence needs the whole cage
                k = depth([r[-1] for r in s["rays"]], home, goal)
                if stop and k > FENCE[1]:           # hard fence on the dot: it slides along the outer zone, never past it
                    goal, k = home + (goal - home) * FENCE[1] / k, FENCE[1]
                kz = zone(k)
                if stop:
                    buzz(float(np.clip((k - FENCE[0]) / (FENCE[1] - FENCE[0]), 0, 1)))   # the pad shakes harder toward the fence
            if snap or sx or sy or dz:
                s["ask"] = (goal.copy(), copy.copy(d))      # the copy is made here: the worker must not read live data
            if go and not was:
                t = time.time()
                if not aid:                         # the study sends every go: the runtime fence is its only gate
                    s["ans"] = check(m, d, sid, Rb0, goal, scheme, mode)
                ok = bool(aid) or s["ans"][0]
                if "t0" in s:
                    s["t0"] += time.time() - t      # do not fast-forward the viewer after the rehearsal
                sends, refused = sends + 1, refused + (not ok)
                if ok:
                    pend.append((d.time + delay, goal.copy()))
                if not aid:
                    print("going, base tilt " + tilt(s["rays"], home, goal) if ok else "out of reach", flush=True)
            was = go
            if aid:
                s["rgba"] = WHITE if aid == "none" else (GREEN, YELLOW, RED)[kz]
            elif "ans" in s:
                s["rgba"], s["far"] = ((YELLOW if kz else GREEN), None) if s["ans"][0] else (RED, s["ans"][1])   # yellow: reachable, but close to the edge
            if "cage" in s and goal is not cutat and len(s["rays"]) == len(DIRS):
                # three rings per plane, on the three world planes through the dot: where the dot sits in each shell
                s["cuts"], cutat = [(cut([r[i] for r in s["rays"]], goal, n), c) for n in np.eye(3) for i, c in enumerate(LINE)], goal
        while pend and pend[0][0] <= d.time:        # commands reach the robot `delay` seconds late
            x = pend.popleft()[1]
            if direct:
                vel = x
            else:
                start, target, tgo = p, x, d.time
        if direct:
            target = p + vel * V_GO / KX_GO         # ctrl turns this into a hand speed of V_GO * stick
        q = p
        p, base = ctrl(m, d, sid, Rb0, target, scheme, mode)
        if not direct and target is not start and (base > BASE_MAX or clear(m, d) < CLEAR or (d.time - tgo > (T_REST if aid else T_MOVE) and np.linalg.norm(p - target) > TOL)):
            # fence: the rehearsal passed this move and the base still went over, or the hand is still short after T_MOVE
            # (the model was wrong). Back the way it came.
            print(f"base at {base:.2f} deg, arm {clear(m, d) * 100:.0f} cm from the bus after {d.time - tgo:.0f} s: going back", flush=True)
            target, trips = start, trips + 1
        if not aid:                                 # live singularity margin on the hand: right from any pose, unlike the cage
            mg = sig6(m, d) / s0
            s["hand"], s["text"] = (p, (GREEN, YELLOW, RED)[int(mg < MARGIN_Y) + int(mg < MARGIN_R)]), f"{mg:.2f}   base {base:.2f} deg"
        if direct:
            goal = p
        peak, gpeak, path, s["link"] = max(peak, base), max(gpeak, base), path + np.linalg.norm(p - q), (p, goal)
        if todo:
            s["tgt"] = todo[0]
            dwell = dwell + dt if np.linalg.norm(p - todo[0]) < NEAR else 0.0
            pk = park()
            gone, wasp = bool(aid) and pk and not wasp, pk
            if dwell >= DWELL or gone or d.time - t0 > T_OUT:
                rows.append({"goal": len(rows) + 1, "reached": dwell >= DWELL, "time_s": d.time - t0 + REPARK_S * gone,
                             "sends": sends, "refused": refused, "path_m": path, "peak_base_deg": gpeak, "parked": gone,
                             "trips": trips})
                todo.pop(0)
                t0, sends, refused, path, gpeak, dwell, trips = d.time, 0, 0, 0.0, 0.0, 0.0, 0
                if not todo:
                    break
                if aid:                             # every study goal starts from rest, as the pool's truth was measured
                    mujoco.mj_resetData(m, d)
                    d.qpos[7:] = Q_REST
                    mujoco.mj_forward(m, d)
                    p = goal = target = start = home
                    t0, tgo, kz, s["t0"] = 0.0, 0.0, 0, time.time()
                    pend.clear()
        if hook(m, d, (goal,)) is False:
            break
        for _ in range(DECIM):
            mujoco.mj_step(m, d)
    s["v"].close()
    print(f"peak base drift {peak:.2f} deg")
    return rows


def pilot(axes, pid, size):
    """Delay pilot: rate control against set-and-go, each with no delay and with 2 s, the same five goals.
    ponytail: same goals in every block, so block order (shuffled per participant) carries a learning effect."""
    order = [(direct, delay) for direct in (True, False) for delay in (0.0, 2.0)]
    random.Random(pid).shuffle(order)
    rows = []
    for n, (direct, delay) in enumerate(order, 1):
        how = "hold the keys to drive the hand" if direct else "place the dot, Enter to send"
        input(f"Block {n}/4: {how}. Hold the hand on each pink ball. Enter to start.")
        r = free_drive(axes, size, 5, "3d", delay, direct, TARGETS)
        effort = int(input("Effort 1-7: "))
        rows += [x | {"pid": pid, "block": n, "direct": direct, "delay_s": delay, "effort_1to7": effort} for x in r]
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / f"delay_{pid}.csv", index=False)   # saved after every block


AIDS = ("none", "binary", "graded", "map")  # decision study: no help, go / no-go, the same plus "close to the edge", the envelope drawn
BINS = (0.5, 0.8, 0.95, 1.05, 1.2, 1.4)     # how far out a goal is, 1 = the 2 degree edge: deep, inside, edge, outside, far
PER_BIN, CHECK_S = 3, 8.0                   # goals per bin in each block; seconds a rehearsal takes
P_GO, P_PARK, P_CHECK = -20, -5, -1         # points: a go that fails, a re-park (needed or not), a rehearsal. A go that works is free.
# ponytail: these three numbers are a design choice (a failed go is 4 re-parks), not mission costs. With them a rehearsal
# pays only within 10% of the edge, on both sides (D25): 8 of 60 go there fail, 27 of 67 no-go there are reachable.


def pool(size, n=240, seed=0):
    """Goals for the decision study, from deep inside the envelope to well outside it, with the aid's answer
    (the tilt shell the goal lies in, read off the envelope drawn from rest; 3 = outside, no-go) and the truth
    (a rehearsal of the move). Cached: a few minutes the first time. One set of goals per kind of help, each with
    PER_BIN goals per depth bin. The aid's errors are whatever falls out: none are planted.
    -> rays, goals (set, goal, 3), depth, band, truth."""
    f = OUT.parent / f"decide4_{size}.npz"
    if f.exists():
        z = np.load(f)
        return z["rays"], z["goals"], z["depth"], z["band"], z["truth"]
    m, d, sid, Rb0 = setup(size)
    p0, rng, rays = d.site_xpos[sid].copy(), np.random.default_rng(seed), []
    for u in DIRS:
        mk = []
        best = ray(m, d, sid, Rb0, u, 5, "3d", marks=mk)
        rays.append(np.array((mk + [best] * len(LIMITS))[:len(LIMITS)]))
    sh = [[r[k] for r in rays] for k in range(len(LIMITS))]
    U = rng.normal(size=(n, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    dep = rng.uniform(BINS[0], BINS[-1], n)
    g = np.array([p0 + u * k / depth(sh[-1], p0, p0 + u) for u, k in zip(U, dep)])
    band = np.array([next((k for k in range(len(LIMITS)) if depth(sh[k], p0, x) < 1), len(LIMITS)) for x in g])
    truth = np.array([check(m, d, sid, Rb0, x, 5, "3d")[0] for x in g])
    k, b = len(AIDS), np.digitize(dep, BINS) - 1
    assert all((b == x).sum() >= k * PER_BIN for x in range(len(BINS) - 1)), np.bincount(b)
    i = np.array([np.concatenate([np.flatnonzero(b == x)[j * PER_BIN:(j + 1) * PER_BIN] for x in range(len(BINS) - 1)])
                  for j in range(k)])
    np.savez(f, rays=np.array(rays), goals=g[i], depth=dep[i], band=band[i], truth=truth[i])
    return np.array(rays), g[i], dep[i], band[i], truth[i]


def decide(pid, size):
    """Decision study: the arm is parked, a pink ball is the grapple point. Send the arm (g), re-park the
    servicer (n), or pay CHECK_S seconds for a rehearsal that gives the true answer (c) and then decide.
    Nothing moves. One block per kind of help; which goal set goes with which help rotates with the participant id.
    The question is where people spend rehearsals: an aid that shows the margin should send them to the edge.
    ponytail: no delay, a wrong go costs nothing but the score, and feedback comes only through rehearsals."""
    rays, goals, dep, band, truth = pool(size)
    m, d, sid, Rb0 = setup(size)
    s, hook = viewer_hook()
    s["rgba"] = [0, 0, 0, 0]
    order, rows, per = list(range(len(AIDS))), [], goals.shape[1]
    random.Random(pid).shuffle(order)
    for b, c in enumerate(order, 1):
        k = (c + pid) % len(AIDS)
        s.pop("cage", None)
        if AIDS[c] == "map":
            s["cage"], s["rays"] = (TRIS, EDGES), list(rays)
        input(f"Block {b}/{len(AIDS)}, help: {AIDS[c]}. g = send the arm (free if it gets there, {P_GO} if not), "
              f"n = re-park ({P_PARK}), c = rehearse ({P_CHECK}, {CHECK_S:g} s). Enter to start.")
        for j in random.Random(pid * 10 + c).sample(range(per), per):
            s["tgt"] = goals[k, j]
            hook(m, d, (goals[k, j],))
            t, go = time.time(), band[k, j] < len(LIMITS)
            say = {"binary": "go" if go else "no-go",
                   # graded by margin, not tilt: every aid error in the pool is a no-go just outside the edge (D24)
                   "graded": ("go" if go else "no-go") + (", close to the edge" if abs(dep[k, j] - 1) < 0.1 else "")}.get(AIDS[c])
            ans, checked = "", False
            while ans not in ("g", "n"):
                ans = input(f"Goal {len(rows) + 1}." + (f" The aid says {say}." if say else "") + " g/n/c: ").strip().lower()
                if ans == "c" and not checked:
                    time.sleep(CHECK_S)
                    checked = True
                    print("Rehearsal:", "it gets there." if truth[k, j] else "it does not get there.", flush=True)
            pts = P_CHECK * checked + (P_PARK if ans == "n" else 0 if truth[k, j] else P_GO)
            if ans == "g":                              # you only find out by going: a re-park teaches nothing
                print("It got there." if truth[k, j] else f"The base tilted past the limit: {P_GO}.", flush=True)
            rows.append({"pid": pid, "block": b, "help": AIDS[c], "set": k, "goal": j, "depth": dep[k, j], "points": pts,
                         "aid_band": int(band[k, j]), "aid_go": bool(go), "true_go": bool(truth[k, j]), "checked": checked,
                         "answer_go": ans == "g", "correct": (ans == "g") == truth[k, j], "time_s": time.time() - t})
            OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT / f"decide_{pid}.csv", index=False)   # saved after every goal
    s["v"].close()
    return rows


CUES = ("none", "colour", "stop")       # feedback study: bare dot / envelope and zone colour on the dot / the same plus the hard stop (and rumble)
REPARK_S = 30.0                         # s charged for giving a goal up. ponytail: a design choice, not a mission cost; a wrong go costs its real ~15-25 s


def feedback(axes, pid, size):
    """Feedback study: put the hand on each pink ball, or give it up as out of reach (Right Ctrl). The score is time.
    Three blocks, one per cue; which goal set goes with which cue rotates with the participant id.
    Nothing is rehearsed: a go that cannot work costs the time the fence takes to bring the arm back."""
    _, goals, dep, _, truth = pool(size)
    m, d, sid, _ = setup(size)
    home, order, rows = d.site_xpos[sid].copy(), list(range(len(CUES))), []
    random.Random(pid).shuffle(order)
    for b, c in enumerate(order, 1):
        k, j = (c + pid) % len(goals), random.Random(pid * 10 + c).sample(range(goals.shape[1]), goals.shape[1])
        input(f"Block {b}/{len(CUES)}, cue: {CUES[c]}. Dot on the pink ball, Enter to send; Right Ctrl = out of reach "
              f"(+{REPARK_S:g} s). Fastest total wins. Enter to start.")
        r = free_drive(axes, size, 5, "3d", targets=goals[k, j] - home, aid=CUES[c])
        rows += [x | {"pid": pid, "block": b, "cue": CUES[c], "set": k, "goal": i, "depth": dep[k, i], "true_go": bool(truth[k, i])}
                 for x, i in zip(r, j)]
        print(f"block time {sum(x['time_s'] for x in r):.0f} s", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / f"feedback_{pid}.csv", index=False)   # saved after every block
    return rows


def operator(axes):
    def op(t, p, R, ref):
        sx, sy, depth, tilt = axes()[:4]
        return np.r_[V_MAX * depth, V_MAX * sx, V_MAX * sy, W_MAX * tilt, 0, 0]
    return op


def session(axes, pid, size):
    order = [(s, mode) for s in range(1, 6) for mode in ("3d", "6d")]
    random.Random(pid).shuffle(order)
    rows = []
    for n, (scheme, mode) in enumerate(order, 1):
        input(f"Trial {n}/{len(order)} ({mode.upper()} task): follow the green dot. Enter to start.")
        s, hook = viewer_hook()
        r = run_trial(size, scheme, mode, loops=1, operator=operator(axes), on_step=hook)
        s["v"].close()
        r |= {"pid": pid, "trial": n, "predictability_1to7": int(input("Predictability 1-7: "))}
        rows.append(r)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / f"P{pid}.csv", index=False)   # saved after every trial


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", action="store_true")
    ap.add_argument("--session", type=int)
    ap.add_argument("--size", default="medium")   # small: level-base reach is only ~5 cm (D14)
    ap.add_argument("--scheme", type=int, default=5)
    ap.add_argument("--mode", default="3d")
    ap.add_argument("--delay", type=float, default=0.0)   # seconds before a command reaches the robot
    ap.add_argument("--direct", action="store_true")      # rate control instead of set-and-go
    ap.add_argument("--pilot", type=int)                  # participant id: 4 blocks, direct/set-and-go x 0/2 s
    ap.add_argument("--feedback", type=int)               # participant id: reach the goals with no cue / colour / colour and hard stop
    ap.add_argument("--decide", type=int)                 # participant id: go / re-park decisions, 3 kinds of help
    a = ap.parse_args()
    axes = gamepad() if a.pad else keyboard
    if a.feedback is not None:
        feedback(axes, a.feedback, a.size)
    elif a.decide is not None:
        decide(a.decide, a.size)
    elif a.pilot is not None:
        pilot(axes, a.pilot, a.size)
    elif a.session is not None:
        session(axes, a.session, a.size)
    else:
        print(f"{NAMES[a.scheme]}, {a.size} servicer. Move the dot, Enter to send. Close the window to stop.", flush=True)
        free_drive(axes, a.size, a.scheme, a.mode, a.delay, a.direct)
