"""Alin's Gen3 servicers dressed as satellites, and the videos for the dynamics talk, in the talk's order. Owner: P2.

    python -m analysis.satellite [out_dir]     # default figures/satellite/

models/ is not changed. The look is added at load time and is paint only: solar wings on the +y and -y faces, a dish
on the antenna face, the Earth on the antenna's boresight, a sun. Every added geom has no mass and no contact, so the
physics is the same to the bit (tests/test_satellite.py). The bus starts turned so the arm points up and the antenna
at the Earth; gravity is off and momentum starts at zero, so that changes no number. The hand site is read as "ee",
the name our v1 code uses.

  orbit.mp4   opening: the camera goes round the 150 kg servicer at rest
  spin.mp4    slide 1 (dynamics): 20 kg bus, only joint 1 turns, 90 deg in 5 s, the rest held; the tilt passes the
              2 deg budget
  sizes.mp4   slide 1, its table: the same turn on the 20, 150 and 1500 kg buses
  map.mp4     slide 3 (map): side and top views, straight lines grow from the hand at rest, coloured by the bus tilt
              in 0.25 deg steps, a solid end where the bus passed 2 deg, dashed where the arm ran out of reach
  gap.mp4     slide 4 (gap): top view. The map from rest, the first move (the hand 0.3 m sideways), then the rest map
              turns dashed, the reach from the new state is drawn solid, orange where the map says go and the hand
              can't get there now, blue where it says no and the hand now can
Map and gap use our controller (v1 scheme 5, 3-D, our constants, not tuned for the Gen3: the deck's first run) on the
150 kg servicer, a line every 2.5 deg, as the deck's map run did.
"""
import copy
import functools
import io
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "v1" / "src"))
import teleop as T                                   # noqa: E402  our controller (v1, scheme 5, 3-D)

W, H, FPS = 1280, 720, 30
TICK = 40                                            # physics steps per control tick: 100 Hz, as in the study

unit = lambda *a: np.array(a, float) / np.linalg.norm(a)
EARTH = unit(1, 0, -.18)                             # the antenna's boresight at the start, 10 deg down
UP = unit(.18, 0, 1)                                 # the arm's axis at q = 0
R0 = np.column_stack([UP, np.cross(EARTH, UP), EARTH])   # bus x, y, z in the world at the start
Q0 = np.zeros(4)
mujoco.mju_mat2Quat(Q0, R0.ravel())
SUN = unit(-.35, 0, 1)                               # square to the wings' axis (bus y = world -y): both face it fully
EARTH_D, EARTH_DEG = 25.0, 5.0                       # m, and the Earth's angular radius in degrees
NOHIT = 'mass="0" contype="0" conaffinity="0" rgba=".5 .5 .5 1"'   # this rgba hands the colour to the material
BUDGET = 2.0                                         # deg of tilt, the budget in the talk
RAMP = ["#DCE7F3", "#C0D0E5", "#A4BAD6", "#89A4C7", "#6E8FB9", "#5479AA", "#3A649B", "#1F4F8C"]  # the deck's map, 0.25 deg steps
RAMP = [np.r_[[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], 1] for h in RAMP]
RED, WHITE = (255, 90, 80), (255, 255, 255)

SKY = """<asset>
<texture name="sky" type="skybox" builtin="gradient" rgb1=".03 .04 .09" rgb2="0 0 0" width="512" height="3072" mark="random" random=".0015" markrgb=".9 .9 1"/>
<texture name="cells" type="2d" builtin="checker" rgb1=".1 .13 .3" rgb2=".07 .09 .22" mark="edge" markrgb=".7 .7 .75" width="64" height="64"/>
<texture name="earth" type="2d" file="earth.png"/>
<material name="foil" rgba=".5 .38 .16 1" specular=".4" shininess=".5"/>
<material name="cells" texture="cells" texrepeat="5 5" texuniform="true" specular=".8" shininess=".9"/>
<material name="metal" rgba=".6 .62 .66 1" specular=".7" shininess=".6"/>
<material name="white" rgba=".9 .9 .92 1" specular=".4"/>
<material name="dark" rgba=".16 .16 .18 1"/>
<material name="earth" texture="earth" specular=".15" shininess=".2"/>
<material name="air" rgba=".5 .72 1 .16"/>
<material name="sun" rgba="1 .96 .85 1" emission="1"/>
{globe}
</asset>
"""


def v(a):
    return " ".join(f"{x:.4g}" for x in np.ravel(a))


def globe(R, nlon=48, nlat=24):
    """A UV sphere of radius R with texture coordinates for an equirectangular map. The seam column is doubled so the
    map wraps; the triangles that would collapse at the poles are left out."""
    lat = np.pi / 2 - np.arange(nlat + 1) / nlat * np.pi
    lon = np.arange(nlon + 1) / nlon * 2 * np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    p = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    uv = np.stack(np.meshgrid(np.arange(nlat + 1) / nlat, np.arange(nlon + 1) / nlon, indexing="ij")[::-1], -1).reshape(-1, 2)
    f = []
    for i in range(nlat):
        for j in range(nlon):
            a, b = i * (nlon + 1) + j, (i + 1) * (nlon + 1) + j
            if i < nlat - 1:
                f.append((a, b, b + 1))
            if i > 0:
                f.append((a, b + 1, a + 1))
    s = v(np.round(p, 4))
    return (f'<mesh name="globe" scale="{R:.4g} {R:.4g} {R:.4g}" vertex="{s}" normal="{s}" '
            f'texcoord="{v(np.round(uv, 4))}" face="{" ".join(map(str, np.ravel(f)))}"/>')


@functools.lru_cache
def earth_png(w=1024, seed=7):
    """A made-up Earth as PNG bytes, from 3-D value noise on the sphere so it has no seam: sea, land coloured by a
    moisture field with a desert band, ice caps, clouds. It only has to read as Earth."""
    rng = np.random.default_rng(seed)
    P, V = rng.permutation(256), rng.random(256)

    def noise(q):
        i = np.floor(q).astype(np.int64)
        t = q - i
        t = t * t * (3 - 2 * t)
        out = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    h = V[P[(P[(P[(i[:, 0] + a) & 255] + i[:, 1] + b) & 255] + i[:, 2] + c) & 255]]
                    out = out + h * (t[:, 0] if a else 1 - t[:, 0]) * (t[:, 1] if b else 1 - t[:, 1]) * (t[:, 2] if c else 1 - t[:, 2])
        return out

    def fbm(q, f0, n, off):
        return sum(.5 ** o * noise(q * f0 * 2 ** o + off + 7.31 * o) for o in range(n)) / (2 - .5 ** (n - 1))

    h = w // 2
    lat = np.pi / 2 - (np.arange(h) + .5) / h * np.pi
    lon = (np.arange(w) + .5) / w * 2 * np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    d = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    z = np.abs(d[:, 2])
    e = fbm(d, 1.4, 7, 0.)
    coast = e - np.quantile(e, .7)                                       # 30% land
    mix = lambda a, b, s: np.asarray(a) * (1 - s[:, None]) + np.asarray(b) * s[:, None]
    sea = mix((.03, .1, .32), (.07, .3, .55), np.clip(1 + coast / .06, 0, 1))
    m = fbm(d, 2.5, 5, 50.)
    land = mix((.12, .29, .1), (.32, .42, .18), np.clip((m - .4) * 4, 0, 1))
    land = mix(land, (.72, .6, .4), np.clip((.52 - m) * 5, 0, 1) * np.exp(-((z - .38) / .16) ** 2))
    land = mix(land, (.45, .4, .33), np.clip((coast - .1) / .1, 0, 1) * .7)
    col = np.where((coast > 0)[:, None], land, sea)
    col = mix(col, (.92, .94, .97), (z > .86 + .08 * (fbm(d, 4., 3, 90.) - .5)).astype(float))
    cl = fbm(d * (1, 1, 1.7), 2.2, 6, 200.)
    col = mix(col, (.96, .97, .98), np.clip((cl - .5) / .2, 0, 1) ** 1.3 * .92)
    buf = io.BytesIO()
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)).save(buf, "png")
    return buf.getvalue()


def paint(hx, hy, hz):
    """On the bus, in its frame: a dark plate behind the arm, two wings of three panels on the +y and -y faces turned
    square to the sun, a dish on the antenna face (+z) along the boresight. Sizes go with the bus."""
    sun = R0.T @ SUN
    L, Wd, rd = 3 * hy, 1.2 * hy, .3 * hy           # wing length and width, dish radius
    y0 = 1.3 * hy                                    # where a wing starts, past its yoke
    a = np.array([0, 1., 0])
    b = np.cross(sun, a)                             # across the wing; the panel normal a x b is the sun
    z0, z1 = hz, hz + .5 * rd + .02                  # dish mast
    g = [f'type="box" pos="{hx + .004:.4g} 0 0" size=".004 {.9 * hy:.4g} {.9 * hz:.4g}" material="dark"']
    for s in (1, -1):
        g.append(f'type="cylinder" fromto="0 {s * hy:.4g} 0 0 {s * (y0 + .01):.4g} 0" size="{.045 * hy:.4g}" material="metal"')
        for k in range(3):
            g.append(f'type="box" pos="0 {s * (y0 + (k + .5) * L / 3):.4g} 0" size="{L / 6 - .01:.4g} {Wd / 2:.4g} .008" '
                     f'xyaxes="{v(a)} {v(b)}" material="cells"')
    g += [f'type="cylinder" fromto="0 0 {z0:.4g} 0 0 {z1:.4g}" size="{.12 * rd:.4g}" material="metal"',
          f'type="ellipsoid" pos="0 0 {z1 + .2 * rd:.4g}" size="{rd:.4g} {rd:.4g} {.2 * rd:.4g}" material="white"',
          f'type="cylinder" fromto="0 0 {z1 + .38 * rd:.4g} 0 0 {z1 + .39 * rd:.4g}" size="{.85 * rd:.4g}" material="metal"',
          f'type="cylinder" fromto="0 0 {z1 + .39 * rd:.4g} 0 0 {z1 + 1.1 * rd:.4g}" size="{.03 * rd:.4g}" material="metal"',
          f'type="sphere" pos="0 0 {z1 + 1.1 * rd:.4g}" size="{.08 * rd:.4g}" material="dark"']
    return "".join(f'\n<geom {NOHIT} {x}/>' for x in g)


def sky():
    """The Earth with a thin air shell, and the sun's disc, on a body fixed to the world: they stay put while the bus
    turns, so a tilt shows against them. The globe is tipped 23 degrees, like the real one."""
    R, C = EARTH_D * np.sin(np.radians(EARTH_DEG)), EARTH_D * EARTH
    return (f'\n<body name="sky">'
            f'\n<geom {NOHIT} type="mesh" mesh="globe" pos="{v(C)}" euler="23 0 45" material="earth"/>'
            f'\n<geom {NOHIT} type="sphere" pos="{v(C)}" size="{1.025 * R:.4g}" material="air"/>'
            f'\n<geom {NOHIT} type="sphere" pos="{v(45 * SUN)}" size=".7" material="sun"/></body>'
            f'\n<light name="sun" directional="true" pos="0 0 6" dir="{v(-SUN)}" diffuse=".6 .58 .52" specular=".4 .4 .4" castshadow="false"/>'
            f'\n<light name="key" pos="0 -4 4" dir="0 1 -1" diffuse=".5 .5 .5"/>\n')


def load(size, look=True, earth_w=1024):
    """models/ff_<size>.xml, dressed when look is True. The dressed model keeps the bare one's statistics, which
    the solver and the framing read."""
    text = (MODELS / f"ff_{size}.xml").read_text()
    assets = {f"common/{p.name}": p.read_bytes() for p in (MODELS / "common").glob("*.xml")}
    assert assets["common/arm_body.xml"].count(b'name="end_effector"') == 1
    assets["common/arm_body.xml"] = assets["common/arm_body.xml"].replace(b'name="end_effector"', b'name="ee"')
    bare = mujoco.MjModel.from_xml_string(text, assets)
    if not look:
        return bare
    R = EARTH_D * np.sin(np.radians(EARTH_DEG))
    text = text.replace("<worldbody>", SKY.format(globe=globe(R)) + "<worldbody>", 1)
    text, n = re.subn(r'(<geom name="base_geom"[^>]*?)\s*/>',
                      lambda mt: mt.group(1) + ' material="foil" rgba=".5 .5 .5 1"/>' + paint(*bare.geom("base_geom").size),
                      text, count=1)
    assert n == 1, "base_geom not found"
    text = text.replace("</worldbody>", sky() + "</worldbody>", 1)
    m = mujoco.MjModel.from_xml_string(text, {**assets, "earth.png": earth_png(earth_w)})
    for f in ("center", "extent", "meansize", "meanmass", "meaninertia"):
        setattr(m.stat, f, getattr(bare.stat, f))
    m.vis.map.zfar, m.vis.map.shadowscale = 200, 1   # the sun is 45 m out; the shadow map spans the spot's whole cone
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = .12, .3, .05
    m.site_rgba[m.site("antenna").id, 3] = 0         # the dish shows where the antenna points
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    return m


def start(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    return d


def tilt(d, q0=Q0):
    return np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, d.qpos[3:7]))))))


def camera(lookat, distance, azimuth=22, elevation=-16):
    c = mujoco.MjvCamera()
    c.lookat[:], c.distance, c.azimuth, c.elevation = lookat, distance, azimuth, elevation
    return c


def middle(m, d):
    """Halfway between the bus and the hand: where the close camera looks."""
    return (d.xpos[m.body("base").id] + d.site("ee").xpos) / 2


def mark(scn, kind, a, size, rgba, b=None):
    """A sphere at a (size = radius), or a capsule from a to b (size = radius), added to the scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.array([size, 0, 0]), np.asarray(a, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    if b is not None:
        mujoco.mjv_connector(g, kind, size, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


SPHERE, CAPSULE = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE


@functools.lru_cache
def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label(img, lines, pos=(40, 32), size=30, colors=None):
    """White text with a dark shadow, one line per entry; colors maps a line number to an RGB."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    x, y = pos
    for i, t in enumerate(lines):
        dr.text((x + 2, y + 2), t, font=_font(size), fill=(0, 0, 0))
        dr.text((x, y), t, font=_font(size), fill=(colors or {}).get(i, WHITE))
        y += int(size * 1.25)
    return np.asarray(im)


def film(path, frames, still=None):
    """frames: an iterable of H x W x 3 uint8 images. Writes the mp4, and frame number `still` as a png."""
    ff = None
    for k, img in enumerate(frames):
        if ff is None:
            h, w = img.shape[:2]
            ff = subprocess.Popen([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "18", str(path)], stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
        if k == still:
            Image.fromarray(img).save(path.with_suffix(".png"))
    ff.stdin.close()
    assert ff.wait() == 0, f"ffmpeg failed on {path}"


def turn_j1(qh, t0=.5, T_=5., total=np.pi / 2, kp=5.):
    """Joint 1 turns `total` in T_ seconds on a smooth profile; every joint is held to it, the rest at home."""
    def law(d):
        s = np.clip((d.time - t0) / T_, 0, 1)
        ref, rate = qh.copy(), np.zeros(7)
        ref[0] += total * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
        rate[0] = total / T_ * (1 - np.cos(2 * np.pi * s))
        return rate + kp * (ref - d.qpos[7:14])
    return law


def ours(m):
    """Point our controller (v1/src/teleop.py: scheme 5, 3-D, our constants) at this Gen3 model, as the deck's map run
    did: the Gen3's own joint limits and rest pose, the arm-to-bus check on the arm only. -> hand site, bus at rest."""
    T.Q_LIM, T.Q_REST = m.jnt_range[1:].copy(), m.key("home").qpos[7:].copy()
    T.DECIM = TICK                                   # T.ray steps one physics step at a time
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] > m.body("link1").id and m.body_rootid[m.geom_bodyid[g]] == 1]
    T.clear = lambda m, d: min(mujoco.mj_geomDistance(m, d, 0, g, 1.0, None) for g in arm)
    return m.site("ee").id, Rot.from_quat(Q0[[1, 2, 3, 0]])


LEV = np.round(np.arange(.25, 2.01, .25), 2)         # the map's tilt steps, deg (the colour key)
TH = np.radians(np.arange(0, 360, 2.5))              # a line every 2.5 deg, as in the deck
PLANES = (lambda th: R0 @ (np.cos(th), 0, np.sin(th)),   # side: the arm's own plane (bus x-z)
          lambda th: R0 @ (np.cos(th), np.sin(th), 0))   # top: level with the hand (bus x-y)
MOVE = .3 * R0[:, 1]                                 # the deck's first move: the hand 0.3 m sideways
TOP_EL = np.degrees(np.arcsin(-EARTH[2]))            # the top view looks down the antenna axis, from the antenna side


def move(m, d, sid, Rb0):
    """The deck's first move: 6 s of our controller toward the hand 0.3 m sideways. Yields the state every tick."""
    d, goal = copy.copy(d), d.site_xpos[sid] + MOVE
    while d.time < 6 - 1e-9:
        T.ctrl(m, d, sid, Rb0, goal, 5, "3d")
        mujoco.mj_step(m, d, nstep=TICK)
        yield d


_cell = None


def _line(job):
    """One map line, in a worker: (plane, angle, after the move?) -> the hand at each tilt step, why the line ended,
    its end, where it started."""
    global _cell
    if _cell is None:
        m = load("medium", look=False)
        sid, Rb0 = ours(m)
        T.LIMITS = tuple(LEV)
        d = start(m)
        for dm in move(m, d, sid, Rb0):
            pass
        mujoco.mj_forward(m, dm)
        _cell = m, sid, Rb0, d, dm
    m, sid, Rb0, d, dm = _cell
    plane, th, after = job
    s = dm if after else d
    mk, why = [], []
    best = T.ray(m, s, sid, Rb0, PLANES[plane](th), 5, "3d", marks=mk, why=why)
    return np.array((mk + [best] * len(LEV))[:len(LEV)]), (why + ["reach"])[0], best, s.site_xpos[sid].copy()


def lines():
    """The map lines on the 150 kg servicer: side and top from rest, and top again after the first move with the tilt
    still counted from rest. About a minute on 10 processes. -> {name: (marks, why, end, start)}"""
    jobs = [(0, th, False) for th in TH] + [(1, th, False) for th in TH] + [(1, th, True) for th in TH]
    with Pool(max(1, min(10, (os.cpu_count() or 4) - 2))) as pool:
        res = pool.map(_line, jobs, chunksize=4)
    n, out = len(TH), {}
    for k, name in enumerate(("side", "top", "moved")):
        r = res[k * n:(k + 1) * n]
        out[name] = tuple(np.array([x[i] for x in r]) for i in range(3)) + (r[0][3],)
    return out


def fan(scn, o, plane, mk, why, end, upto=None):
    """Map lines from o, straight along each direction, coloured by the tilt step the hand had reached there; a dark
    cap where the bus passed 2 deg, the last 6 cm dashed where the arm ran out of reach. upto: lines drawn, its
    fraction how far the next one has grown."""
    upto = len(TH) if upto is None else upto
    for i in range(min(len(TH), int(np.ceil(upto)))):
        u = PLANES[plane](TH[i])
        L = max(0., (end[i] - o) @ u)
        b = np.minimum(np.r_[0, np.maximum.accumulate(np.clip((mk[i] - o) @ u, 0, None))], L)
        g = L * min(1., upto - i)
        cut = g if why[i] == "tilt" or g < L else L - .06
        for j in range(len(LEV)):
            if b[j] < min(b[j + 1], cut):
                mark(scn, CAPSULE, o + b[j] * u, .004, RAMP[j], o + min(b[j + 1], cut) * u)
        if g >= L and why[i] == "tilt":
            mark(scn, SPHERE, o + L * u, .009, RAMP[-1])
        elif g >= L:
            for s in (L - .05, L - .02):
                if s > 0:
                    c = RAMP[min(len(LEV) - 1, int(np.searchsorted(b, s, "right")) - 1)]
                    mark(scn, CAPSULE, o + s * u, .004, c, o + (s + .015) * u)


def edge(scn, o, plane, end, rgba, dashed=False):
    """The map's outline: the ends of its lines joined up, solid or dashed."""
    us = [PLANES[plane](th) for th in TH]
    p = [o + max(0., (e - o) @ u) * u for e, u in zip(end, us)]
    for i in range(len(p)):
        if not dashed or i % 2 == 0:
            mark(scn, CAPSULE, p[i], .004, rgba, p[(i + 1) % len(p)])


def inside(g, o, end, plane=1):
    """Which goals g (n x 3, in the plane) the map whose lines start at o says go to: closer than its edge, the edge
    interpolated between the two nearest lines."""
    u0, u1 = PLANES[plane](0), PLANES[plane](np.pi / 2)
    v = g - o
    r = np.array([max(0., (e - o) @ PLANES[plane](th)) for e, th in zip(end, TH)])
    th = np.arctan2(v @ u1, v @ u0) % (2 * np.pi)
    return np.hypot(v @ u0, v @ u1) < np.interp(th, np.r_[TH, 2 * np.pi], np.r_[r, r[0]])


def key(img, x=24, y=H - 40):
    """The colour key: bus tilt in the deck's eight 0.25 deg steps."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.text((x, y - 30), "bus tilt, °", font=_font(22), fill=WHITE)
    for j, c in enumerate(RAMP):
        dr.rectangle([x + 46 * j, y, x + 46 * j + 42, y + 14], fill=tuple(int(255 * v) for v in c[:3]))
        dr.text((x + 46 * j + 4, y + 16), f"{LEV[j]:g}", font=_font(16), fill=WHITE)
    return np.asarray(im)


def videos(out):
    out.mkdir(parents=True, exist_ok=True)
    tail = lambda: [print(f"  {p.name}") for p in sorted(out.glob("*.mp4"))]

    # opening: the 150 kg servicer at rest, the camera from behind the bus (Earth in view) round past the dish
    m, n = load("medium"), 10 * FPS
    d, r = start(m), mujoco.Renderer(m, H, W)
    c = middle(m, d)
    def orbit():
        for k in range(n):
            r.update_scene(d, camera(c, 5.6, -35 + 160 * k / n, -14 - 8 * np.sin(np.pi * k / n)))
            yield r.render()
    film(out / "orbit.mp4", orbit(), n // 2)
    r.close()

    # slide 1: 20 kg bus, joint 1 turns 90 deg in 5 s, the rest held; the camera fixed in the world, close
    m = load("small")
    d, r = start(m), mujoco.Renderer(m, H, W)
    qh, cam = d.qpos[7:14].copy(), camera(middle(m, d) + [0, 0, .05], 2.0)
    law = turn_j1(qh)
    def spin():
        for k in range(7 * FPS):
            while d.time < k / FPS - 1e-9:
                d.ctrl[:] = law(d)
                mujoco.mj_step(m, d, nstep=TICK)
            r.update_scene(d, cam)
            t = tilt(d)
            yield label(r.render(), ["20 kg bus, only joint 1 turns", f"joint 1   {np.degrees(d.qpos[7] - qh[0]):5.1f}°",
                                     f"bus tilt  {t:5.2f}°   budget {BUDGET:.0f}°"], colors={2: RED if t > BUDGET else WHITE})
    film(out / "spin.mp4", spin(), 6 * FPS)
    r.close()
    print(f"spin: joint 1 {np.degrees(d.qpos[7] - qh[0]):.2f} deg, bus tilt {tilt(d):.2f} deg")

    # slide 1, the table: the same turn on the three buses, side by side, each framed to fit
    pw, panels = W // 3, []
    for size, kg, dist in (("small", 20, 2.6), ("medium", 150, 3.4), ("large", 1500, 6.0)):
        m = load(size)
        m.vis.global_.offwidth = pw
        d = start(m)
        panels.append((m, d, mujoco.Renderer(m, H, pw), camera(middle(m, d) + [0, 0, .1], dist),
                       turn_j1(d.qpos[7:14].copy()), kg))
    def sizes():
        for k in range(7 * FPS):
            tiles = []
            for m, d, r, cam, law, kg in panels:
                while d.time < k / FPS - 1e-9:
                    d.ctrl[:] = law(d)
                    mujoco.mj_step(m, d, nstep=TICK)
                r.update_scene(d, cam)
                t = tilt(d)
                tiles.append(label(r.render(), [f"{kg} kg bus", f"tilt {t:5.2f}°"], pos=(24, 74),
                                   colors={1: RED if t > BUDGET else WHITE}))
            yield label(np.hstack(tiles), ["Joint 1 turns 90° in 5 s, the rest held"], pos=(24, 20))
    film(out / "sizes.mp4", sizes(), 6 * FPS)
    for p in panels:
        p[2].close()
    print("sizes:", ", ".join(f"{kg} kg {tilt(d):.2f} deg" for m, d, r, cam, law, kg in panels))

    # the lines both slides draw: side and top from rest, top again after the first move
    ln = lines()
    for name, (mk, why, end, o) in ln.items():
        print(f"{name}: lines ended by", {w: int((why == w).sum()) for w in np.unique(why)})

    # slide 3: the map as the deck draws it, side (the arm's own plane) and top (level with the hand), the lines
    # growing one by one from the hand at rest, 150 kg bus
    m = load("medium")
    sid, Rb0 = ours(m)
    d = start(m)
    p0, pw, n = d.site_xpos[sid].copy(), W // 2, len(TH)
    assert np.array_equal(p0, ln["top"][3]) and np.array_equal(p0, ln["side"][3]), "lines not from this rest pose"
    m.vis.global_.offwidth = pw
    r = mujoco.Renderer(m, H, pw)
    views = [(0, ln["side"], camera(p0 - .25 * UP, 2.6, 90, 0), "side: the arm's own plane"),
             (1, ln["top"], camera(p0 - .25 * UP, 2.6, 180, TOP_EL), "top: level with the hand")]
    def mapv():
        for k in range(FPS + 2 * n + 3 * FPS):
            tiles = []
            for plane, (mk, why, end, o), cam, name in views:
                r.update_scene(d, cam)
                fan(r.scene, o, plane, mk, why, end, np.clip((k - FPS) / 2, 0, n))
                tiles.append(label(r.render(), [name], pos=(24, 92), size=26))
            yield key(label(np.hstack(tiles), ["The map: straight lines from rest, 150 kg bus",
                                               "solid end: the bus passed 2°   dashed end: out of reach"], pos=(24, 14)))
    film(out / "map.mp4", mapv(), FPS + 2 * n + FPS)
    r.close()

    # slide 4: top view level with the hand. The map from rest; the first move, the hand 0.3 m sideways; then the
    # rest map turns dashed, the reach from where the arm is now is drawn solid, and the goals the rest map gets wrong
    m = load("medium")
    sid, Rb0 = ours(m)
    d = start(m)
    r = mujoco.Renderer(m, H, W)
    cam = camera(p0 + MOVE / 2 - .25 * UP, 2.4, 180, TOP_EL)
    end0, end1, p1 = ln["top"][2], ln["moved"][2], ln["moved"][3]
    title = "150 kg bus, top view level with the hand"
    def view(dd, dashed, now=False, goals=()):
        r.update_scene(dd, cam)
        edge(r.scene, p0, 1, end0, RAMP[1], dashed)
        if now:
            edge(r.scene, p1, 1, end1, [1, 1, 1, 1])
        for g, c in goals:
            mark(r.scene, SPHERE, g, .007, c)
        return r.render()
    def gapv():
        for _ in range(2 * FPS):
            yield label(view(d, False), [title, "the map from rest"])
        f = 0
        for dm in move(m, d, sid, Rb0):
            if dm.time >= f / FPS - 1e-9:
                f += 1
                yield label(view(dm, False), [title, "the first move: the hand 0.3 m sideways", f"bus tilt  {tilt(dm):4.2f}°"])
        mujoco.mj_forward(m, dm)
        assert np.array_equal(dm.site_xpos[sid], p1), "the move on screen is not the one the lines start from"
        u0, u1 = PLANES[1](0), PLANES[1](np.pi / 2)
        a = np.arange(-1.2, 1.2001, .03)
        G = (p0 + a[:, None, None] * u0 + a[None, :, None] * u1).reshape(-1, 3)
        was, now = inside(G, p0, end0), inside(G, p1, end1)
        goals = [(g, [1, .55, .1, 1]) for g in G[was & ~now]] + [(g, [.3, .6, 1, 1]) for g in G[~was & now]]
        msg = [title, f"after the move the bus is {tilt(dm):.2f}° off", "dashed: the map from rest   solid: the reach from here"]
        for _ in range(3 * FPS // 2):
            yield label(view(dm, True, True), msg)
        for _ in range(4 * FPS):
            yield label(label(view(dm, True, True, goals), msg), ["orange: the map says go, the hand can't get there now",
                                                                  "blue: the map says no, the hand gets there now"],
                        pos=(40, H - 90), size=26)
        print(f"gap: bus {tilt(dm):.2f} deg off after the move; grid goals (3 cm) wrong go {int((was & ~now).sum())}, "
              f"wrong no {int((~was & now).sum())}")
    film(out / "gap.mp4", gapv(), 8 * FPS + 3 * FPS // 2 + 2 * FPS)
    r.close()
    tail()


if __name__ == "__main__":
    videos(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "satellite")
