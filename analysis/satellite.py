"""Alin's Gen3 servicers dressed as satellites, and the videos for the dynamics talk, in the talk's order. Owner: P2.

    python -m analysis.satellite [out_dir]     # default figures/satellite/

models/ is not changed. The look is added at load time and is paint only: solar wings on the +y and -y faces, a dish
on the antenna face, the Earth on the antenna's boresight, a sun. Every added geom has no mass and no contact, so the
physics is the same to the bit (tests/test_satellite.py). The bus starts turned so the arm points up and the antenna
at the Earth; gravity is off and momentum starts at zero, so that changes no number. The hand site is read as "ee",
the name our v1 code uses.

  orbit.mp4   opening: the camera goes round the 150 kg servicer at rest
  spin.mp4    slide 1: 20 kg bus, only joint 1 turns, 90 deg in 5 s, the rest held
  sizes.mp4   slide 1: the same turn on the 20, 150 and 1500 kg buses
  map.mp4     slide 3: 150 kg bus, our controller takes the hand straight out from rest, the line coloured by tilt,
              until the bus passes 2 deg or the arm runs out of reach (the deck's first run, gains not tuned)
  gap.mp4     slide 4: 150 kg bus, our controller takes the hand 0.3 m out and back to the same spot; the hand
              comes back, the joints do not
"""
import copy
import functools
import io
import re
import shutil
import subprocess
import sys
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
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] > m.body("link1").id and m.body_rootid[m.geom_bodyid[g]] == 1]
    T.clear = lambda m, d: min(mujoco.mj_geomDistance(m, d, 0, g, 1.0, None) for g in arm)
    return m.site("ee").id, Rot.from_quat(Q0[[1, 2, 3, 0]])


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

    # slide 3: straight out from rest under our controller, 12 directions in the level plane through the hand (the
    # deck's top map), each line coloured by the tilt when the hand got there, 3x speed
    m = load("medium")
    sid, Rb0 = ours(m)
    d0, r = start(m), mujoco.Renderer(m, H, W)
    p0 = d0.site_xpos[sid].copy()
    cam = camera(p0 - [0, 0, .22], 1.8)
    done, ends = [], []                              # finished segments (a, b, rgba); ends (point, passed 2 deg)
    def scene(d, segs, u=None):
        r.update_scene(d, cam)
        if u is not None:                            # the straight line the controller pulls along
            mark(r.scene, CAPSULE, p0, .002, [.7, .7, .7, .5], p0 + .9 * u)
        for a, b, c in done + segs:
            mark(r.scene, CAPSULE, a, .007, c, b)
        for p, hit in ends:
            mark(r.scene, SPHERE, p, .022, [.95, .3, .25, 1] if hit else [.85, .85, .85, 1])
        mark(r.scene, SPHERE, p0, .015, [1, .55, .1, 1])
        return r.render()
    key = ["From rest, our controller pulls the hand along a straight line",
           "colour: bus tilt   red end: passed 2°   grey end: out of reach"]
    def rays():
        for i, th in enumerate(np.radians(np.arange(0, 360, 30))):
            u = R0 @ (np.cos(th), np.sin(th), 0)
            d, segs, prev, last, hit = copy.copy(d0), [], p0, 0., False
            for tick in range(int((T.SLACK * T.REACH / T.V_GO + 1) * 100)):
                p, base = T.ctrl(m, d, sid, Rb0, p0 + T.REACH * u, 5, "3d")
                if base > T.BASE_MAX or T.clear(m, d) < T.CLEAR:
                    hit = base > T.BASE_MAX
                    break
                if tick and tick % 50 == 0:                       # every 0.5 s: stalled if under 2 mm of progress
                    if (p - p0) @ u - last < .002:
                        break
                    last = (p - p0) @ u
                if tick % 10 == 0:                                # one frame per 0.1 s of simulation: 3x speed
                    segs.append((prev, p, RAMP[min(7, int(base / .25))]))
                    prev = p
                    yield label(scene(d, segs, u), key + [f"line {i + 1} of 12   bus tilt {base:4.2f}°"],
                                colors={2: RED if base > BUDGET else WHITE})
                mujoco.mj_step(m, d, nstep=TICK)
            done.extend(segs)
            ends.append((prev, hit))
            print(f"  ray {int(np.degrees(th)):3d} deg: {100 * np.linalg.norm(prev - p0):4.0f} cm, {'tilt' if hit else 'reach'}")
            for _ in range(FPS // 2):
                yield label(scene(d, []), key + [f"line {i + 1} of 12   bus tilt {base:4.2f}°"],
                            colors={2: RED if hit else WHITE})
        for _ in range(2 * FPS):
            yield label(scene(d0, []), key + ["the map, drawn once from rest"])
    film(out / "map.mp4", rays(), 200)
    r.close()

    # slide 4: the hand 0.3 m out (sideways) and back to the same spot, 6 s each way, as the deck's gap test moves it
    m = load("medium")
    sid, Rb0 = ours(m)
    d, r = start(m), mujoco.Renderer(m, H, W)
    p0, q0 = d.site_xpos[sid].copy(), d.qpos[7:14].copy()
    p1 = p0 + .3 * R0[:, 1]
    cam = camera((p0 + p1) / 2 - [0, 0, .3], 1.8)
    def gap():
        for k in range(14 * FPS):
            while d.time < k / FPS - 1e-9:
                T.ctrl(m, d, sid, Rb0, p0 if d.time < .5 or d.time >= 6.5 else p1, 5, "3d")
                mujoco.mj_step(m, d, nstep=TICK)
            r.update_scene(d, cam)
            mark(r.scene, CAPSULE, p0, .004, [.8, .8, .8, .8], p1)
            mark(r.scene, SPHERE, p0, .015, [1, .55, .1, 1])
            t = tilt(d)
            yield label(r.render(), ["150 kg bus, the hand 0.3 m out and back to the same spot",
                                     f"hand from start  {100 * np.linalg.norm(d.site_xpos[sid] - p0):4.1f} cm",
                                     f"bus tilt  {t:4.2f}°", f"joints off  {np.linalg.norm(d.qpos[7:14] - q0):4.2f} rad"]
                        + (["the hand is back, the joints are not"] if d.time > 12.5 else []))
    film(out / "gap.mp4", gap(), 4 * FPS)
    r.close()
    print(f"gap: hand {1000 * np.linalg.norm(d.site_xpos[sid] - p0):.1f} mm from start, bus {tilt(d):.2f} deg off, "
          f"joints {np.linalg.norm(d.qpos[7:14] - q0):.2f} rad off")
    tail()


if __name__ == "__main__":
    videos(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures" / "satellite")
