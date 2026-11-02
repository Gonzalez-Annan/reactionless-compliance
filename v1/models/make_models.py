"""Writes ff_small/medium/large.xml and earth.png: the same 7-joint arm on three base masses, dressed as a satellite."""
from pathlib import Path
import numpy as np
import mujoco

SIZES = {"small": 30.0, "medium": 300.0, "large": 3000.0}  # base mass, kg (arm is 12.5 kg)
DENSITY = 300.0  # kg/m^3, sets the base cube side from its mass

# (joint axis, link length, link mass): shoulder-elbow-wrist arm, z-y-z-y-z-y-z
LINKS = [("0 0 1", .12, 2.5), ("0 1 0", .225, 2.5), ("0 0 1", .225, 2.0), ("0 1 0", .2, 2.0),
         ("0 0 1", .2, 1.5), ("0 1 0", .06, 1.0), ("0 0 1", .06, 1.0)]

# Seen from the bus at rest. The wings sit on the +y and -y faces and turn about y, so the sun is put square to that
# axis, up and a little behind the dish face: both wings face it fully. The dish is on the +x face and the Earth lies
# straight out from it, 10 degrees down. The Earth is 10 degrees across, out of the study screen's opening view.
SUN = np.array([-.35, 0, 1.]) / np.linalg.norm([-.35, 0, 1.])
EARTH = np.array([1., 0, -.18]) / np.linalg.norm([1., 0, -.18])
EARTH_D, EARTH_DEG = 25.0, 5.0       # m, and the Earth's angular radius in degrees
EARTH_PNG = "earth.png"              # made by earth_png(), read through the model file's folder

# The look is paint: textures, a sun, massless contact-free geoms on the bus, and the Earth and sun on a body of
# their own fixed to the world, after the arm. Geoms on the arm would enter the self-clearance and arm-to-bus checks,
# and a geom on the world body would take geom id 0, which the code reads as the bus.
SKY = """<asset>
<texture name="sky" type="skybox" builtin="gradient" rgb1=".03 .04 .09" rgb2="0 0 0" width="512" height="3072" mark="random" random=".0015" markrgb=".9 .9 1"/>
<texture name="cells" type="2d" builtin="checker" rgb1=".1 .13 .3" rgb2=".07 .09 .22" mark="edge" markrgb=".7 .7 .75" width="64" height="64"/>
<texture name="earth" type="2d" file="{png}"/>
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


def globe(nlon=48, nlat=24):
    """A unit UV sphere with texture coordinates for an equirectangular map: <mesh>. The seam column is doubled so
    the map wraps; the triangles that would collapse at the poles are left out."""
    lat = np.pi / 2 - np.arange(nlat + 1) / nlat * np.pi
    lon = np.arange(nlon + 1) / nlon * 2 * np.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    p = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    uv = np.stack(np.meshgrid(np.arange(nlat + 1) / nlat, np.arange(nlon + 1) / nlon, indexing="ij")[::-1], -1).reshape(-1, 2)
    f = []
    for i in range(nlat):
        for j in range(nlon):
            a, b = i * (nlon + 1) + j, (i + 1) * (nlon + 1) + j      # a top left, a+1 top right, b below a
            if i < nlat - 1:
                f.append((a, b, b + 1))
            if i > 0:
                f.append((a, b + 1, a + 1))
    s = v(np.round(p, 4))
    return f'<mesh name="globe" vertex="{s}" normal="{s}" texcoord="{v(np.round(uv, 4))}" face="{" ".join(map(str, np.ravel(f)))}"/>'


def earth_png(path, w=1024, seed=7):
    """A made-up Earth, equirectangular, from 3-D value noise on the sphere so it has no seam: sea, land coloured by a
    moisture field with a desert band, ice caps, clouds. Not our planet's map, it only has to read as Earth."""
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
    from PIL import Image
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)).save(path, optimize=True)


def paint(h):
    """Dark top deck so the white arm and the map colours stand out, arm flange,
    two wings of three panels on the +y and -y faces turned square to the sun, one dish on the +x face pointed at the
    Earth. Sizes go with the bus."""
    L, W, rd = 3 * h, 1.2 * h, .3 * h                        # wing length and width, dish radius, m
    y0 = 1.3 * h                                             # where a wing starts, past its yoke
    a = np.array([0, 1., 0])
    b = np.cross(SUN, a)                                     # across the wing; the panel normal is a x b = SUN
    b0 = np.array([h, 0, -.25 * h])                          # dish mast foot on the +x face
    b1 = b0 + .12 * EARTH
    g = [f'type="box" pos="0 0 {h + .004:.4g}" size="{.9 * h:.4g} {.9 * h:.4g} .004" material="dark"',
         f'type="cylinder" fromto="0 0 {h:.4g} 0 0 {h + .03:.4g}" size=".075" material="metal"']
    for s in (1, -1):
        g.append(f'type="cylinder" fromto="0 {s * h:.4g} 0 0 {s * (y0 + .01):.4g} 0" size=".018" material="metal"')
        for k in range(3):
            g.append(f'type="box" pos="0 {s * (y0 + (k + .5) * L / 3):.4g} 0" size="{L / 6 - .01:.4g} {W / 2:.4g} .008" '
                     f'xyaxes="{v(a)} {v(b)}" material="cells"')
    g += [f'type="cylinder" fromto="{v(b0)} {v(b1)}" size=".02" material="metal"',
          f'type="ellipsoid" pos="{v(b1 + .2 * rd * EARTH)}" size="{rd:.4g} {rd:.4g} {.2 * rd:.4g}" zaxis="{v(EARTH)}" material="white"',
          f'type="cylinder" fromto="{v(b1 + .38 * rd * EARTH)} {v(b1 + .39 * rd * EARTH)}" size="{.85 * rd:.4g}" material="metal"',
          f'type="cylinder" fromto="{v(b1 + .39 * rd * EARTH)} {v(b1 + 1.1 * rd * EARTH)}" size=".006" material="metal"',
          f'type="sphere" pos="{v(b1 + 1.1 * rd * EARTH)}" size="{max(.06 * rd, .012):.4g}" material="dark"']
    return "".join(f'\n<geom mass="0" {x}/>' for x in g)


def sky():
    """The Earth with a thin air shell, and the sun's disc, far off and fixed to the world: they do not turn with the
    bus, so a tilt shows against them. The globe is tipped 23 degrees, like the real one."""
    R, C = EARTH_D * np.sin(np.radians(EARTH_DEG)), EARTH_D * EARTH
    return (f'\n<body name="sky">'
            f'\n<geom mass="0" type="mesh" mesh="globe" pos="{v(C)}" euler="23 0 45" material="earth"/>'
            f'\n<geom mass="0" type="sphere" pos="{v(C)}" size="{1.025 * R:.4g}" material="air"/>'
            f'\n<geom mass="0" type="sphere" pos="{v(45 * SUN)}" size=".7" material="sun"/></body>')


def xml(name, mass, look=True):
    """The model. look=False is the bare model: the same physics, nothing to read from disk."""
    h = 0.5 * (mass / DENSITY) ** (1 / 3)
    body, close, z = "", "", h
    for i, (axis, length, m) in enumerate(LINKS, 1):
        body += (f'<body name="l{i}" pos="0 0 {z}"><joint name="j{i}" axis="{axis}"/>'
                 f'<geom type="capsule" fromto="0 0 0 0 0 {length}" size=".04" mass="{m}"/>\n')
        close += "</body>"
        z = length
    body += f'<site name="ee" pos="0 0 {z}" size=".02" rgba="1 0 0 1"/>'
    act = "".join(f'<velocity joint="j{i}" kv="150"/>' for i in range(1, 8))
    head, arm, sun, bus, rod, deco, far = "", 'rgba=".7 .7 .8 1"', "", 'rgba=".9 .75 .3 1"', 'rgba=".2 .6 1 1"', "", ""
    if look:                                                  # pin the bare model's framing: the Earth would widen it
        s = mujoco.MjModel.from_xml_string(xml(name, mass, False)).stat
        R = EARTH_D * np.sin(np.radians(EARTH_DEG))
        head = (f'<statistic center="{" ".join("%.17g" % v for v in s.center)}" extent="{s.extent:.17g}" '
                f'meansize="{s.meansize:.17g}" meanmass="{s.meanmass:.17g}" meaninertia="{s.meaninertia:.17g}"/>\n'
                '<visual><map shadowscale="1"/></visual>\n'   # the spot's shadow map spans its whole cone: no hard line on the Earth
                + SKY.format(png=EARTH_PNG, globe=globe().replace('name="globe"', f'name="globe" scale="{R:.4g} {R:.4g} {R:.4g}"')))
        # the -x rod is not drawn: it stays in the physics (1 g), so no number moves
        arm, bus, rod, deco, far = 'material="white"', 'material="foil"', 'rgba="0 0 0 0"', paint(h), sky()
        sun = f'\n<light name="sun" directional="true" pos="0 0 6" dir="{v(-SUN)}" diffuse=".6 .58 .52" specular=".4 .4 .4" castshadow="false"/>'
    return f"""<mujoco model="ff_{name}">
<option gravity="0 0 0" timestep="0.0005" integrator="RK4"/>
{head}<default><joint armature="0.5"/><geom contype="0" conaffinity="0" {arm}/></default>
<worldbody>
<light pos="0 -3 3" dir="0 1 -1"/>{sun}
<body name="base"><freejoint name="root"/>
<geom type="box" size="{h} {h} {h}" mass="{mass}" {bus}/>
<geom type="cylinder" fromto="0 0 0 {-h - .6} 0 0" size=".02" mass="0.001" {rod}/>{deco}
{body}{close}
</body>{far}
</worldbody>
<actuator>{act}</actuator>
</mujoco>
"""


if __name__ == "__main__":
    here = Path(__file__).parent
    earth_png(here / EARTH_PNG)
    for name, mass in SIZES.items():
        (here / f"ff_{name}.xml").write_text(xml(name, mass))
