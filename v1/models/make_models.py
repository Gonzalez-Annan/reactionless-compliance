"""Writes ff_small/medium/large.xml: the same 7-joint arm on three base masses, dressed as a satellite in space."""
from pathlib import Path
import numpy as np
import mujoco

SIZES = {"small": 30.0, "medium": 300.0, "large": 3000.0}  # base mass, kg (arm is 12.5 kg)
DENSITY = 300.0  # kg/m^3, sets the base cube side from its mass

# (joint axis, link length, link mass): shoulder-elbow-wrist arm, z-y-z-y-z-y-z
LINKS = [("0 0 1", .12, 2.5), ("0 1 0", .225, 2.5), ("0 0 1", .225, 2.0), ("0 1 0", .2, 2.0),
         ("0 0 1", .2, 1.5), ("0 1 0", .06, 1.0), ("0 0 1", .06, 1.0)]

# Where the sun and the Earth are, seen from the bus at rest. The sun is up and behind the opening camera, so the arm
# stays lit from the front; the Earth sits low on the right of the opening view, clear of the map.
SUN = np.array([.5, -.8, 1.]) / np.linalg.norm([.5, -.8, 1.])
EARTH = np.array([.55, .53, -.884]) / np.linalg.norm([.55, .53, -.884])
EARTH_D, EARTH_DEG = 25.0, 8.7       # m and degrees: the Earth as seen from geostationary orbit, 17.4 degrees across

# The look is paint: textures, a sun, massless contact-free geoms on the bus, and the Earth and sun on a body of
# their own fixed to the world, after the arm. Geoms on the arm would enter the self-clearance and arm-to-bus checks,
# and a geom on the world body would take geom id 0, which the code reads as the bus.
SKY = """<asset>
<texture name="sky" type="skybox" builtin="gradient" rgb1=".03 .04 .09" rgb2="0 0 0" width="512" height="3072" mark="random" random=".0015" markrgb=".9 .9 1"/>
<texture name="cells" type="2d" builtin="checker" rgb1=".1 .13 .3" rgb2=".07 .09 .22" mark="edge" markrgb=".7 .7 .75" width="64" height="64"/>
<material name="foil" rgba=".5 .38 .16 1" specular=".4" shininess=".5"/>
<material name="cells" texture="cells" texrepeat="6 3" specular=".8" shininess=".9"/>
<material name="metal" rgba=".6 .62 .66 1" specular=".7" shininess=".6"/>
<material name="white" rgba=".9 .9 .92 1" specular=".4"/>
<material name="dark" rgba=".16 .16 .18 1"/>
<material name="earth" rgba=".08 .28 .66 1" specular=".3" shininess=".4"/>
<material name="land" rgba=".26 .42 .2 1" specular="0"/>
<material name="sand" rgba=".78 .6 .34 1" specular="0"/>
<material name="cloud" rgba=".95 .96 1 .7" specular="0"/>
<material name="air" rgba=".55 .75 1 .22"/>
<material name="sun" rgba="1 .96 .85 1" emission="1"/>
</asset>
"""


def v(a):
    return " ".join(f"{x:.4g}" for x in a)


def paint(h):
    """Dark top deck so the white arm and the map colours stand out, arm flange, one solar panel on the -x boom (the
    old rod) turned to face the sun, one dish on the +x face pointing at the Earth. Sizes go with the bus."""
    L, W, rd = 2.4 * h, 1.2 * h, .3 * h                       # panel length and width, dish radius, m
    tip = np.array([-h - .6, 0, 0])                          # end of the boom
    u = np.array([-1., 0, 0]) + SUN[0] * SUN                 # along the boom, turned square to the sun
    u /= np.linalg.norm(u)
    c = tip + (.08 + L / 2) * u                              # panel centre, a short yoke past the boom's end
    b0 = np.array([h, 0, -.25 * h])                          # dish mast foot on the +x face
    b1 = b0 + .12 * EARTH
    g = [f'type="box" pos="0 0 {h + .004:.4g}" size="{.9 * h:.4g} {.9 * h:.4g} .004" material="dark"',
         f'type="cylinder" fromto="0 0 {h:.4g} 0 0 {h + .03:.4g}" size=".075" material="metal"',
         f'type="cylinder" fromto="{v(tip)} {v(tip + .09 * u)}" size=".015" material="metal"',
         f'type="box" pos="{v(c)}" size="{L / 2:.4g} {W / 2:.4g} .008" xyaxes="{v(u)} {v(np.cross(SUN, u))}" material="cells"',
         f'type="cylinder" fromto="{v(b0)} {v(b1)}" size=".02" material="metal"',
         f'type="ellipsoid" pos="{v(b1 + .2 * rd * EARTH)}" size="{rd:.4g} {rd:.4g} {.2 * rd:.4g}" zaxis="{v(EARTH)}" material="white"',
         f'type="cylinder" fromto="{v(b1 + .38 * rd * EARTH)} {v(b1 + .39 * rd * EARTH)}" size="{.85 * rd:.4g}" material="metal"',
         f'type="cylinder" fromto="{v(b1 + .39 * rd * EARTH)} {v(b1 + 1.1 * rd * EARTH)}" size=".006" material="metal"',
         f'type="sphere" pos="{v(b1 + 1.1 * rd * EARTH)}" size="{max(.06 * rd, .012):.4g}" material="dark"']
    return "".join(f'\n<geom mass="0" {x}/>' for x in g)


# Land and cloud patches on the face of the Earth the satellite sees: (right, up offset on that face, two half-sizes / R,
# turn in degrees, material). Overlapping patches make two continents and an island; three thin cloud streaks.
PATCHES = [(-.15, .05, .3, .16, 80, "land"), (-.05, .32, .22, .14, 10, "sand"), (-.22, -.25, .16, .1, 60, "land"),
           (.45, .25, .26, .12, 30, "land"), (.56, 0, .14, .1, -20, "land"), (.2, -.5, .13, .07, -10, "land"),
           (-.45, -.1, .2, .05, 25, "cloud"), (.25, .5, .22, .05, -15, "cloud"), (.15, -.2, .18, .04, 5, "cloud")]


def sky():
    """The Earth, a thin air shell and the sun's disc, far off and fixed to the world: they do not turn with the bus.
    Each patch is an ellipsoid whose top bends like the globe along its long axis, sunk so its top stands 0.6% of R proud."""
    R, C = EARTH_D * np.sin(np.radians(EARTH_DEG)), EARTH_D * EARTH
    n0 = -EARTH
    a = np.cross(n0, [0, 0, 1.]); a /= np.linalg.norm(a)
    b = np.cross(n0, a)
    g = [f'type="sphere" pos="{v(C)}" size="{R:.4g}" material="earth"']
    for x, y, sa, sb, turn, mat in PATCHES:
        n = n0 + x * a + y * b; n /= np.linalg.norm(n)
        t0 = np.cross(n, b); t0 /= np.linalg.norm(t0)
        t = np.cos(np.radians(turn)) * t0 + np.sin(np.radians(turn)) * np.cross(n, t0)
        c = sa * sa                                          # depth / R that gives the top the globe's curvature: c/sa^2 = 1/R
        g.append(f'type="ellipsoid" pos="{v(C + (1 - c + .006 + (mat == "cloud") * .004) * R * n)}" size="{sa * R:.4g} {sb * R:.4g} {c * R:.4g}" '
                 f'xyaxes="{v(t)} {v(np.cross(n, t))}" material="{mat}"')
    g += [f'type="sphere" pos="{v(C)}" size="{1.03 * R:.4g}" material="air"',
          f'type="sphere" pos="{v(45 * SUN)}" size=".7" material="sun"']
    return '\n<body name="sky">' + "".join(f'\n<geom mass="0" {x}/>' for x in g) + '</body>'


def xml(name, mass, look=True):
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
        head = (f'<statistic center="{" ".join("%.17g" % v for v in s.center)}" extent="{s.extent:.17g}" '
                f'meansize="{s.meansize:.17g}" meanmass="{s.meanmass:.17g}" meaninertia="{s.meaninertia:.17g}"/>\n'
                '<visual><map shadowscale="1"/></visual>\n' + SKY)  # the spot's shadow map spans its whole cone: no hard line on the Earth
        arm, bus, rod, deco, far = 'material="white"', 'material="foil"', 'material="metal"', paint(h), sky()
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
    for name, mass in SIZES.items():
        (Path(__file__).parent / f"ff_{name}.xml").write_text(xml(name, mass))
