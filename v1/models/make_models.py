"""Writes ff_small/medium/large.xml: the same 7-joint arm on three base masses, dressed as a satellite in space."""
from pathlib import Path
import mujoco

SIZES = {"small": 30.0, "medium": 300.0, "large": 3000.0}  # base mass, kg (arm is 12.5 kg)
DENSITY = 300.0  # kg/m^3, sets the base cube side from its mass

# (joint axis, link length, link mass): shoulder-elbow-wrist arm, z-y-z-y-z-y-z
LINKS = [("0 0 1", .12, 2.5), ("0 1 0", .225, 2.5), ("0 0 1", .225, 2.0), ("0 1 0", .2, 2.0),
         ("0 0 1", .2, 1.5), ("0 1 0", .06, 1.0), ("0 0 1", .06, 1.0)]

# The look is paint: textures, a sun, and massless contact-free geoms on the bus only. Geoms on the arm would enter
# the self-clearance and arm-to-bus checks, and a world geom would take geom id 0, which the code reads as the bus.
SKY = """<asset>
<texture name="sky" type="skybox" builtin="gradient" rgb1=".03 .04 .09" rgb2="0 0 0" width="512" height="3072" mark="random" random=".0015" markrgb=".9 .9 1"/>
<texture name="cells" type="2d" builtin="checker" rgb1=".1 .13 .3" rgb2=".07 .09 .22" mark="edge" markrgb=".7 .7 .75" width="64" height="64"/>
<material name="foil" rgba=".88 .68 .22 1" specular=".5" shininess=".6"/>
<material name="cells" texture="cells" texrepeat="6 3" specular=".8" shininess=".9"/>
<material name="metal" rgba=".6 .62 .66 1" specular=".7" shininess=".6"/>
<material name="white" rgba=".9 .9 .92 1" specular=".4"/>
<material name="dark" rgba=".16 .16 .18 1"/>
</asset>
"""


def paint(h):
    """Top deck, arm flange, two solar wings on booms (the old -x rod is one boom), a dish on the front face."""
    P, W, z = 2.2 * h, 1.3 * h, -.3 * h                       # wing length and width, dish height, m
    g = [f'type="box" pos="0 0 {h + .004:.4g}" size="{.9 * h:.4g} {.9 * h:.4g} .004" material="metal"',
         f'type="cylinder" fromto="0 0 {h:.4g} 0 0 {h + .03:.4g}" size=".075" material="dark"',
         f'type="cylinder" fromto="0 0 0 {h + .6:.4g} 0 0" size=".02" material="metal"',
         f'type="cylinder" fromto="0 {-h:.4g} {z:.4g} 0 {-h - .08:.4g} {z:.4g}" size=".025" material="metal"',
         f'type="ellipsoid" pos="0 {-h - .1:.4g} {z:.4g}" size="{.32 * h:.4g} .03 {.32 * h:.4g}" material="white"',
         f'type="cylinder" fromto="0 {-h - .1:.4g} {z:.4g} 0 {-h - .1 - .25 * h:.4g} {z:.4g}" size=".008" material="metal"']
    g += [f'type="box" pos="{s * (h + .6 + P / 2):.4g} 0 0" size="{P / 2:.4g} {W / 2:.4g} .01" material="cells"' for s in (1, -1)]
    return "".join(f'\n<geom mass="0" {x}/>' for x in g)


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
    head, arm, sun, bus, rod, deco = "", 'rgba=".7 .7 .8 1"', "", 'rgba=".9 .75 .3 1"', 'rgba=".2 .6 1 1"', ""
    if look:                                                  # pin the bare model's framing: the wings would widen it
        s = mujoco.MjModel.from_xml_string(xml(name, mass, False)).stat
        head = (f'<statistic center="{" ".join("%.17g" % v for v in s.center)}" extent="{s.extent:.17g}" '
                f'meansize="{s.meansize:.17g}"/>\n' + SKY)
        arm, bus, rod, deco = 'material="white"', 'material="foil"', 'material="metal"', paint(h)
        sun = '\n<light name="sun" directional="true" pos="0 0 6" dir="-.5 .8 -1" diffuse=".6 .58 .52" specular=".4 .4 .4" castshadow="false"/>'
    return f"""<mujoco model="ff_{name}">
<option gravity="0 0 0" timestep="0.0005" integrator="RK4"/>
{head}<default><joint armature="0.5"/><geom contype="0" conaffinity="0" {arm}/></default>
<worldbody>
<light pos="0 -3 3" dir="0 1 -1"/>{sun}
<body name="base"><freejoint name="root"/>
<geom type="box" size="{h} {h} {h}" mass="{mass}" {bus}/>
<geom type="cylinder" fromto="0 0 0 {-h - .6} 0 0" size=".02" mass="0.001" {rod}/>{deco}
{body}{close}
</body>
</worldbody>
<actuator>{act}</actuator>
</mujoco>
"""


if __name__ == "__main__":
    for name, mass in SIZES.items():
        (Path(__file__).parent / f"ff_{name}.xml").write_text(xml(name, mass))
