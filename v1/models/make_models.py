"""Writes ff_small/medium/large.xml: the same 7-joint arm on three base masses."""
from pathlib import Path

SIZES = {"small": 30.0, "medium": 300.0, "large": 3000.0}  # base mass, kg (arm is 12.5 kg)
DENSITY = 300.0  # kg/m^3, sets the base cube side from its mass

# (joint axis, link length, link mass): shoulder-elbow-wrist arm, z-y-z-y-z-y-z
LINKS = [("0 0 1", .12, 2.5), ("0 1 0", .225, 2.5), ("0 0 1", .225, 2.0), ("0 1 0", .2, 2.0),
         ("0 0 1", .2, 1.5), ("0 1 0", .06, 1.0), ("0 0 1", .06, 1.0)]


def xml(name, mass):
    h = 0.5 * (mass / DENSITY) ** (1 / 3)
    body, close, z = "", "", h
    for i, (axis, length, m) in enumerate(LINKS, 1):
        body += (f'<body name="l{i}" pos="0 0 {z}"><joint name="j{i}" axis="{axis}"/>'
                 f'<geom type="capsule" fromto="0 0 0 0 0 {length}" size=".04" mass="{m}"/>\n')
        close += "</body>"
        z = length
    body += f'<site name="ee" pos="0 0 {z}" size=".02" rgba="1 0 0 1"/>'
    act = "".join(f'<velocity joint="j{i}" kv="150"/>' for i in range(1, 8))
    return f"""<mujoco model="ff_{name}">
<option gravity="0 0 0" timestep="0.0005" integrator="RK4"/>
<default><joint armature="0.5"/><geom contype="0" conaffinity="0" rgba=".7 .7 .8 1"/></default>
<worldbody>
<light pos="0 -3 3" dir="0 1 -1"/>
<body name="base"><freejoint name="root"/>
<geom type="box" size="{h} {h} {h}" mass="{mass}" rgba=".9 .75 .3 1"/>
<geom type="cylinder" fromto="0 0 0 {-h - .6} 0 0" size=".02" mass="0.001" rgba=".2 .6 1 1"/>
{body}{close}
</body>
</worldbody>
<actuator>{act}</actuator>
</mujoco>
"""


if __name__ == "__main__":
    for name, mass in SIZES.items():
        (Path(__file__).parent / f"ff_{name}.xml").write_text(xml(name, mass))
