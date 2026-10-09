"""Operator view for the paper: the bus, the arm at rest, the three cages and a yellow target in the caution zone.
Run from the repo root. Drawn offscreen with the same draw_envelope() the teleop viewer uses, from the cached
cage of the medium bus. -> paper/figs/fig_opview.png"""
import sys; sys.path.insert(0, "src")
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import teleop as T
from view import draw_envelope

m, d, sid, Rb0 = T.setup("medium"); p0 = d.site_xpos[sid].copy()
rays = list(np.load(T.OUT.parent / "decide4_medium.npz")["rays"])
tips = [r[-1] for r in rays]
u = np.array([0.3, 0.8, 0.5]); u /= np.linalg.norm(u)
goal = p0 + u / T.depth(tips, p0, p0 + u)                 # depth 1.0: the middle of the caution zone
assert T.zone(T.depth(tips, p0, goal)) == 1
m.vis.global_.offwidth, m.vis.global_.offheight = 1100, 1100
r = mujoco.Renderer(m, 1100, 1100, max_geom=4000)         # the cage is about 1500 triangles
cam = mujoco.MjvCamera(); cam.lookat[:] = p0 - [0, 0, 0.3]; cam.distance = 2.9
imgs = []
for az, el in ((135, -20), (60, -35)):
    cam.azimuth, cam.elevation = az, el
    r.update_scene(d, cam)
    n = draw_envelope(r.scene, r.scene.ngeom, rays, T.TRIS, T.EDGES)
    mujoco.mjv_initGeom(r.scene.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, [0.045, 0, 0], goal, np.eye(3).flatten(), [1, 1, 0, 1])
    r.scene.geoms[n].emission = 1                         # the dot must read through the cage faces
    r.scene.ngeom = n + 1
    imgs.append(r.render())
plt.imsave("paper/figs/fig_opview.png", np.hstack(imgs))
print("target depth %.2f, %d geoms" % (T.depth(tips, p0, goal), n + 1))
