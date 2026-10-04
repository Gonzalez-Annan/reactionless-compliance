"""Reachable workspace of the arm, in the base frame. Owner: P1.

Samples joint configurations uniformly inside the joint limits, discards
those where the arm collides with the base or itself, and records the
end-effector position relative to the base. The base is held at the origin
with identity orientation, so world coordinates are base coordinates.

For display, the points are binned into cubes (voxels) and only the cubes
on the surface of the reachable set are kept.
"""
import mujoco
import numpy as np


def reachable_points(model, n_samples=30000, seed=0):
    """End-effector positions (N, 3) in the base frame for collision-free,
    within-limit joint configurations."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]  # joint 0 is the free joint
    sid = model.site("end_effector").id
    points = []
    for _ in range(n_samples):
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[7:] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        if data.ncon == 0:
            points.append(data.site_xpos[sid].copy())
    return np.array(points)


def surface_voxels(points, voxel=0.1):
    """Centres (M, 3) of occupied voxels that have at least one empty
    face-neighbour, i.e. the outer (and inner) surface of the point set."""
    idx = np.floor(points / voxel).astype(int)
    occupied = set(map(tuple, idx))
    neighbours = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    surface = [
        v for v in occupied
        if any((v[0] + a, v[1] + b, v[2] + c) not in occupied for a, b, c in neighbours)
    ]
    return (np.array(surface) + 0.5) * voxel
