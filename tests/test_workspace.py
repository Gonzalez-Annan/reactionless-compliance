"""Tests for the reachable-workspace sampler. Owner: P1."""
from pathlib import Path

import mujoco
import numpy as np

from src.workspace import reachable_points, surface_voxels

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load(name):
    return mujoco.MjModel.from_xml_path(str(MODELS_DIR / f"{name}.xml"))


def test_reach_is_bounded_by_arm_length():
    pts = reachable_points(load("ff_medium"), n_samples=3000)
    shoulder = np.array([0.4, 0, 0])  # mount on the medium base's +x face
    arm_length = 0.18 + 0.18 + 0.15 + 0.15 + 0.12 + 0.10 + 0.08
    assert np.linalg.norm(pts - shoulder, axis=1).max() <= arm_length + 1e-6


def test_large_base_blocks_reaching_behind_it():
    small = reachable_points(load("ff_small"), n_samples=5000)
    large = reachable_points(load("ff_large"), n_samples=5000)
    assert small[:, 0].min() < 0.0                     # small base: hand gets behind the centre
    assert large[:, 0].min() > 0.4                     # large base: can reach over its top, never near its centre


def test_surface_voxels_are_subset_of_occupied():
    pts = reachable_points(load("ff_small"), n_samples=3000)
    v = surface_voxels(pts, voxel=0.1)
    occupied = set(map(tuple, np.floor(pts / 0.1).astype(int)))
    assert len(v) > 0
    assert all(tuple(np.floor(c / 0.1).astype(int)) in occupied for c in v)
