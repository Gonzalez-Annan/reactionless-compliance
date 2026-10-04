"""Tests for the closed-loop task. Owner: P1."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.task import DWELL, WAYPOINTS, ClosedLoopTask, run_scripted


def make(mode="6d", loops=2):
    return ClosedLoopTask(np.zeros(3), Rotation.identity(), mode, loops)


def test_waypoint_needs_dwell_and_advances():
    task = make("3d")
    _, pos, rot = task.current
    task.update(pos, rot, DWELL / 2)
    assert task.index == 0          # not long enough yet
    task.update(pos, rot, DWELL / 2)
    assert task.index == 1


def test_leaving_tolerance_resets_dwell():
    task = make("3d")
    _, pos, rot = task.current
    task.update(pos, rot, DWELL * 0.9)
    task.update(pos + 1.0, rot, 0.01)
    task.update(pos, rot, DWELL * 0.9)
    assert task.index == 0


def test_6d_checks_orientation_3d_does_not():
    for mode, expect in (("6d", 0), ("3d", 1)):
        task = make(mode)
        _, pos, rot = task.current
        wrong = Rotation.from_euler("z", 45, degrees=True) * rot
        task.update(pos, wrong, DWELL)
        assert task.index == expect


def test_done_after_all_loops():
    task = make("6d", loops=2)
    for _ in range(2 * len(WAYPOINTS)):
        _, pos, rot = task.current
        task.update(pos, rot, DWELL)
    assert task.done and task.loop == 2


def test_reference_passes_through_every_waypoint():
    task = make("6d", loops=1)
    duration, ref = task.reference()
    ts = np.linspace(0, duration, 20000)
    for _, wp_pos, _ in task.waypoints:
        assert min(np.linalg.norm(ref(t)[0] - wp_pos) for t in ts) < 1e-3
    pos, rot, twist = ref(duration)
    assert np.allclose(pos, 0, atol=1e-9) and np.allclose(twist, 0, atol=1e-9)


@pytest.mark.parametrize("mode", ["3d", "6d"])
def test_scripted_demo_completes_on_medium(mode):
    result = run_scripted("medium", mode, "demo")
    assert result["completed"]
