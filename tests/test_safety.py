"""Tests for the contact-velocity limit. Owner: P4 (first version by P3)."""
import numpy as np
import pytest

from src.safety import m_eff, op_space_inertia, reduced_mass, v_max_impulse, v_max_iso, vmax_map
from tests.test_momentum import load


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_lambda_is_symmetric_positive_definite(size):
    m, d = load(size)
    for locked in (True, False):
        L = op_space_inertia(m, d, locked)
        assert np.allclose(L, L.T, atol=1e-8 * np.abs(L).max())
        assert np.all(np.linalg.eigvalsh((L + L.T) / 2) > 0)


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_m_eff_bounds(size):
    """Locking the joints can only make the hand heavier, and never heavier than everything."""
    m, d = load(size)
    total = m.body_subtreemass[1]
    for u in np.eye(3):
        locked, free = m_eff(m, d, u, locked=True), m_eff(m, d, u, locked=False)
        assert 0 < free <= locked + 1e-9 <= total + 1e-6


def test_bigger_base_hits_harder():
    me = [m_eff(*load(s)) for s in ("small", "medium", "large")]
    assert me[0] < me[1] < me[2]
    v = [v_max_iso(x) for x in me]
    assert v[0] > v[1] > v[2]


def test_v_max_formulas():
    assert reduced_mass(10.0) == 10.0 and reduced_mass(10.0, 10.0) == 5.0
    assert v_max_impulse(2.0, 1.0) == pytest.approx(0.5)            # limit / ((1+e) m_eff)
    assert v_max_impulse(2.0, 1.0, e=1.0) == pytest.approx(0.25)
    assert v_max_iso(1.0, restrained=True) > v_max_iso(10.0, restrained=True)


def test_vmax_varies_with_pose():
    """The point of the safety map: no single scalar limit fits the whole workspace."""
    m, _ = load("medium")
    _, _, V = vmax_map(m, n=9)
    assert np.nanmax(V) > 1.5 * np.nanmin(V)
