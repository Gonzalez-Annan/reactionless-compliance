"""
Momentum-conservation gate (the Week 6 gate). Owner: P2.

Once src/dynamics.py exists, this must assert:
    || H_b @ v_b + H_bm @ qdot || < 1e-6
for randomly driven qdot over 10s of simulated motion, for each of the
three servicer models. All downstream results are invalid until this
passes for ff_small, ff_medium, and ff_large.
"""
import pytest


@pytest.mark.skip(reason="Pending P2's src/dynamics.py (H_b, H_bm, J*)")
def test_momentum_conserved():
    raise NotImplementedError
