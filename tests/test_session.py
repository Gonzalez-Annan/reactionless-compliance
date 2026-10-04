"""Tests for the session runner's counterbalancing. Owner: P1."""
from collections import Counter

import pytest

from src.session import MODES, SCHEMES, condition_order, williams_square


def test_williams_square_balanced():
    n = 5
    rows = williams_square(n)
    assert len(rows) == 2 * n
    for pos in range(n):  # each condition once per position, twice over the 2n rows
        assert Counter(r[pos] for r in rows) == {c: 2 for c in range(n)}
    pairs = Counter((r[i], r[i + 1]) for r in rows for i in range(n - 1))
    assert len(pairs) == n * (n - 1) and len(set(pairs.values())) == 1  # every ordered pair equally often


@pytest.mark.parametrize("pid", ["P01", "P02", "P07", "P12"])
def test_every_condition_exactly_once(pid):
    order = condition_order(pid)
    assert sorted(order) == sorted((m, s) for m in MODES for s in SCHEMES)


def test_mode_blocks_alternate_between_participants():
    assert condition_order("P01")[0][0] == "3d"
    assert condition_order("P02")[0][0] == "6d"
    for pid in ("P01", "P02"):
        modes = [m for m, _ in condition_order(pid)]
        assert modes[:5] == [modes[0]] * 5 and modes[5:] == [modes[5]] * 5


def test_blocks_use_different_scheme_orders():
    order = condition_order("P03")
    assert [s for _, s in order[:5]] != [s for _, s in order[5:]]


def test_bad_participant_id():
    with pytest.raises(ValueError):
        condition_order("Alice")
