"""Tests for the CSV logger and the headless teleop control loop. Owner: P1."""
import numpy as np
import pandas as pd
import pytest

from src.logger import CsvLogger
from src.teleop import CONTROL_HZ, Teleop


def test_logger_expands_vectors_and_writes_meta(tmp_path):
    path = tmp_path / "run.csv"
    with CsvLogger(path, meta={"participant": "P01"}) as log:
        log.write(t_sim=0.0, q=np.array([1.0, 2.0]))
        log.write(t_sim=0.01, q=np.array([3.0, 4.0]))

    assert path.read_text().startswith("# participant: P01\n")
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["t_wall", "t_sim", "q_0", "q_1"]
    assert df["q_1"].tolist() == [2.0, 4.0]


def test_logger_rejects_changed_fields(tmp_path):
    with CsvLogger(tmp_path / "run.csv") as log:
        log.write(t_sim=0.0)
        with pytest.raises(ValueError):
            log.write(t_sim=0.01, extra=1.0)


def test_teleop_tick_advances_sim_and_logs(tmp_path):
    path = tmp_path / "teleop.csv"
    t = Teleop("medium", "6d", "hold", log_path=path, headless=True)
    xdot = np.array([0.05, 0, 0, 0, 0, 0])
    for _ in range(CONTROL_HZ):  # 1 s
        t.control_tick(xdot)
    t.close()

    df = pd.read_csv(path, comment="#")
    assert len(df) == CONTROL_HZ
    assert df["t_sim"].iloc[-1] == pytest.approx(1.0)
    assert df["target_pos_0"].iloc[-1] - df["target_pos_0"].iloc[0] == pytest.approx(0.05 * 0.99)


def test_teleop_3d_mode_zeroes_rotation():
    t = Teleop("medium", "3d", "hold", headless=True)
    xdot = t.twist_from_input(np.ones(6))
    assert np.all(xdot[3:] == 0)
