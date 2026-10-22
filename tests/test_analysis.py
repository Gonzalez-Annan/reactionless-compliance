"""Pilot data flows end to end: log -> metrics -> statistics -> figures (plan, recess gate)."""
import numpy as np
import pandas as pd
import pytest

from analysis import figures
from analysis.metrics import (collect, friedman, pairwise_wilcoxon, synthetic_questionnaires, trial_metrics)
from src.experiments import run
from src.verify import HAZARDS


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    """One scripted 1-loop trial per scheme pair, written like a participant's session."""
    root = tmp_path_factory.mktemp("participants")
    for scheme in ("dls", "impedance_rns"):
        run("medium", "3d", scheme, loops=1, log_path=root / "P99" / f"P99_{scheme}.csv")
    return root


def test_metrics_from_a_log(pilot):
    m = trial_metrics(pilot / "P99" / "P99_dls.csv", loops=1)
    assert m["completed"] and m["rms_pos_err_mm"] < 20
    assert 0 <= m["peak_antenna_deg"] < 5 and m["speed_limited_frac"] == 0.0
    df = collect(pilot)
    assert len(df) == 2 and set(df.scheme) == {"dls", "impedance_rns"}


def test_statistics_on_synthetic_ratings():
    q = synthetic_questionnaires(n=10)
    f = friedman(q, "predictability")
    assert f["n"] == 10 and f["k"] == 5 and 0 <= f["W"] <= 1
    w = pairwise_wilcoxon(q, "tlx_raw")
    assert len(w) == 10 and (w.p_holm >= w.p - 1e-12).all() and w.p_holm.is_monotonic_increasing


def test_figures_render(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    rows = [{"size": s, "mode": mo, "scheme": sc, "peak_antenna_deg": 1 + i, "rms_pos_err_mm": 5.0,
             "joint_return_rad": 0.1, "completed": True, "gain_beta": np.nan}
            for s in figures.SIZES for mo in ("3d", "6d") for i, sc in enumerate(figures.SCHEMES)]
    df = pd.DataFrame(rows)
    for f in (figures.fig1_tradeoff(df), figures.fig2_size(df), figures.fig3_joint_return(df),
              figures.fig4_subjective(synthetic_questionnaires(n=4), synthetic=True)):
        assert f.exists() and f.with_suffix(".png").exists()


def test_hazard_table_size():
    assert 8 <= len(HAZARDS) <= 12      # plan, Week 9
