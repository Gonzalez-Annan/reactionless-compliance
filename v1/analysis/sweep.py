"""Scripted benchmark: 5 schemes x 2 modes x 3 servicers, plus the beta sweep. -> data/sweep.csv"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from task import run_trial  # noqa: E402

BETAS = np.logspace(-2, 6, 9)

if __name__ == "__main__":
    rows = []
    for size in ["small", "medium", "large"]:
        for mode in ["3d", "6d"]:
            for s in range(1, 6):
                rows.append(run_trial(size, s, mode) | {"sweep": False})
        for s in (4, 5):
            for b in BETAS:
                rows.append(run_trial(size, s, "6d", beta=b) | {"sweep": True})
        print(size, "done", flush=True)
    (ROOT / "data").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "sweep.csv", index=False)
    print(df[~df.sweep].round(3).to_string())
