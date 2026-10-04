"""Single-clock CSV logging. Owner: P1.

One row per control tick. The clock is simulation time (data.time); wall
time is logged next to it only to detect real-time overruns. Vector fields
are expanded into one column each (name_0, name_1, ...), so the CSV loads
straight into pandas.

    with CsvLogger("data/pilot/run.csv") as log:
        log.write(t_sim=data.time, qpos=data.qpos, xdot_cmd=xdot)
"""
import csv
import time
from pathlib import Path

import numpy as np


class CsvLogger:
    def __init__(self, path, meta=None):
        """meta: optional dict written as '# key: value' lines above the header
        (participant, scheme, model, ...)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        for key, value in (meta or {}).items():
            self._file.write(f"# {key}: {value}\n")
        self._writer = None
        self._columns = None
        self._t0_wall = time.perf_counter()

    def write(self, **fields):
        """Log one row. The first call fixes the columns; later calls must
        pass the same fields with the same vector lengths."""
        row = {"t_wall": time.perf_counter() - self._t0_wall}
        for name, value in fields.items():
            arr = np.atleast_1d(np.asarray(value, dtype=float))
            if arr.size == 1 and np.ndim(value) == 0:
                row[name] = float(arr[0])
            else:
                for i, v in enumerate(arr.ravel()):
                    row[f"{name}_{i}"] = float(v)

        if self._writer is None:
            self._columns = list(row)
            self._writer = csv.DictWriter(self._file, fieldnames=self._columns)
            self._writer.writeheader()
        elif list(row) != self._columns:
            raise ValueError(f"logged fields changed: {sorted(set(row) ^ set(self._columns))}")
        self._writer.writerow(row)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
