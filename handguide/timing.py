"""Per-stage timestamps and a median / p95 summary.

Usage:
    t = StageTimer()
    t.start_frame()
    with t.stage("hand"): ...
    t.end_frame()
    print(t.summary())
"""
from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class StageTimer:
    def __init__(self):
        self.rows: list[dict] = []
        self._cur: dict | None = None
        self._t0 = 0.0

    def start_frame(self, **meta):
        self._t0 = time.perf_counter()
        self._cur = dict(meta)

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            if self._cur is not None:
                self._cur[name] = self._cur.get(name, 0.0) + (time.perf_counter() - t) * 1000.0

    def add(self, name: str, ms: float):
        if self._cur is not None:
            self._cur[name] = self._cur.get(name, 0.0) + ms

    def end_frame(self, **meta):
        if self._cur is None:
            return
        self._cur["total"] = (time.perf_counter() - self._t0) * 1000.0
        self._cur.update(meta)
        self.rows.append(self._cur)
        self._cur = None

    def stage_names(self) -> list[str]:
        names: list[str] = []
        for r in self.rows:
            for k, v in r.items():
                if k not in names and isinstance(v, float):
                    names.append(k)
        return names

    def summary(self) -> dict:
        out = {}
        for name in self.stage_names():
            vals = np.array([r[name] for r in self.rows if isinstance(r.get(name), float)])
            if len(vals) == 0:
                continue
            out[name] = {
                "n": int(len(vals)),
                "median_ms": float(np.median(vals)),
                "p95_ms": float(np.percentile(vals, 95)),
                "mean_ms": float(np.mean(vals)),
                "max_ms": float(np.max(vals)),
            }
        return out

    def summary_table(self) -> str:
        s = self.summary()
        lines = ["| stage | n | median ms | p95 ms | max ms |", "|---|---|---|---|---|"]
        for k, v in s.items():
            lines.append(f"| {k} | {v['n']} | {v['median_ms']:.1f} | {v['p95_ms']:.1f} | {v['max_ms']:.1f} |")
        return "\n".join(lines)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.summary(), f, indent=2)
