"""The walk diagnostic logger writes the columns the report depends on."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loka.diagnose_walk import (
    COLUMNS,
    format_report,
    load_trace,
    record_walk,
    save_trace,
    summarize,
)


def test_short_walk_records_every_declared_column(tmp_path: Path):
    trace = record_walk(speed=0.20, duration=1.2, decimate=10)
    assert set(trace.columns) == set(COLUMNS)
    assert trace.n > 20
    assert trace["t"][0] < trace["t"][-1]
    # Opening transfer is step 0; a 1.2 s walk must have left it.
    assert trace["step"].max() >= 1

    path = tmp_path / "walk.npz"
    save_trace(trace, path)
    loaded = load_trace(path)
    np.testing.assert_allclose(loaded["com_l"], trace["com_l"])
    assert loaded.fell == trace.fell

    summary = summarize(trace)
    assert summary["samples"] == trace.n
    assert "cop_edge_p95" in summary
    report = format_report(trace)
    assert "Hypothesis checks" in report
    assert "H-sat" in report
