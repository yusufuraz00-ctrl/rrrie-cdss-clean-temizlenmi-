"""Unit tests for W6.2 J.3 worker_trust EMA + persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cdss.reasoning import worker_trust as wt


@pytest.fixture
def tmp_trust_path(tmp_path: Path) -> Path:
    return tmp_path / "worker_reliability.json"


def test_quality_score_top1():
    assert wt.quality_score(["acute_mi", "pe", "gerd"], "acute_mi") == 1.0


def test_quality_score_top3():
    assert wt.quality_score(["pe", "acute_mi", "gerd"], "acute_mi") == 0.5


def test_quality_score_miss():
    assert wt.quality_score(["pe", "gerd", "panic"], "acute_mi") == 0.0


def test_quality_score_empty_safe():
    assert wt.quality_score([], "acute_mi") == 0.0
    assert wt.quality_score(["x"], "") == 0.0


def test_ema_pulls_toward_quality(tmp_trust_path):
    # cold-start → COLD_TRUST = 0.8
    new = wt.update("cardiology", "cardiovascular_syndrome", 1.0, path=tmp_trust_path)
    expected = wt.ETA * 1.0 + (1.0 - wt.ETA) * wt.COLD_TRUST  # 0.7 + 0.24 = 0.94
    assert abs(new - expected) < 1e-3


def test_ema_floors_at_trust_floor(tmp_trust_path):
    # 5 successive zero-quality updates from cold start.
    for _ in range(5):
        wt.update("worst_worker", "any_family", 0.0, path=tmp_trust_path)
    data = json.loads(tmp_trust_path.read_text(encoding="utf-8"))
    final = data["trust"]["any_family"]["worst_worker"]
    assert final >= wt.TRUST_FLOOR - 1e-6


def test_persistence_round_trip(tmp_trust_path):
    wt.update("cardiology", "cardiovascular_syndrome", 1.0, path=tmp_trust_path)
    wt.update("zebra_hunter", "metabolic_decompensation", 0.5, path=tmp_trust_path)
    cv = wt.load_worker_trust("cardiovascular_syndrome", path=tmp_trust_path)
    me = wt.load_worker_trust("metabolic_decompensation", path=tmp_trust_path)
    assert "cardiology" in cv
    assert "zebra_hunter" in me
    assert cv["cardiology"] > wt.COLD_TRUST


def test_load_unknown_family_falls_back_to_average(tmp_trust_path):
    wt.update("cardiology", "cardiovascular_syndrome", 1.0, path=tmp_trust_path)
    wt.update("cardiology", "respiratory_syndrome", 1.0, path=tmp_trust_path)
    # ask for an unseen family — should average across known families.
    out = wt.load_worker_trust("unknown_family", path=tmp_trust_path)
    assert "cardiology" in out
    assert wt.COLD_TRUST < out["cardiology"] <= 1.0


def test_load_empty_returns_empty(tmp_trust_path):
    assert wt.load_worker_trust("any", path=tmp_trust_path) == {}


def test_update_panel_one_write(tmp_trust_path):
    out = wt.update_panel(
        ["cardiology", "general", "zebra_hunter"],
        "cardiovascular_syndrome",
        {"cardiology": 1.0, "general": 0.5, "zebra_hunter": 0.0},
        path=tmp_trust_path,
    )
    assert set(out.keys()) >= {"cardiology", "general", "zebra_hunter"}
    assert out["cardiology"] > out["general"] > out["zebra_hunter"]
