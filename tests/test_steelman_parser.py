"""Unit tests for W7.2 K.2 STEELMAN protocol parser."""

from __future__ import annotations

from src.cdss.runtime.llm_bridge import _parse_steelman_protocol


def test_steelman_basic():
    raw = "STEELMAN|pulmonary_embolism|0.78|tachycardia + pleuritic pain + recent immobility"
    out = _parse_steelman_protocol(raw)
    assert out["label"] == "pulmonary_embolism"
    assert abs(out["argument_score"] - 0.78) < 1e-6
    assert "tachycardia" in out["rationale"]


def test_steelman_score_clamped():
    out = _parse_steelman_protocol("STEELMAN|x|1.7|over_one")
    assert out["argument_score"] == 1.0
    out2 = _parse_steelman_protocol("STEELMAN|x|-0.4|negative")
    assert out2["argument_score"] == 0.0


def test_steelman_picks_strongest():
    raw = (
        "STEELMAN|weak_dx|0.20|weak\n"
        "STEELMAN|strong_dx|0.85|strong evidence\n"
        "STEELMAN|mid_dx|0.50|mid\n"
    )
    out = _parse_steelman_protocol(raw)
    assert out["label"] == "strong_dx"
    assert abs(out["argument_score"] - 0.85) < 1e-6


def test_steelman_ignores_garbage():
    raw = "garbage line\nALT|x|0.5|wrong format\nSTEELMAN|valid|0.4|ok"
    out = _parse_steelman_protocol(raw)
    assert out["label"] == "valid"


def test_steelman_empty():
    out = _parse_steelman_protocol("")
    assert out["label"] == ""
    assert out["argument_score"] == 0.0


def test_steelman_malformed_score_skipped():
    raw = "STEELMAN|bad|notanumber|x\nSTEELMAN|good|0.6|ok"
    out = _parse_steelman_protocol(raw)
    assert out["label"] == "good"
