"""Unit tests for W7 K.4/K.5 challenger output parsers."""

from __future__ import annotations

from src.cdss.runtime.llm_bridge import (
    _parse_conflict_resolution_protocol,
    _parse_pairwise_judge_protocol,
)


# ── pairwise_judge parser ────────────────────────────────────────────────


def test_pairwise_basic():
    out = _parse_pairwise_judge_protocol("WIN|acute_mi|0.72|elevated trop + LBBB")
    assert out["winner"] == "acute_mi"
    assert abs(out["prob"] - 0.72) < 1e-6
    assert "trop" in out["rationale"]


def test_pairwise_prob_clamp():
    out = _parse_pairwise_judge_protocol("WIN|x|1.4|over_one")
    assert out["prob"] == 1.0
    out2 = _parse_pairwise_judge_protocol("WIN|x|-0.2|negative")
    assert out2["prob"] == 0.0


def test_pairwise_picks_last_valid():
    raw = "WIN|first|0.3|early\nWIN|second|0.6|later"
    out = _parse_pairwise_judge_protocol(raw)
    assert out["winner"] == "second"
    assert abs(out["prob"] - 0.6) < 1e-6


def test_pairwise_empty():
    out = _parse_pairwise_judge_protocol("")
    assert out["winner"] == ""
    assert out["prob"] == 0.5


def test_pairwise_skips_garbage():
    raw = "garbage\nALT|wrong\nWIN|good|0.55|ok"
    out = _parse_pairwise_judge_protocol(raw)
    assert out["winner"] == "good"


# ── premise-conflict resolver parser ─────────────────────────────────────


def test_conflict_full():
    raw = (
        "ARTIFACT|premise_a|likely transcription noise\n"
        "DISCRIMINATOR|troponin|elevated rules in MI\n"
        "DECISION|escalate|both premises plausible\n"
    )
    out = _parse_conflict_resolution_protocol(raw)
    assert out["artifact_premise"] == "premise_a"
    assert "transcription" in out["artifact_rationale"]
    assert out["discriminator"] == "troponin"
    assert out["decision"] == "escalate"


def test_conflict_decision_normalizes_lower():
    out = _parse_conflict_resolution_protocol("DECISION|KEEP|reasonable")
    assert out["decision"] == "keep"


def test_conflict_partial_ok():
    raw = "DISCRIMINATOR|d-dimer|negative rules out PE"
    out = _parse_conflict_resolution_protocol(raw)
    assert out["discriminator"] == "d-dimer"
    assert out["artifact_premise"] == ""
    assert out["decision"] == ""


def test_conflict_empty():
    out = _parse_conflict_resolution_protocol("")
    assert out["decision"] == ""
    assert out["discriminator"] == ""
