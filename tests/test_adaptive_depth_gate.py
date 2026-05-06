"""Unit tests for adaptive-depth + abstention policy gates."""

from __future__ import annotations

import pytest

from src.cdss.runtime.policy_gates import (
    AbstentionDecision,
    GateSignals,
    abstain_or_escalate,
    should_skip_deep_simulation,
)


def test_skip_deep_simulation_when_all_conditions_met():
    signals = GateSignals(
        top1_score=0.85,
        runner_up_score=0.50,
        grounding_pass_rate=0.95,
        grounding_risk=0.05,
        evidence_coverage=0.7,
        has_red_flags=False,
        evidence_starvation=False,
        urgency="ROUTINE",
    )
    assert should_skip_deep_simulation(signals) is True


def test_skip_blocked_by_red_flags():
    signals = GateSignals(
        top1_score=0.85, runner_up_score=0.50,
        grounding_pass_rate=0.95, evidence_coverage=0.7,
        has_red_flags=True, urgency="ROUTINE",
    )
    assert should_skip_deep_simulation(signals) is False


def test_skip_blocked_by_emergency_urgency():
    signals = GateSignals(
        top1_score=0.85, runner_up_score=0.50,
        grounding_pass_rate=0.95, evidence_coverage=0.7,
        urgency="EMERGENCY",
    )
    assert should_skip_deep_simulation(signals) is False


def test_skip_blocked_by_low_top_score():
    signals = GateSignals(
        top1_score=0.5, runner_up_score=0.20,
        grounding_pass_rate=0.95, evidence_coverage=0.7,
        urgency="ROUTINE",
    )
    assert should_skip_deep_simulation(signals) is False


def test_skip_blocked_by_starvation():
    signals = GateSignals(
        top1_score=0.85, runner_up_score=0.30,
        grounding_pass_rate=0.95, evidence_coverage=0.7,
        evidence_starvation=True, urgency="ROUTINE",
    )
    assert should_skip_deep_simulation(signals) is False


def test_abstain_when_low_margin_and_high_risk():
    signals = GateSignals(
        top1_score=0.4, runner_up_score=0.35,
        grounding_pass_rate=0.4, grounding_risk=0.6,
    )
    decision = abstain_or_escalate(signals)
    assert decision.abstain is True
    assert "high_grounding_risk" in decision.reason or "starvation" in decision.reason


def test_abstain_when_low_margin_and_starved():
    signals = GateSignals(
        top1_score=0.4, runner_up_score=0.38,
        grounding_pass_rate=1.0, grounding_risk=0.0,
        evidence_starvation=True,
    )
    decision = abstain_or_escalate(signals)
    assert decision.abstain is True
    assert "starvation" in decision.reason


def test_no_abstention_when_margin_wide():
    signals = GateSignals(
        top1_score=0.8, runner_up_score=0.4,
        grounding_pass_rate=0.4, grounding_risk=0.6,
    )
    decision = abstain_or_escalate(signals)
    assert decision.abstain is False


def test_no_abstention_when_grounding_clean():
    signals = GateSignals(
        top1_score=0.4, runner_up_score=0.39,
        grounding_pass_rate=1.0, grounding_risk=0.0,
        evidence_starvation=False,
    )
    decision = abstain_or_escalate(signals)
    assert decision.abstain is False


def test_signals_margin_property():
    signals = GateSignals(top1_score=0.8, runner_up_score=0.5)
    assert signals.margin == pytest.approx(0.3)


def test_abstention_decision_records_margin_and_risk():
    signals = GateSignals(top1_score=0.4, runner_up_score=0.35, grounding_risk=0.6)
    decision = abstain_or_escalate(signals)
    assert decision.margin == pytest.approx(0.05, abs=0.01)
    assert decision.grounding_risk == pytest.approx(0.6)
