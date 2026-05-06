"""Unit tests for the typed eval-metrics functions."""

from __future__ import annotations

import pytest

from tests.eval.eval_metrics import (
    AbstentionRecord,
    EvalRecord,
    abstention_precision,
    calibration_ece,
    cohort_report,
    family_hit_at_k,
    family_hit_at_k_cohort,
    hallucination_rate,
    margin_histogram,
    reliability_diagram,
    safety_correctness,
    safety_correctness_cohort,
)


def _packet(candidates, urgency: str = "ROUTINE", grounding_pass_rate=None):
    p = {
        "differential": {"candidates": candidates},
        "risk_profile": {"urgency": urgency},
    }
    if grounding_pass_rate is not None:
        p["typed_case_bundle"] = {"inline_grounding_pass_rate": grounding_pass_rate}
    return p


def test_family_hit_at_k_top1_match():
    pkt = _packet([{"label": "x", "score": 0.7, "parent_category": "vascular"}])
    assert family_hit_at_k(pkt, "vascular", 1) == 1
    assert family_hit_at_k(pkt, "metabolic", 1) == 0


def test_family_hit_at_k_uses_label_when_family_blank():
    pkt = _packet([{"label": "migraine", "score": 0.7, "parent_category": ""}])
    assert family_hit_at_k(pkt, "migraine", 1) == 1


def test_family_hit_cohort_mean():
    a = _packet([{"label": "a", "parent_category": "neuro"}])
    b = _packet([{"label": "b", "parent_category": "cardio"}])
    pairs = [(a, "neuro"), (b, "neuro")]
    assert family_hit_at_k_cohort(pairs, 1) == pytest.approx(0.5)


def test_calibration_ece_perfect():
    confidences = [0.1, 0.5, 0.9]
    correctness = [0, 1, 1]
    # Slight imperfection because each bin holds only one case.
    assert 0.0 <= calibration_ece(confidences, correctness, n_bins=10) <= 0.5


def test_calibration_ece_total_miscalibration():
    confidences = [1.0, 1.0, 1.0]
    correctness = [0, 0, 0]
    assert calibration_ece(confidences, correctness, n_bins=10) == pytest.approx(1.0)


def test_calibration_ece_empty_returns_zero():
    assert calibration_ece([], []) == pytest.approx(0.0)


def test_reliability_diagram_shape():
    diagram = reliability_diagram([0.1, 0.55, 0.9], [0, 1, 1], n_bins=5)
    assert len(diagram) == 5
    assert all("count" in row for row in diagram)


def test_hallucination_rate_uses_pass_rate():
    packets = [
        _packet([], grounding_pass_rate=1.0),
        _packet([], grounding_pass_rate=0.5),
        _packet([], grounding_pass_rate=0.0),
    ]
    out = hallucination_rate(packets)
    assert out["hallucination_rate"] == pytest.approx(1.0 - 0.5)
    assert out["covered_packets"] == 3


def test_hallucination_rate_empty_returns_zero():
    out = hallucination_rate([_packet([])])
    assert out["hallucination_rate"] == pytest.approx(0.0)
    assert out["covered_packets"] == 0


def test_safety_correctness_passes_when_urgent_and_top3():
    pkt = _packet(
        [
            {"label": "aortic_dissection", "score": 0.9, "parent_category": "vascular"},
        ],
        urgency="URGENT",
    )
    out = safety_correctness(pkt, "vascular", must_not_miss=True)
    assert out["passed"] is True


def test_safety_correctness_fails_when_routine():
    pkt = _packet(
        [{"label": "x", "score": 0.5, "parent_category": "vascular"}],
        urgency="ROUTINE",
    )
    out = safety_correctness(pkt, "vascular", must_not_miss=True)
    assert out["passed"] is False


def test_safety_correctness_skip_when_not_must_not_miss():
    pkt = _packet([{"label": "x", "parent_category": "neuro"}], urgency="ROUTINE")
    out = safety_correctness(pkt, "neuro", must_not_miss=False)
    assert out["applicable"] is False
    assert out["passed"] is True


def test_safety_correctness_cohort_pass_rate():
    a_pkt = _packet([{"label": "a", "parent_category": "vasc"}], urgency="EMERGENCY")
    b_pkt = _packet([{"label": "b", "parent_category": "neuro"}], urgency="ROUTINE")
    cohort = [(a_pkt, "vasc", True), (b_pkt, "neuro", True)]
    out = safety_correctness_cohort(cohort)
    assert out["safety_pass_rate"] == pytest.approx(0.5)
    assert out["applicable_cases"] == 2


def test_abstention_precision_rewards_correct_abstention():
    records = [
        AbstentionRecord(abstained=True, top1_family="x", expected_family="y"),
        AbstentionRecord(abstained=True, top1_family="x", expected_family="x"),
        AbstentionRecord(abstained=False, top1_family="z", expected_family="z"),
    ]
    out = abstention_precision(records)
    # 1 of 2 abstentions was correct (top-1 was wrong)
    assert out["abstention_precision"] == pytest.approx(0.5)
    assert out["abstentions"] == 2


def test_abstention_precision_no_abstentions():
    records = [AbstentionRecord(abstained=False, top1_family="a", expected_family="a")]
    out = abstention_precision(records)
    assert out["abstention_precision"] == pytest.approx(0.0)
    assert out["abstentions"] == 0


def test_margin_histogram_buckets():
    packets = [
        _packet([{"label": "x", "score": 0.9}, {"label": "y", "score": 0.85}]),  # margin 0.05
        _packet([{"label": "x", "score": 0.7}, {"label": "y", "score": 0.3}]),   # margin 0.40
        _packet([{"label": "x", "score": 0.5}]),  # margin 0.50 (single candidate)
    ]
    hist = margin_histogram(packets)
    total = sum(hist.values())
    assert total == 3


def test_cohort_report_assembles_all_metrics():
    pkt = _packet(
        [{"label": "x", "score": 0.8, "parent_category": "fam"}],
        urgency="EMERGENCY",
        grounding_pass_rate=1.0,
    )
    record = EvalRecord(
        case_id="c1",
        expected_family="fam",
        must_not_miss=True,
        packet=pkt,
        abstained=False,
        top1_family="fam",
        confidence=0.8,
    )
    report = cohort_report([record])
    assert report["n"] == 1
    assert report["family_hit_at_1"] == pytest.approx(1.0)
    assert "calibration_ece" in report
    assert report["safety"]["safety_pass_rate"] == pytest.approx(1.0)
    assert "margin_histogram" in report
