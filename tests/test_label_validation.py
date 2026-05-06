"""Unit tests for W6.1 J.6 inference-time label validator.

Validator drops LLM-hallucinated symptom-concat ids and projects near-misses
onto the registry via Jaccard token overlap. Pure function — no LLM.
"""

from __future__ import annotations

from src.cdss.runtime.llm_bridge import _validate_rank_candidates


def test_disabled_passthrough():
    cands = [{"label": "anything_at_all", "score": 0.5, "rationale": "x"}]
    kept, dropped = _validate_rank_candidates(cands, enabled=False)
    assert kept == cands
    assert dropped == 0


def test_explicit_candidate_set_match():
    """Exact id match against the supplied candidate_set should pass even
    when no live registry is required (registry presence is required to
    short-circuit, but pool-based Jaccard always works)."""
    cands = [{"label": "acute_mi", "score": 0.8, "rationale": "x"}]
    kept, dropped = _validate_rank_candidates(
        cands,
        enabled=True,
        candidate_set=["acute_mi", "pulmonary_embolism", "aortic_dissection"],
    )
    # Either accepted directly via registry or via Jaccard=1.0 against pool.
    assert dropped == 0
    assert len(kept) == 1
    assert kept[0]["label"] == "acute_mi"


def test_jaccard_near_miss_projects():
    """LLM mis-stems 'acute mi cardio' → projects to 'acute_mi' if Jaccard ≥ 0.6."""
    cands = [{"label": "acute mi", "score": 0.7, "rationale": "x"}]
    kept, dropped = _validate_rank_candidates(
        cands,
        enabled=True,
        candidate_set=["acute_mi", "pulmonary_embolism"],
    )
    assert dropped == 0
    assert len(kept) == 1
    assert kept[0]["label"] == "acute_mi"


def test_symptom_concat_hallucination_dropped():
    """The plan calls these out by name — they MUST be dropped."""
    cands = [
        {
            "label": "dysphagia_food_impaction_sensation_frequent_belching",
            "score": 0.6,
            "rationale": "fake",
        }
    ]
    kept, dropped = _validate_rank_candidates(
        cands,
        enabled=True,
        candidate_set=["acute_mi", "pulmonary_embolism", "gerd"],
    )
    assert dropped == 1
    assert kept == []


def test_normalization_preserves_id():
    cands = [{"label": "Acute MI", "score": 0.7, "rationale": "x"}]
    kept, _ = _validate_rank_candidates(
        cands,
        enabled=True,
        candidate_set=["acute_mi"],
    )
    assert kept and kept[0]["label"] == "acute_mi"


def test_empty_inputs_safe():
    kept, dropped = _validate_rank_candidates([], enabled=True)
    assert kept == [] and dropped == 0
