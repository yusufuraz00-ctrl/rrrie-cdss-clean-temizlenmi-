"""Unit tests for W7.2 K.3 counterfactual finding-ablation probe."""

from __future__ import annotations

from src.cdss.contracts.models import DiagnosticBelief, EvidenceDelta
from src.cdss.reasoning.counterfactual_ablation import find_pivot_findings


def _belief(species_post: dict, deltas: list[EvidenceDelta]) -> DiagnosticBelief:
    return DiagnosticBelief(
        family_posterior={},
        family_alpha={},
        genus_posterior={},
        genus_alpha={},
        species_posterior=species_post,
        species_alpha={k: v * 100.0 for k, v in species_post.items()},
        ds_mass={},
        active_level=2,
        evidence_log=deltas,
        entropy_history=[],
        miss_risk_history=[],
        step=len(deltas),
    )


def test_no_evidence_no_pivots():
    out = find_pivot_findings(_belief({"a": 0.6, "b": 0.4}, []))
    assert out["pivot_findings"] == []
    assert out["top1_current"] == "a"


def test_pivot_when_finding_flips_top1():
    # Posterior favors A. Removing finding f1 (which heavily favored A) flips to B.
    delta = EvidenceDelta(
        source="swarm",
        finding="dyspnea",
        step=1,
        level=2,
        likelihoods={"a": 0.95, "b": 0.10},  # ablation divides by these
        likelihood_var={},
        posterior_before={"a": 0.5, "b": 0.5},
        posterior_after={"a": 0.6, "b": 0.4},
        entropy_before=1.0,
        entropy_after=0.7,
    )
    belief = _belief({"a": 0.55, "b": 0.45}, [delta])
    out = find_pivot_findings(belief, level=2)
    assert out["top1_current"] == "a"
    assert len(out["pivot_findings"]) == 1
    assert out["pivot_findings"][0]["from_top1"] == "a"
    assert out["pivot_findings"][0]["to_top1"] == "b"


def test_no_pivot_when_finding_consistent_with_top1():
    # Removing this finding doesn't flip top-1 (B's lead is independent).
    delta = EvidenceDelta(
        source="swarm",
        finding="cough",
        step=1,
        level=2,
        likelihoods={"a": 0.5, "b": 0.5},  # equal — ablation no-op
        likelihood_var={},
        posterior_before={"a": 0.3, "b": 0.7},
        posterior_after={"a": 0.3, "b": 0.7},
        entropy_before=0.88,
        entropy_after=0.88,
    )
    belief = _belief({"a": 0.3, "b": 0.7}, [delta])
    out = find_pivot_findings(belief, level=2)
    assert out["pivot_findings"] == []


def test_absence_dependent_flag():
    deltas = []
    for i, name in enumerate(["neg:fever", "neg:cough"], 1):
        deltas.append(
            EvidenceDelta(
                source="swarm",
                finding=name,
                step=i,
                level=2,
                likelihoods={"a": 0.9, "b": 0.1},
                likelihood_var={},
                posterior_before={"a": 0.5, "b": 0.5},
                posterior_after={"a": 0.6, "b": 0.4},
                entropy_before=1.0,
                entropy_after=0.6,
            )
        )
    belief = _belief({"a": 0.55, "b": 0.45}, deltas)
    out = find_pivot_findings(belief, level=2)
    assert len(out["pivot_findings"]) >= 2
    assert all(p["is_negative"] for p in out["pivot_findings"])
    assert out["absence_dependent"] is True


def test_top_n_limit():
    deltas = []
    for i in range(8):
        deltas.append(
            EvidenceDelta(
                source="swarm",
                finding=f"f{i}",
                step=i + 1,
                level=2,
                likelihoods={"a": 0.9, "b": 0.1},
                likelihood_var={},
                posterior_before={"a": 0.5, "b": 0.5},
                posterior_after={"a": 0.6, "b": 0.4},
                entropy_before=1.0 - (i * 0.05),
                entropy_after=0.5 - (i * 0.05),
            )
        )
    belief = _belief({"a": 0.55, "b": 0.45}, deltas)
    out = find_pivot_findings(belief, level=2, top_n=3)
    # capped at top-3 most-influential deltas (all flip → 3 pivots)
    assert len(out["pivot_findings"]) <= 3


def test_empty_posterior_safe():
    belief = _belief({}, [])
    out = find_pivot_findings(belief, level=2)
    assert out["pivot_findings"] == []
    assert out["top1_current"] == ""
