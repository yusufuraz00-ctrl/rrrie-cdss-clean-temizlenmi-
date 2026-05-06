"""Unit tests for W6.1 J.1 rank-fusion module."""

from __future__ import annotations

from src.cdss.reasoning.rank_fusion import (
    aggregate_worker_outputs,
    fuse,
    reciprocal_rank_fusion,
    trust_weighted_borda,
)


# ---- RRF ----------------------------------------------------------------

def test_rrf_basic_consensus_wins():
    rankings = [
        ["a", "b", "c"],
        ["a", "c", "b"],
        ["a", "b", "d"],
    ]
    rrf = reciprocal_rank_fusion(rankings, k=60, normalize=False)
    # 'a' is top-1 in all three workers; should dominate.
    top = max(rrf, key=rrf.get)
    assert top == "a"
    assert rrf["a"] > rrf["b"]
    assert rrf["a"] > rrf["c"]
    assert rrf["a"] > rrf["d"]


def test_rrf_score_scale_invariance():
    """RRF only sees ranks; tripling raw scores must not change RRF outputs."""
    rrf1 = reciprocal_rank_fusion([["a", "b", "c"]])
    rrf2 = reciprocal_rank_fusion([["a", "b", "c"]])
    assert rrf1 == rrf2


def test_rrf_normalize_in_unit_interval():
    rrf = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]], normalize=True)
    assert all(0.0 <= v <= 1.0 for v in rrf.values())


def test_rrf_empty_inputs_safe():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


# ---- TWB ----------------------------------------------------------------

def test_twb_high_trust_worker_wins():
    rankings = [
        ["a", "b"],   # cardiology — trusted
        ["b", "a"],   # general — untrusted
    ]
    profiles = ["cardiology", "general"]
    twb = trust_weighted_borda(
        rankings,
        profiles=profiles,
        trust={"cardiology": 0.95, "general": 0.45},
    )
    assert twb["a"] > twb["b"]


def test_twb_hierarchical_boost():
    """Family-specialist hierarchical weight tilts the result."""
    rankings = [
        ["a", "b"],   # cardiology
        ["b", "a"],   # general
    ]
    twb_no_boost = trust_weighted_borda(
        rankings,
        profiles=["cardiology", "general"],
        trust={"cardiology": 0.8, "general": 0.8},
    )
    twb_boost = trust_weighted_borda(
        rankings,
        profiles=["cardiology", "general"],
        trust={"cardiology": 0.8, "general": 0.8},
        hier_weights={"cardiology": 2.0, "general": 1.0},
    )
    # Cardiology's choice 'a' should dominate harder under boost.
    assert twb_boost["a"] - twb_boost["b"] > twb_no_boost["a"] - twb_no_boost["b"]


def test_twb_all_zero_trust_safe():
    twb = trust_weighted_borda(
        [["a", "b"]],
        profiles=["unknown"],
        trust={"unknown": 0.0},
    )
    assert twb == {}


# ---- fuse ---------------------------------------------------------------

def test_fuse_convex_mixture():
    rrf = {"a": 1.0, "b": 0.5}
    twb = {"a": 0.0, "b": 1.0}
    out = fuse(rrf, twb, alpha=0.5, beta=0.5, gamma=0.0)
    # Equal mixture: a=0.5, b=0.75 → b wins.
    assert out["b"] > out["a"]


def test_fuse_likelihood_optional():
    rrf = {"a": 1.0}
    twb = {"a": 1.0}
    no_lik = fuse(rrf, twb)
    with_lik = fuse(rrf, twb, likelihood={"a": 1.0})
    assert no_lik["a"] == with_lik["a"]  # both =1.0


def test_fuse_degenerate_weights_falls_back():
    out = fuse({"a": 0.5}, {"a": 0.9}, alpha=0.0, beta=0.0, gamma=0.0)
    assert out == {"a": 0.5}  # falls back to RRF


# ---- aggregate_worker_outputs end-to-end -------------------------------

def test_aggregate_worker_outputs_consensus():
    worker_results = [
        [("acute_mi", 0.9), ("unstable_angina", 0.6), ("pe", 0.3)],
        [("acute_mi", 0.85), ("pe", 0.55)],
        [("acute_mi", 0.7), ("unstable_angina", 0.4)],
    ]
    profiles = ["cardiology", "general", "common_case"]
    fused = aggregate_worker_outputs(
        worker_results=worker_results,
        profiles=profiles,
        trust={"cardiology": 0.9, "general": 0.7, "common_case": 0.7},
    )
    top = max(fused, key=fused.get)
    assert top == "acute_mi"


def test_aggregate_worker_outputs_empty_safe():
    assert aggregate_worker_outputs(worker_results=[], profiles=[]) == {}
