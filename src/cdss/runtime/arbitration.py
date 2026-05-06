"""Late-fusion arbitration between LLM reasoning and support-side signals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.cdss.clinical.diagnosis_normalization import normalize_differential
from src.cdss.clinical.finding_fit import score_candidate_finding_fit
from src.cdss.contracts.models import (
    CandidateScoreBreakdown,
    DecisionTrace,
    DifferentialCandidate,
    DifferentialSet,
    EvidenceBundle,
    ExplanationGraph,
    FactGraph,
    HypothesisFrontier,
    HypothesisNode,
    InterventionSafetyAssessment,
    InterventionSet,
    ModelSupportSignals,
    RetrievalRankingStats,
    StructuredFindings,
)
from src.cdss.core.state import StatePatch
from src.cdss.reasoning.clinical_scoring import EpiPriorResult
from src.cdss.reasoning.specificity_calibrator import SpecificityJudgment
from src.cdss.reasoning.trigger_compatibility import trigger_compatibility_adjustment
from src.cdss.runtime.policy import CdssRuntimePolicy, load_runtime_policy


@dataclass(frozen=True)
class _StateFitCapPolicy:
    severe_mismatch_state_fit_max: float = 0.2
    severe_mismatch_contradiction_min: float = 0.38
    severe_mismatch_explanation_debt_min: float = 0.58
    must_not_miss_score_cap: float = 0.45
    default_score_multiplier: float = 0.78
    default_score_absolute_cap: float = 0.34


@dataclass(frozen=True)
class _StateMismatchPenaltyPolicy:
    catastrophic_penalty: float = 0.14
    catastrophic_state_fit_max: float = 0.2
    cross_system_penalty_primary: float = 0.12
    cross_system_state_fit_max: float = 0.28
    cross_system_explanation_debt_min: float = 0.55
    benign_intrusion_penalty: float = 0.08
    benign_intrusion_state_fit_max: float = 0.34
    benign_intrusion_contradiction_min: float = 0.2
    cross_system_penalty_secondary: float = 0.08
    secondary_cross_system_state_fit_max: float = 0.22
    secondary_cross_system_contradiction_min: float = 0.26


@dataclass(frozen=True)
class _AnchorLockTier:
    state_gap_min: float
    contradiction_gap_min: float
    penalty: float


@dataclass(frozen=True)
class _AnchorLockPolicy:
    state_fit_weight: float = 0.62
    contradiction_weight: float = 0.22
    explanation_debt_weight: float = 0.16
    pass_state_fit_min: float = 0.34
    pass_contradiction_max: float = 0.32
    tiers: tuple[_AnchorLockTier, ...] = field(default_factory=lambda: (
        _AnchorLockTier(state_gap_min=0.22, contradiction_gap_min=0.08, penalty=0.14),
        _AnchorLockTier(state_gap_min=0.16, contradiction_gap_min=0.04, penalty=0.08),
    ))


@dataclass(frozen=True)
class _SiblingDedupPolicy:
    overlap_threshold: float = 0.6
    absorb_bonus_per_sibling: float = 0.02
    absorb_bonus_cap: float = 0.06


@dataclass(frozen=True)
class _AdjustmentPolicy:
    state_fit_hard_cap: _StateFitCapPolicy = field(default_factory=_StateFitCapPolicy)
    state_mismatch_penalty: _StateMismatchPenaltyPolicy = field(default_factory=_StateMismatchPenaltyPolicy)
    anchor_lock: _AnchorLockPolicy = field(default_factory=_AnchorLockPolicy)
    sibling_dedup: _SiblingDedupPolicy = field(default_factory=_SiblingDedupPolicy)


def _load_adjustment_policy() -> _AdjustmentPolicy:
    """Load arbitration policy from config/adjustment_policy.json; fall back to defaults."""
    config_path = Path(__file__).resolve().parents[3] / "config" / "adjustment_policy.json"
    if not config_path.exists():
        return _AdjustmentPolicy()
    try:
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        cap_raw = raw.get("state_fit_hard_cap", {})
        mismatch_raw = raw.get("state_mismatch_penalty", {})
        anchor_raw = raw.get("anchor_lock_penalty", {})
        sibling_raw = raw.get("sibling_dedup", {})
        tiers = tuple(
            _AnchorLockTier(**{k: float(v) for k, v in t.items()})
            for t in anchor_raw.get("tiers", [])
        ) or _AnchorLockPolicy().tiers
        return _AdjustmentPolicy(
            state_fit_hard_cap=_StateFitCapPolicy(**{k: float(v) for k, v in cap_raw.items() if not isinstance(v, dict)}),
            state_mismatch_penalty=_StateMismatchPenaltyPolicy(**{k: float(v) for k, v in mismatch_raw.items() if not isinstance(v, dict)}),
            anchor_lock=_AnchorLockPolicy(
                state_fit_weight=float(anchor_raw.get("state_fit_weight", 0.62)),
                contradiction_weight=float(anchor_raw.get("contradiction_weight", 0.22)),
                explanation_debt_weight=float(anchor_raw.get("explanation_debt_weight", 0.16)),
                pass_state_fit_min=float(anchor_raw.get("pass_state_fit_min", 0.34)),
                pass_contradiction_max=float(anchor_raw.get("pass_contradiction_max", 0.32)),
                tiers=tiers,
            ),
            sibling_dedup=_SiblingDedupPolicy(**{k: float(v) for k, v in sibling_raw.items()}),
        )
    except Exception:
        return _AdjustmentPolicy()


_POLICY = _load_adjustment_policy()


GENERIC_BUCKET_LABELS = {
    "cardiorespiratory_process",
    "infectious_inflammatory_process",
    "metabolic_or_abdominal_process",
    "neurologic_process",
    "undifferentiated_high_variance_process",
}


def _is_generic_label(label: str) -> bool:
    return str(label or "") in GENERIC_BUCKET_LABELS or str(label or "").endswith("_process")


def _updated_uncertainty(candidates: list[DifferentialCandidate]) -> float:
    if not candidates:
        return 0.9
    uncertainty = max(0.06, round(1.0 - candidates[0].score, 2))
    if len(candidates) > 1 and abs(candidates[0].score - candidates[1].score) < 0.08:
        uncertainty = min(0.94, round(uncertainty + 0.08, 2))
    if not _is_generic_label(candidates[0].label):
        uncertainty = max(0.05, round(uncertainty - 0.06, 2))
    return uncertainty


def _state_mismatch_penalty(label: str, explanation_graph: ExplanationGraph, breakdown: CandidateScoreBreakdown) -> tuple[float, float, float]:
    del label
    p = _POLICY.state_mismatch_penalty
    catastrophic_penalty = 0.0
    benign_intrusion_penalty = 0.0
    cross_system_penalty = 0.0
    state_frames = explanation_graph.state_frames[:4]
    if state_frames and breakdown.state_fit < p.catastrophic_state_fit_max and explanation_graph.critical_unexplained_count > 0:
        catastrophic_penalty += p.catastrophic_penalty
    if breakdown.state_fit < p.cross_system_state_fit_max and breakdown.explanation_debt > p.cross_system_explanation_debt_min:
        cross_system_penalty += p.cross_system_penalty_primary
    if state_frames and breakdown.state_fit < p.benign_intrusion_state_fit_max and breakdown.contradiction_penalty > p.benign_intrusion_contradiction_min:
        benign_intrusion_penalty += p.benign_intrusion_penalty
    if breakdown.state_fit < p.secondary_cross_system_state_fit_max and breakdown.contradiction_penalty > p.secondary_cross_system_contradiction_min:
        cross_system_penalty += p.cross_system_penalty_secondary
    return round(catastrophic_penalty, 2), round(benign_intrusion_penalty, 2), round(cross_system_penalty, 2)


def _anchor_lock_penalty(
    *,
    label: str,
    anchor_label: str,
    breakdown: CandidateScoreBreakdown,
    adjudication: dict[str, CandidateScoreBreakdown],
) -> float:
    if not anchor_label or label != anchor_label:
        return 0.0
    ap = _POLICY.anchor_lock
    if breakdown.state_fit >= ap.pass_state_fit_min and breakdown.contradiction_penalty <= ap.pass_contradiction_max:
        return 0.0

    rivals = [item for candidate_label, item in adjudication.items() if candidate_label != anchor_label]
    if not rivals:
        return 0.0
    best_rival = max(
        rivals,
        key=lambda item: (item.state_fit * ap.state_fit_weight)
            + ((1.0 - item.contradiction_penalty) * ap.contradiction_weight)
            + ((1.0 - item.explanation_debt) * ap.explanation_debt_weight),
    )
    state_gap = best_rival.state_fit - breakdown.state_fit
    contradiction_gap = breakdown.contradiction_penalty - best_rival.contradiction_penalty

    for tier in ap.tiers:
        if state_gap >= tier.state_gap_min and contradiction_gap >= tier.contradiction_gap_min:
            return tier.penalty
    return 0.0


def _state_fit_hard_cap(
    *,
    label: str,
    candidate_score: float,
    breakdown: CandidateScoreBreakdown,
    explanation_graph: ExplanationGraph,
    must_not_miss_labels: set[str],
) -> float | None:
    cp = _POLICY.state_fit_hard_cap
    severe_mismatch = (
        breakdown.state_fit < cp.severe_mismatch_state_fit_max
        and breakdown.contradiction_penalty >= cp.severe_mismatch_contradiction_min
        and breakdown.explanation_debt >= cp.severe_mismatch_explanation_debt_min
    )
    if not severe_mismatch or explanation_graph.critical_unexplained_count <= 0:
        return None
    if label in must_not_miss_labels:
        return round(min(candidate_score, cp.must_not_miss_score_cap), 2)
    return round(min(candidate_score * cp.default_score_multiplier, cp.default_score_absolute_cap), 2)


def _label_tokens(label: str) -> frozenset[str]:
    """Significant tokens (≥5 chars) from a snake_case label for sibling detection."""
    return frozenset(t for t in str(label or "").split("_") if len(t) >= 5)


def _sibling_overlap(a: str, b: str) -> float:
    """Jaccard-min overlap between two label token sets.  1.0 = identical tokens."""
    ta, tb = _label_tokens(a), _label_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _deduplicate_sibling_candidates(
    candidates: list[DifferentialCandidate],
    *,
    must_not_miss_labels: set[str],
    overlap_threshold: float | None = None,
    absorb_bonus_per_sibling: float | None = None,
    absorb_bonus_cap: float | None = None,
) -> list[DifferentialCandidate]:
    """Collapse near-duplicate label variants into the highest-scoring representative.

    Two candidates are considered siblings when their label token overlap (Jaccard-min,
    tokens ≥5 chars) exceeds *overlap_threshold*.  The higher-scored sibling survives;
    it receives a small score absorb bonus and its evidence_needed list is enriched with
    the absorbed sibling's evidence_needed items.  Must-not-miss candidates are never
    dropped — they keep a separate slot even if a sibling is present.
    """
    sp = _POLICY.sibling_dedup
    _overlap_threshold = overlap_threshold if overlap_threshold is not None else sp.overlap_threshold
    _absorb_bonus_per_sibling = absorb_bonus_per_sibling if absorb_bonus_per_sibling is not None else sp.absorb_bonus_per_sibling
    _absorb_bonus_cap = absorb_bonus_cap if absorb_bonus_cap is not None else sp.absorb_bonus_cap

    if len(candidates) <= 1:
        return candidates

    absorbed: set[int] = set()
    absorb_bonuses: dict[int, float] = {}

    for i in range(len(candidates)):
        if i in absorbed:
            continue
        for j in range(i + 1, len(candidates)):
            if j in absorbed:
                continue
            if _sibling_overlap(candidates[i].label, candidates[j].label) < _overlap_threshold:
                continue
            # They are siblings.  Keep higher-scored (i always wins here because
            # candidates are sorted descending before this call).
            loser_idx = j
            winner_idx = i
            if candidates[j].score > candidates[i].score:
                winner_idx, loser_idx = i, j
            # Never drop a must-not-miss candidate.
            if candidates[loser_idx].label in must_not_miss_labels:
                continue
            absorbed.add(loser_idx)
            absorb_bonuses[winner_idx] = round(
                min(_absorb_bonus_cap, absorb_bonuses.get(winner_idx, 0.0) + _absorb_bonus_per_sibling), 2
            )

    if not absorbed:
        return candidates

    result: list[DifferentialCandidate] = []
    for idx, candidate in enumerate(candidates):
        if idx in absorbed:
            continue
        bonus = absorb_bonuses.get(idx, 0.0)
        if bonus > 0.0:
            merged_evidence = list(dict.fromkeys(
                candidate.evidence_needed
                + [item for j in absorbed if _sibling_overlap(candidates[idx].label, candidates[j].label) >= _overlap_threshold for item in candidates[j].evidence_needed]
            ))[:5]
            candidate = candidate.model_copy(update={
                "score": round(min(0.99, candidate.score + bonus), 2),
                "evidence_needed": merged_evidence,
            })
        result.append(candidate)
    return result


def _helper_like_source(source: str) -> bool:
    del source
    return False


_WITHDRAWAL_SUBSTANCE_LABEL_TOKENS: frozenset[str] = frozenset({
    "withdrawal", "detox", "abstinence", "alcohol", "benzodiazepine",
    "opioid", "substance", "intoxication", "detoxification",
})


def _should_route_to_rule_out(
    candidate: DifferentialCandidate,
    final_score: float,
    breakdown: CandidateScoreBreakdown,
    *,
    is_must_not_miss: bool,
    epi_prior_tier: str | None,
    findings: StructuredFindings | None = None,
) -> bool:
    """Route candidate to rule_out_candidates when it is a must-not-miss with
    poor evidence fit AND the epidemiological prior classifies it as rare/exceptional.

    Also routes substance/withdrawal diagnoses to rule_out when no substance context
    is present in the case — these require a documented causative agent.
    """
    if not is_must_not_miss:
        return False

    # Substance/withdrawal diagnoses without any documented substance context
    # should not occupy top-1 — route to rule_out so the safety signal is preserved
    # but does not dominate the primary differential.
    label_tokens = frozenset(candidate.label.lower().split("_"))
    if label_tokens & _WITHDRAWAL_SUBSTANCE_LABEL_TOKENS:
        has_substance_context = False
        if findings is not None:
            substance_text = " ".join(
                list(findings.medications or []) + list(findings.exposures or [])
            ).lower()
            has_substance_context = any(
                tok in substance_text for tok in _WITHDRAWAL_SUBSTANCE_LABEL_TOKENS
            )
        if not has_substance_context and final_score < 0.55:
            return True

    poor_fit = breakdown.state_fit < 0.28 and breakdown.contradiction_penalty > 0.22
    low_prior = epi_prior_tier in {"rare", "exceptional"}
    low_score = final_score < 0.40
    return poor_fit and low_prior and low_score


def _verified_edge_counts(evidence: EvidenceBundle, label: str) -> tuple[int, int]:
    supports = 0
    refutes = 0
    if not evidence.items:
        return supports, refutes
    for item in evidence.items:
        if str(item.verification_status or "").strip().lower() != "verified":
            continue
        linked = {str(candidate).strip().lower() for candidate in (item.linked_hypotheses or []) if str(candidate).strip()}
        if str(label or "").strip().lower() not in linked:
            continue
        relation = str(item.relation_type or "").strip().lower()
        if relation == "supports":
            supports += 1
        elif relation == "refutes":
            refutes += 1
    return supports, refutes


def _order_frontier_nodes(nodes: list[HypothesisNode], *, limit: int = 5) -> list[HypothesisNode]:
    ordered = sorted(nodes, key=lambda item: item.score, reverse=True)
    primary = [node for node in ordered if not _helper_like_source(node.source)]
    helper = [node for node in ordered if _helper_like_source(node.source)]
    if primary:
        ordered = [*primary[:2], *helper[:1], *primary[2:], *helper[1:]]
    else:
        ordered = helper
    trimmed = ordered[:limit]
    for rank, node in enumerate(trimmed, start=1):
        node.rank = rank
    return trimmed


def _stabilize_candidate_order(
    candidates: list[DifferentialCandidate],
    *,
    prior_rank_by_label: dict[str, int],
    must_not_miss_labels: set[str],
    epi_prior_tier_map: dict[str, str] | None = None,
) -> list[DifferentialCandidate]:
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    if len(ordered) < 2:
        return ordered

    tier_map = epi_prior_tier_map or {}

    # Prefer safer and more specific labels when scores are effectively tied.
    for index in range(len(ordered) - 1):
        current = ordered[index]
        nxt = ordered[index + 1]
        score_delta = abs(float(current.score or 0.0) - float(nxt.score or 0.0))
        if score_delta > 0.02:
            # Extended epi window: allow common-beats-rare swap within a 0.08 gap.
            # Pattern 3 failures have rank-1 rare/exceptional ahead of rank-2 common
            # by 0.04–0.08 — beyond the old 0.02 tie-break, so they were never corrected.
            _cur_pre = tier_map.get(current.label, "uncommon")
            _nxt_pre = tier_map.get(nxt.label, "uncommon")
            if not (_cur_pre in {"rare", "exceptional"} and _nxt_pre == "common" and score_delta <= 0.08):
                continue

        # Epidemiological tie-break: common beats rare/exceptional regardless of must_not_miss.
        current_epi = tier_map.get(current.label, "uncommon")
        next_epi = tier_map.get(nxt.label, "uncommon")
        _rare_tiers = {"rare", "exceptional"}
        _common_tiers = {"common"}
        if current_epi in _rare_tiers and next_epi in _common_tiers:
            ordered[index], ordered[index + 1] = nxt, current
            continue
        if current_epi in _common_tiers and next_epi in _rare_tiers:
            # current already wins; don't swap
            continue

        # Must-not-miss promotion only when NO epidemiological counterweight.
        current_mnm = current.label in must_not_miss_labels
        next_mnm = nxt.label in must_not_miss_labels
        if (not current_mnm) and next_mnm and next_epi not in _rare_tiers:
            ordered[index], ordered[index + 1] = nxt, current
            continue

        current_generic = _is_generic_label(current.label)
        next_generic = _is_generic_label(nxt.label)
        if current_generic and not next_generic:
            ordered[index], ordered[index + 1] = nxt, current
            continue

        current_prior = prior_rank_by_label.get(current.label, 999)
        next_prior = prior_rank_by_label.get(nxt.label, 999)
        if current_prior <= 3 and next_prior <= 3 and next_prior < current_prior:
            ordered[index], ordered[index + 1] = nxt, current

    return ordered


import math as _math


def _softmax_normalize_scores(
    candidates: list,
    temperature: float = 0.65,
) -> dict[str, float]:
    """Temperature-scaled softmax normalization across candidate pool.

    Converts raw LLM scores to a proper probability distribution before arbitration.
    Temperature < 1.0 sharpens differences (amplifies leading candidates).
    Temperature > 1.0 smooths (reduces overconfidence when scores are clustered).
    Default 0.65 compresses overcrowded score ranges while preserving ordering.
    """
    if not candidates:
        return {}
    scores = [max(0.001, float(getattr(c, "score", 0.5))) for c in candidates]
    logits = [_math.log(s) / temperature for s in scores]
    max_logit = max(logits)
    exp_logits = [_math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    if total <= 0:
        return {getattr(c, "label", str(i)): 1.0 / len(candidates) for i, c in enumerate(candidates)}
    return {
        getattr(c, "label", str(i)): round(e / total, 4)
        for i, (c, e) in enumerate(zip(candidates, exp_logits))
    }


def _compute_competition_amplifier(
    score_spread: float,
    max_spread: float = 0.5,
    base_amplifier: float = 1.0,
    max_amplifier: float = 1.9,
) -> float:
    """Evidence-spread-aware amplifier for rare-vs-common competition penalty.

    When candidate scores are spread far apart, evidence discriminates strongly → amplify more.
    When scores are clustered (ambiguous case), be conservative → return near base_amplifier.
    Replaces the hardcoded 1.6× constant with a continuous evidence-adaptive function.

    Returns float in [base_amplifier, max_amplifier].
    """
    evidence_weight = min(1.0, max(0.0, score_spread / max_spread))
    return base_amplifier + (evidence_weight * (max_amplifier - base_amplifier))


class ArbitrationLayer:
    """Bounded score combiner that keeps LLM primary and support models secondary."""

    def __init__(self, policy: CdssRuntimePolicy | None = None) -> None:
        self.policy = policy or load_runtime_policy()

    def apply(
        self,
        *,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
        model_support: ModelSupportSignals,
        evidence: EvidenceBundle | None = None,
        retrieval_stats: RetrievalRankingStats,
        intervention_safety: InterventionSafetyAssessment,
        findings: StructuredFindings | None = None,
        fact_graph: FactGraph | None = None,
        interventions: InterventionSet | None = None,
        explanation_graph: ExplanationGraph | None = None,
        epi_prior: EpiPriorResult | None = None,
        specificity_judgments: dict[str, SpecificityJudgment] | None = None,
        rule_out_labels: set[str] | None = None,
    ) -> StatePatch:
        if not differential.candidates:
            return StatePatch()
        evidence = evidence or EvidenceBundle()
        findings = findings or StructuredFindings()
        fact_graph = fact_graph or FactGraph()
        interventions = interventions or InterventionSet()
        explanation_graph = explanation_graph or ExplanationGraph()
        normalization_result = normalize_differential(differential)
        differential = normalization_result.differential
        normalization_absorbed = normalization_result.absorbed

        # Build tier maps for fast lookup
        epi_tier_map: dict[str, str] = {}
        epi_weight_map: dict[str, float] = {}
        if epi_prior is not None:
            for tier_item in epi_prior.tiers:
                epi_tier_map[tier_item.label] = tier_item.tier
                epi_weight_map[tier_item.label] = tier_item.prior_weight
        _rule_out_labels_set: set[str] = set(rule_out_labels or set())
        rule_out_routed: list[DifferentialCandidate] = []

        rerank_gain = retrieval_stats.specificity_gain
        updated_candidates: list[DifferentialCandidate] = []
        updated_nodes: list[HypothesisNode] = []
        breakdowns: dict[str, CandidateScoreBreakdown] = {}
        must_not_miss_labels = {
            item.label
            for item in frontier.hypotheses
            if item.must_not_miss
        }
        must_not_miss_labels |= set(frontier.must_not_miss)
        prior_rank_by_label = {
            node.label: int(node.rank or 999)
            for node in frontier.hypotheses
            if str(node.label or "").strip()
        }

        registry = None
        if self.policy.trigger_compatibility_rerank_enabled:
            from src.cdss.knowledge.registry import load_syndrome_registry
            registry = load_syndrome_registry()

        # Severity proxy for risk-adjusted epi tier weighting.
        # Uplifts rare/exceptional-tier candidates when patient looks sick so rare-but-lethal dx
        # surface instead of being buried by base-rate priors. Bounded to [1.0, 1.8].
        _rf_count = len(findings.red_flags or [])
        _redflag_norm = min(1.0, _rf_count / 4.0)
        _vitals_instability = 0.0
        try:
            _dv = findings.derived_vitals or {}
            _shock = float(_dv.get("shock_index", 0.0) or 0.0)
            if _shock >= 1.0:
                _vitals_instability = 1.0
            elif _shock >= 0.7:
                _vitals_instability = 0.5
        except Exception:
            _vitals_instability = 0.0
        _severity_factor = 1.0 + 0.5 * _redflag_norm + 0.3 * _vitals_instability

        for candidate in differential.candidates[:5]:
            # Generic bucket penalty — generic labels should not outrank specific diagnoses
            generic_penalty = 0.0
            if _is_generic_label(candidate.label):
                generic_penalty = 0.18
                if str(candidate.label).endswith("_process"):
                    generic_penalty = 0.28

            # Trigger compatibility: registry-driven clinical constraint
            trigger_adjustment = 0.0
            if registry:
                trigger_adjustment = trigger_compatibility_adjustment(registry.by_id(candidate.label), findings)

            # Retrieval signal: small bonus when retrieval confirms the diagnosis
            retrieval_score = round(rerank_gain * 0.10, 2) if not _is_generic_label(candidate.label) else 0.0

            # Epidemiological prior: advisory signal from LLM-assessed base rates.
            # Rare/exceptional-tier candidates get a severity-scaled uplift so that
            # rare-but-lethal dx are not out-ranked by base rates when patient is sick.
            epi_signal = 0.0
            if epi_prior is not None:
                prior_weight = epi_weight_map.get(candidate.label, 0.6)
                base = prior_weight * 0.08
                _tier = epi_tier_map.get(candidate.label, "uncommon")
                if _tier in {"rare", "exceptional"} and _severity_factor > 1.0:
                    base *= _severity_factor
                epi_signal = round(base, 3)

            # Neural cognitive learning delta — replaces SQLite score_adjustments.
            # CognitiveLearningEngine combines:
            #   k-NN outcome-weighted signal from similar past cases (±0.15)
            #   + prototype centroid cosine similarity (±0.08)
            #   + online MLP prediction delta (±0.075)
            # Total clamped to [-0.20, +0.20]. Degrades to 0.0 if engine unavailable.
            neural_delta = 0.0
            try:
                from src.cdss.learning.cognitive_engine import get_cognitive_engine
                _engine = get_cognitive_engine()
                _case_emb = getattr(self, "_current_case_embedding", None)
                neural_delta = _engine.score_candidate(_case_emb, candidate.label)
            except Exception:
                neural_delta = 0.0

            finding_fit = score_candidate_finding_fit(candidate, findings)
            finding_fit_bonus = round(finding_fit.evidence_fit_score * 0.1, 3)
            orphan_penalty = round(finding_fit.orphan_penalty * 0.22, 3)

            # LLM-first scoring: LLM clinical reasoning is the primary source of truth
            raw_score = (
                candidate.score * 0.80
                + epi_signal
                + retrieval_score
                + trigger_adjustment
                + neural_delta
                + finding_fit_bonus
                - orphan_penalty
                - generic_penalty
            )
            final_score = round(max(0.0, min(0.99, raw_score)), 2)

            # Verified evidence gates: minor refinement from confirmed evidence
            if self.policy.evidence_confidence_requires_verified_edges:
                verified_supports, verified_refutes = _verified_edge_counts(evidence, candidate.label)
                if verified_refutes > verified_supports and verified_refutes > 0:
                    final_score = round(max(0.0, final_score - min(0.12, verified_refutes * 0.04)), 2)
                elif verified_supports > 0:
                    final_score = round(min(0.99, final_score + min(0.06, verified_supports * 0.02)), 2)
            updated_candidate = candidate.model_copy(
                update={
                    "score": final_score,
                    "specificity_tier": candidate.specificity_tier or ("family" if _is_generic_label(candidate.label) else "disease"),
                }
            )
            # Route to rule_out: substance/withdrawal diagnoses without documented context
            is_mnm = candidate.label in must_not_miss_labels
            epi_tier_for_candidate = epi_tier_map.get(candidate.label)
            candidate_breakdown = CandidateScoreBreakdown(
                state_fit=round(finding_fit.evidence_fit_score, 2),
                finding_fit=round(finding_fit.evidence_fit_score, 2),
                contradiction_penalty=round(finding_fit.orphan_penalty, 2),
                explanation_debt=round(finding_fit.orphan_penalty, 2),
            )
            if _should_route_to_rule_out(
                candidate,
                final_score,
                candidate_breakdown,
                is_must_not_miss=is_mnm,
                epi_prior_tier=epi_tier_for_candidate,
                findings=findings,
            ):
                rule_out_routed.append(updated_candidate)
            else:
                updated_candidates.append(updated_candidate)
            breakdowns[candidate.label] = CandidateScoreBreakdown(
                proposer_score=round(candidate.score, 2),
                llm_score=round(candidate.score, 2),
                retrieval_score=round(retrieval_score, 2),
                state_fit=round(finding_fit.evidence_fit_score, 2),
                finding_fit=round(finding_fit.evidence_fit_score, 2),
                contradiction_penalty=round(finding_fit.orphan_penalty, 2),
                explanation_debt=round(finding_fit.orphan_penalty, 2),
                final_score=final_score,
                learned_adjustment=round(neural_delta, 4),
                generic_penalty=round(generic_penalty, 2),
                safety_penalty=round(orphan_penalty, 3),
            )

        # Collapse near-duplicate label variants (e.g. dengue_hemorrhagic_fever vs
        # dengue_viral_hemorrhagic_fever) before final ordering to avoid wasting slots.
        updated_candidates = _deduplicate_sibling_candidates(
            sorted(updated_candidates, key=lambda c: c.score, reverse=True),
            must_not_miss_labels=must_not_miss_labels,
        )
        updated_candidates = _stabilize_candidate_order(
            updated_candidates,
            prior_rank_by_label=prior_rank_by_label,
            must_not_miss_labels=must_not_miss_labels,
            epi_prior_tier_map=epi_tier_map if epi_tier_map else None,
        )[:5]
        candidate_scores = {item.label: item.score for item in updated_candidates}
        for node in frontier.hypotheses:
            if node.label not in candidate_scores:
                continue
            updated_nodes.append(
                node.model_copy(
                    update={
                        "score": candidate_scores[node.label],
                    }
                )
            )
        updated_nodes = _order_frontier_nodes(updated_nodes, limit=5)

        uncertainty = _updated_uncertainty(updated_candidates)
        preferred_anchor_nodes = [node for node in updated_nodes if not _helper_like_source(node.source)]
        anchor_hypothesis = preferred_anchor_nodes[0].label if preferred_anchor_nodes else ""
        danger_source_nodes = updated_nodes
        effective_support_influence = 0.0
        updated_frontier = frontier.model_copy(
            update={
                "hypotheses": updated_nodes,
                "anchor_hypothesis": anchor_hypothesis,
                "frontier_entropy": uncertainty,
                "must_not_miss": [node.label for node in updated_nodes if node.must_not_miss][:4],
                "dangerous_if_treated_as": [node.dangerous_if_treated_as for node in danger_source_nodes if node.dangerous_if_treated_as][:4],
            }
        )
        updated_differential = differential.model_copy(
            update={
                "candidates": updated_candidates,
                "rule_out_candidates": rule_out_routed,
                "uncertainty": uncertainty,
                "strategy": "late_fusion_llm_ml_retrieval_arbitration",
            }
        )
        updated_support = model_support.model_copy(
            update={
                "score_breakdowns": breakdowns,
                "effective_support_influence": effective_support_influence,
            }
        )
        return StatePatch(
            hypothesis_frontier=updated_frontier,
            differential=updated_differential,
            model_support=updated_support,
            trace=[
                DecisionTrace(
                    stage="arbitration",
                    message="Late-fusion arbitration combined LLM, support-ML, retrieval, and safety signals.",
                    payload={
                        "anchor_hypothesis": updated_frontier.anchor_hypothesis,
                        "model_disagreement": model_support.model_disagreement,
                        "effective_support_influence": effective_support_influence,
                        "support_proposer_correlation": model_support.support_proposer_correlation,
                        "retrieval_specificity_gain": retrieval_stats.specificity_gain,
                        "state_frames": explanation_graph.state_frames[:4],
                        "rule_out_candidates": [c.label for c in rule_out_routed],
                        "normalization_absorbed": normalization_absorbed,
                        "epi_prior_applied": epi_prior is not None,
                        "specificity_judgments_applied": specificity_judgments is not None,
                    },
                )
            ],
        )
