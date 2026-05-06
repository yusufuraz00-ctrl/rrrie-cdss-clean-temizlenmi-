"""Typed model boundary used by the legacy swarm runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.cdss.agents.extractor import ExtractorAgent
from src.cdss.agents.semantic_parser import SemanticParserAgent
from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    DiseaseHypothesis,
    EvidenceNeed,
    EvidenceBundle,
    FactGraph,
    HypothesisFrontier,
    HypothesisNode,
    InterventionSet,
    LoopDirective,
    MechanismFrame,
    PatientInput,
    RiskProfile,
    StructuredFindings,
    VerificationIssue,
)
from src.cdss.knowledge.ontology import is_plausible_diagnosis_label, normalize_candidate_label
from src.cdss.runtime.llm_bridge import LocalLlmClinicalExtractor
from src.cdss.runtime.policy import CdssRuntimePolicy, load_runtime_policy


@dataclass
class GatewayIntakeResult:
    findings: StructuredFindings = field(default_factory=StructuredFindings)
    fact_graph: FactGraph = field(default_factory=FactGraph)
    interventions: InterventionSet = field(default_factory=InterventionSet)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayDifferentialResult:
    differential: DifferentialSet = field(default_factory=DifferentialSet)
    frontier: HypothesisFrontier = field(default_factory=HypothesisFrontier)
    disease_hypotheses: list[DiseaseHypothesis] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)
    # W6.2 J.2 — per-candidate ensemble stats (mean, var, alpha, beta, samples)
    # populated only when caller passed `temperature_ensemble=True`.
    ensemble_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class GatewayResearchPlan:
    active_hypotheses: list[str] = field(default_factory=list)
    discriminators: list[dict[str, str]] = field(default_factory=list)
    queries: dict[int, str] = field(default_factory=dict)
    expected_impact: dict[int, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayMechanismResult:
    mechanism_frames: list[MechanismFrame] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayChallengeResult:
    issues: list[VerificationIssue] = field(default_factory=list)
    alt_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    anchor: str = ""
    anchor_delta: float = 0.0
    loop_directive: LoopDirective = field(default_factory=LoopDirective)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayVerificationResult:
    issues: list[VerificationIssue] = field(default_factory=list)
    reliability_delta: float = 0.0
    decision_hint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


def _context_string(findings: StructuredFindings) -> str:
    parts = [
        findings.summary,
        *findings.positive_findings[:10],
        *findings.red_flags[:8],
        *findings.exposures[:6],
        *findings.timeline[:6],
        *findings.suspected_conditions[:6],
        *findings.raw_segments[:8],
    ]
    return " | ".join(str(item or "").strip() for item in parts if str(item or "").strip())


def _payload_error(payload: dict[str, Any]) -> dict[str, Any]:
    meta = dict((payload.get("_meta", {}) or {}))
    return dict(meta.get("error", {}) or {})


class SwarmModelGateway:
    """Shared typed model adapter for the state-machine runtime."""

    def __init__(self, execution_mode: str | None = None, policy: CdssRuntimePolicy | None = None) -> None:
        self.policy = policy or load_runtime_policy()
        self.bridge = LocalLlmClinicalExtractor(self.policy)
        normalized_mode = str(execution_mode or "").strip().lower()
        if normalized_mode in {"local_qwen", "cloud_gemini"}:
            self.bridge.lock_engine_mode(normalized_mode)
        extractor = ExtractorAgent(llm_extractor=self.bridge)
        self.semantic_parser = SemanticParserAgent(extractor=extractor)

    @property
    def active_mode(self) -> str:
        return self.bridge.active_mode

    @property
    def active_engine_model(self) -> str:
        return self.bridge.active_engine_model

    async def extract_case(self, patient_input: PatientInput) -> GatewayIntakeResult:
        patch = await self.semantic_parser.run(patient_input)
        trace_payload = dict((patch.trace[0].payload if patch.trace else {}) or {})
        return GatewayIntakeResult(
            findings=patch.findings or StructuredFindings(),
            fact_graph=patch.fact_graph or FactGraph(),
            interventions=patch.interventions or InterventionSet(),
            metrics=dict(trace_payload.get("metrics", {}) or {}),
            error=dict(trace_payload.get("error", {}) or {}),
        )

    async def induce_differential(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        *,
        evidence: EvidenceBundle | None = None,
        focus_profile: str = "general",
        deep_thinking: bool = False,
        hierarchy_level: int | None = None,
        candidate_set: list[str] | None = None,
        shared_belief: str = "",
        temperature_ensemble: bool = False,
    ) -> GatewayDifferentialResult:
        # W6.2 J.2 — caller opts into temperature-ensemble likelihood by
        # passing `temperature_ensemble=True`. Routes to the ensemble bridge
        # method which fires N parallel calls and merges per-candidate stats.
        if temperature_ensemble:
            payload = await self.bridge.induce_diagnoses_ensemble(
                findings,
                risk_profile,
                fact_graph,
                evidence=evidence,
                focus_profile=focus_profile,
                deep_thinking=deep_thinking,
                hierarchy_level=hierarchy_level,
                candidate_set=candidate_set,
                shared_belief=shared_belief,
            )
        else:
            payload = await self.bridge.induce_diagnoses(
                findings,
                risk_profile,
                fact_graph,
                evidence=evidence,
                focus_profile=focus_profile,
                deep_thinking=deep_thinking,
                hierarchy_level=hierarchy_level,
                candidate_set=candidate_set,
                shared_belief=shared_belief,
            )
        raw_candidates = list(payload.get("candidates", []) or [])
        raw_labels = [str((item or {}).get("label", "") or "").strip() for item in raw_candidates if str((item or {}).get("label", "") or "").strip()]
        validated_map: dict[str, str] = {}
        if raw_labels and bool(getattr(self.policy, "llm_label_validation_enabled", False)):
            validation = await self.bridge.validate_diagnostic_labels(raw_labels, context=_context_string(findings))
            for item in list(validation.get("accepted", []) or []):
                raw_label = str(item.get("raw_label", "") or "").strip().lower()
                canonical = normalize_candidate_label(str(item.get("canonical_label", "") or "").strip())
                if raw_label and canonical:
                    validated_map[raw_label] = canonical
        elif raw_labels:
            for raw_label in raw_labels:
                canonical = normalize_candidate_label(raw_label)
                if canonical and is_plausible_diagnosis_label(canonical):
                    validated_map[raw_label.lower()] = canonical

        candidates: list[DifferentialCandidate] = []
        hypotheses: list[HypothesisNode] = []
        for index, item in enumerate(raw_candidates[:5], start=1):
            raw_label = str((item or {}).get("label", "") or "").strip()
            if not raw_label:
                continue
            canonical = validated_map.get(raw_label.lower()) or normalize_candidate_label(raw_label) or raw_label.strip().lower().replace(" ", "_")
            if not canonical or not is_plausible_diagnosis_label(canonical):
                continue
            try:
                score = max(0.0, min(0.99, float((item or {}).get("score", 0.0) or 0.0)))
            except (TypeError, ValueError):
                score = 0.0
            rationale_text = str((item or {}).get("rationale", "") or "").strip()
            rationale = [rationale_text] if rationale_text else [f"Gateway induced candidate rank {index}."]
            candidates.append(
                DifferentialCandidate(
                    label=canonical,
                    score=round(score, 2),
                    rationale=rationale[:3],
                    status="candidate",
                )
            )
            hypotheses.append(
                HypothesisNode(
                    node_id=f"gateway_hyp_{index}",
                    label=canonical,
                    score=round(score, 2),
                    rank=index,
                    rationale=rationale[:3],
                    source="gateway_induction",
                    canonical_label=canonical,
                    mapped=(canonical != raw_label),
                )
            )

        frontier = HypothesisFrontier(
            hypotheses=hypotheses,
            frontier_entropy=round(max(0.0, 1.0 - (candidates[0].score if candidates else 0.0)), 2),
            strategy="gateway_induction",
            anchor_hypothesis=hypotheses[0].label if hypotheses else "",
        )
        disease_hypotheses = [
            DiseaseHypothesis(
                label=item.label,
                score=item.score,
                rationale=item.rationale[:3],
                status="candidate",
            )
            for item in candidates
        ]
        # Re-key ensemble stats by canonical labels so callers can look up
        # candidate.label → (mean, var, alpha, beta).
        raw_ensemble = dict(payload.get("_ensemble", {}) or {})
        canon_ensemble: dict[str, dict[str, Any]] = {}
        for raw_label, stats in raw_ensemble.items():
            cn = validated_map.get(str(raw_label).lower()) or normalize_candidate_label(str(raw_label)) or str(raw_label)
            if cn:
                canon_ensemble[cn] = stats
        return GatewayDifferentialResult(
            differential=DifferentialSet(
                candidates=candidates,
                uncertainty=frontier.frontier_entropy if candidates else 1.0,
                strategy="gateway_induction",
            ),
            frontier=frontier,
            disease_hypotheses=disease_hypotheses,
            metrics=dict((payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
            error=_payload_error(payload),
            ensemble_stats=canon_ensemble,
        )

    async def plan_research(
        self,
        findings: StructuredFindings,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        needs: list[EvidenceNeed],
        *,
        iteration: int = 0,
    ) -> GatewayResearchPlan:
        payload = await self.bridge.generate_retrieval_queries(
            findings,
            fact_graph,
            frontier,
            [item.model_dump(mode="json") for item in needs],
            iteration=iteration,
        )
        return GatewayResearchPlan(
            active_hypotheses=list(payload.get("active_hypotheses", []) or []),
            discriminators=list(payload.get("discriminators", []) or []),
            queries={int(key): str(value) for key, value in dict(payload.get("queries", {}) or {}).items()},
            expected_impact={int(key): str(value) for key, value in dict(payload.get("expected_impact", {}) or {}).items()},
            metrics=dict((payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
            error=_payload_error(payload),
        )

    async def generate_mechanism_frames(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        interventions: InterventionSet,
    ) -> GatewayMechanismResult:
        payload = await self.bridge.generate_mechanism_frames(findings, risk_profile, fact_graph, interventions)
        return GatewayMechanismResult(
            mechanism_frames=[
                MechanismFrame.model_validate(item)
                for item in list(payload.get("mechanism_frames", []) or [])
                if isinstance(item, dict)
            ],
            metrics=dict((payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
            error=_payload_error(payload),
        )

    async def challenge(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
        interventions: InterventionSet,
        deep_thinking: bool = False,
        loop_iteration: int = 0,
        tier: str = "full",
    ) -> GatewayChallengeResult:
        payload = await self.bridge.challenge(
            findings, risk_profile, fact_graph, frontier, differential, evidence, interventions, deep_thinking, loop_iteration, tier
        )
        return GatewayChallengeResult(
            issues=[
                VerificationIssue.model_validate(item)
                for item in list(payload.get("issues", []) or [])
                if isinstance(item, dict)
            ],
            alt_hypotheses=list(payload.get("alt_hypotheses", []) or []),
            anchor=str(payload.get("anchor", "") or ""),
            anchor_delta=float(payload.get("anchor_delta", 0.0) or 0.0),
            loop_directive=LoopDirective(
                action=str(payload.get("loop_action", "") or "").strip() or "none",
                reason=str(payload.get("loop_reason", "") or "").strip(),
            ),
            metrics=dict((payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
            error=_payload_error(payload),
        )

    async def verify(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        interventions: InterventionSet,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
    ) -> GatewayVerificationResult:
        payload = await self.bridge.verify(findings, risk_profile, fact_graph, frontier, interventions, differential, evidence)
        return GatewayVerificationResult(
            issues=[
                VerificationIssue.model_validate(item)
                for item in list(payload.get("issues", []) or [])
                if isinstance(item, dict)
            ],
            reliability_delta=float(payload.get("reliability_delta", 0.0) or 0.0),
            decision_hint=str(payload.get("decision_hint", "") or "").strip(),
            metrics=dict((payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
            error=_payload_error(payload),
        )

    async def calibrate_differential(
        self,
        patient_summary: str,
        top_candidates: list[str],
        epidemiology_context: str,
    ) -> list[str]:
        """Re-rank candidates via epidemiological calibration. Returns re-ordered label list."""
        payload = await self.bridge.calibrate_differential(patient_summary, top_candidates, epidemiology_context)
        ranked = list(payload.get("ranked", []) or [])
        if not ranked:
            return top_candidates  # no change if LLM returned nothing useful
        reordered = [item["label"] for item in ranked if item.get("label")]
        # Only apply if calibrator returned labels that exist in the original list
        valid = [label for label in reordered if label in top_candidates]
        if len(valid) < 2:
            return top_candidates
        # Append any candidates not mentioned by calibrator at the end
        remaining = [c for c in top_candidates if c not in valid]
        return valid + remaining
