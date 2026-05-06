import asyncio
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from dotenv import load_dotenv

from src.cdss.contracts.models import (
    CdssPhase,
    DifferentialCandidate,
    DifferentialSet,
    DiseaseHypothesis,
    EvidenceBundle,
    EvidenceItem,
    EvidenceNeed,
    EvidenceNode,
    FactGraph as TypedFactGraph,
    HypothesisFrontier,
    HypothesisNode,
    InterventionSafetyAssessment,
    InterventionSet,
    MechanismFrame,
    ModelSupportSignals,
    PatientInput,
    RetrievalIntent,
    RetrievalRankingStats,
    RiskProfile,
    StructuredFindings,
    UrgencyTier,
    VerificationIssue,
    WorkflowState,
)
from src.cdss.core import CaseSignal, SignalJournal, SignalKind
from src.cdss.knowledge.ontology import normalize_candidate_label
from src.cdss.knowledge.registry import load_syndrome_registry
from src.cdss.retrieval.planner import build_evidence_bundle, summarize_query_quality
# Coarse-to-fine Bayesian belief (W1 Module A/B). Lazy-imported within
# `_fold_swarm_into_belief` to keep cold-start fast when flags are off.
from src.cdss.runtime.model_gateway import SwarmModelGateway
from src.cdss.learning.memory import EvidenceMemoryStore

load_dotenv()
logger = logging.getLogger("rrrie-cdss")

QWEN_ENDPOINT = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080") + "/v1/chat/completions"
MEMORY_GRAPH_PATH = "data/learning/swarm_knowledge_graph.json"

# LLM mode can be injected per request; env fallback is kept for non-GUI callers.
def _get_llm_mode() -> str:
    return os.environ.get("LLM_MODE", "LOCAL").upper()


def _execution_mode_to_llm_mode(execution_mode: str | None) -> str:
    normalized = str(execution_mode or "").strip().lower()
    if normalized == "cloud_gemini":
        return "GEMINI"
    if normalized == "local_qwen":
        return "LOCAL"
    return ""

# Support both GEMINI_API_KEY and GOOGLE_API_KEY (whichever .env defines)
def _get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

def _get_gemini_endpoint() -> str:
    key = _get_gemini_api_key()
    return f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key={key}"


def _normalize_web_search_results(raw: Any) -> list[dict[str, Any]]:
    """Accept both direct list results and the standard web_search dict shape."""
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        candidates = raw.get("results", [])
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []
    return [item for item in candidates if isinstance(item, dict)]


def _web_search_result_from_cache(raw: Any) -> bool:
    return isinstance(raw, dict) and bool(raw.get("_from_cache"))


def _sm_entropy(values: list[float]) -> float:
    weights = [max(float(value or 0.0), 0.0) for value in values if float(value or 0.0) > 0.0]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return round(-sum((weight / total) * math.log2(weight / total) for weight in weights), 4)


def _label_similarity(a: str, b: str) -> float:
    """Q2: fuzzy similarity 0..1 between two normalized labels.

    Catches sibling labels that mean the same thing but differ by typo / suffix
    (acute_appendicitis vs acute_appendedicitis; acetaminophen_toxicity vs
    acetaminophen_overdose). Cheap: equality + prefix overlap fast paths,
    SequenceMatcher only for the remaining cases.
    """
    if not a or not b:
        return 0.0
    la = str(a).lower()
    lb = str(b).lower()
    if la == lb:
        return 1.0
    if la.startswith(lb) or lb.startswith(la):
        full = max(len(la), len(lb))
        return min(len(la), len(lb)) / full if full else 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, la, lb).ratio()


def _registry_canonical_id(label: str) -> str:
    """Return canonical registry profile id for a label, or "" if unknown.

    Ontology-aware merge precedence: if two labels resolve to the same
    registry id, they are the same diagnosis and may be merged regardless
    of string similarity. If they resolve to different ids (e.g.
    epidural_hematoma vs subdural_hematoma — both registered), they are
    distinct must-not-miss conditions and must NOT be merged even if
    string similarity is high. Falls back to "" silently if the registry
    is unavailable, which lets the pure-string fallback take over.
    """
    norm = (normalize_candidate_label(label) or str(label or "").lower()).strip()
    if not norm:
        return ""
    try:
        from src.cdss.knowledge.registry import load_syndrome_registry
        registry = load_syndrome_registry()
    except Exception:  # noqa: BLE001 — registry must never crash dedup
        return ""
    if registry is None:
        return ""
    profile = registry.by_id(norm)
    if profile is not None:
        return profile.id
    # Fallback: scan profiles for a label match (registry uses snake-case ids
    # but legacy labels may differ).
    try:
        for prof in registry.profiles:
            if (prof.label or "").lower().replace(" ", "_") == norm or prof.id == norm:
                return prof.id
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _dedup_differential_candidates(
    candidates: list,
    *,
    threshold: float = 0.95,
) -> list:
    """Collapse near-duplicate diagnosis candidates into the higher-scored
    representative.

    Two-stage matching, ontology-aware:

    1. **Canonical-id match**: if both labels resolve to the same
       registry profile id, merge unconditionally. This handles synonym
       and typo variants when the registry knows them.
    2. **String-similarity fallback**: only when canonical lookup is
       inconclusive (one or both labels unknown to the registry), fall
       back to string similarity at a deliberately-conservative
       threshold (default ``0.95``, was ``0.85`` and merged distinct
       sibling diagnoses like ``epidural_hematoma`` vs
       ``subdural_hematoma`` whose ratio is 0.867).

    A **sibling guard** vetoes string-similarity merges when both labels
    resolve to *different* known registry ids: distinct must-not-miss
    siblings must never collapse silently regardless of how similar
    their labels look. This is the structural fix for the Batch-2 case
    where a hematoma sibling's score was absorbed into a different
    sibling and lost from the differential.

    Survivor keeps its score; loser's first rationale line is appended
    for audit trail. Pure function; safe to call from multiple stages.
    """
    if not candidates:
        return list(candidates or [])
    sorted_cands = sorted(
        candidates,
        key=lambda c: float(getattr(c, "score", 0.0) or 0.0),
        reverse=True,
    )
    survivors: list = []
    cand_canonical: list[str] = []  # canonical id for each survivor (parallel index)
    for cand in sorted_cands:
        raw_label = str(getattr(cand, "label", "") or "")
        cand_norm = normalize_candidate_label(raw_label) or raw_label.lower()
        cand_id = _registry_canonical_id(raw_label)
        merged = False
        for idx, surv in enumerate(survivors):
            surv_label = str(getattr(surv, "label", "") or "")
            surv_norm = normalize_candidate_label(surv_label) or surv_label.lower()
            surv_id = cand_canonical[idx]

            # Stage 1: canonical-id match. Same registry id → same dx.
            same_canonical = bool(cand_id and surv_id and cand_id == surv_id)
            # Sibling guard: distinct known ids → never merge.
            distinct_canonical = bool(cand_id and surv_id and cand_id != surv_id)

            should_merge = False
            if same_canonical:
                should_merge = True
            elif distinct_canonical:
                should_merge = False  # explicit veto for sibling siblings
            else:
                # Stage 2: conservative string-similarity fallback.
                if _label_similarity(cand_norm, surv_norm) >= threshold:
                    should_merge = True

            if should_merge:
                cand_rationale = list(getattr(cand, "rationale", []) or [])
                surv_rationale = list(getattr(surv, "rationale", []) or [])
                merge_note = f"(merged near-duplicate '{raw_label}'"
                if cand_rationale:
                    merge_note += f": {cand_rationale[0][:140]}"
                merge_note += ")"
                survivors[idx] = surv.model_copy(
                    update={"rationale": (surv_rationale + [merge_note])[:5]}
                )
                merged = True
                break
        if not merged:
            survivors.append(cand)
            cand_canonical.append(cand_id)
    return survivors


def _compute_case_complexity(entropy: float, risk_profile: RiskProfile) -> str:
    """Derives complexity tier from entropy + urgency. Drives adaptive routing decisions.

    Bug-guard: NaN entropy (can occur when posterior collapses to a
    single label and log(0) sneaks through) is treated as 0.0 so the
    boundary check is deterministic. Thresholds are read from config
    with safe in-code defaults.
    """
    import math as _math
    from src.cdss.core import thresholds as _ct

    if entropy is None or (isinstance(entropy, float) and _math.isnan(entropy)):
        entropy = 0.0
    urgency = getattr(risk_profile, "urgency", None)
    is_emergency = urgency == UrgencyTier.EMERGENCY
    crit_thresh = _ct.get_float("complexity.entropy_critical_gte", 3.0)
    elev_thresh = _ct.get_float("complexity.entropy_elevated_gte", 2.0)
    if entropy >= crit_thresh or is_emergency:
        return "critical"
    if entropy >= elev_thresh:
        return "elevated"
    return "routine"


class CdssStateMachineAgent:
    """
    V3.0 Auto-Evolving Clinical Swarm (Consilium Engine)
    Moves away from linear pipelines to a Mixture of Agents (MoA) approach,
    incorporating Test-Time Compute (Simulations), and Graph-based Episodic Memory.
    """
    def __init__(self, patient_id: str, raw_context: str, trace_callback=None, execution_mode: str | None = None, deep_thinking: bool = False):
        self.state = WorkflowState(
            patient_id=patient_id,
            raw_context=raw_context,
            phase=CdssPhase.INTAKE,
            fact_graph={"patient_id": patient_id, "nodes": [], "edges": []}
        )
        self.knowledge_graph = self._load_knowledge_graph()
        self.trace_callback = trace_callback
        self.deep_thinking = deep_thinking
        self._case_complexity: str = "routine"
        self.requested_execution_mode = str(execution_mode or "").strip().lower()
        self.active_llm_mode = _execution_mode_to_llm_mode(self.requested_execution_mode) or _get_llm_mode()
        self.active_engine_model = "qwen"
        self._http_client: httpx.AsyncClient | None = None
        self._stage_profiles: dict[str, dict[str, float | int | bool]] = {}
        self._current_profile_stage = "system"
        self.signal_journal = SignalJournal(case_id=patient_id)
        self.typed_case_bundle: dict[str, Any] = {}
        self._deep_reasoning_loop_count: int = 0
        self._gateway = SwarmModelGateway(execution_mode=self.requested_execution_mode)
        print(f"[{time.strftime('%H:%M:%S')}] [Engine Registry] Mode: {self.active_llm_mode} — Engine Instance: {self._gateway.active_engine_model}")

        self.runtime_policy = self._gateway.policy
        self._evidence_memory = EvidenceMemoryStore(self.runtime_policy.learning_dir)
        from src.cdss.runtime.arbitration import ArbitrationLayer
        self._arbitration = ArbitrationLayer()

    async def _emit_trace(self, stage: str, message: str, payload: dict = None):
        """Asynchronously dispatches real-time UI updates (traces)"""
        if self.trace_callback:
            from datetime import datetime
            from src.cdss.contracts.models import DecisionTrace
            trace_obj = DecisionTrace(
                timestamp=datetime.now().isoformat(),
                stage=stage,
                message=message,
                payload=payload or {}
            )
            await self.trace_callback(trace_obj)

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def _close_http_client(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _set_profile_stage(self, stage: str) -> None:
        self._current_profile_stage = stage

    def _profile_bucket(self, stage: str | None = None) -> dict[str, float | int | bool]:
        bucket = self._stage_profiles.setdefault(
            stage or self._current_profile_stage,
            {
                "wall_time_s": 0.0,
                "model_wait_s": 0.0,
                "prompt_chars": 0,
                "output_chars": 0,
                "llm_calls": 0,
                "cache_hits": 0,
                "web_hits": 0,
            },
        )
        return bucket

    def _record_stage_profile(
        self,
        *,
        stage: str | None = None,
        wall_time_s: float = 0.0,
        model_wait_s: float = 0.0,
        prompt_chars: int = 0,
        output_chars: int = 0,
        llm_calls: int = 0,
        cache_hits: int = 0,
        web_hits: int = 0,
    ) -> None:
        bucket = self._profile_bucket(stage)
        bucket["wall_time_s"] = round(float(bucket["wall_time_s"]) + float(wall_time_s), 2)
        bucket["model_wait_s"] = round(float(bucket["model_wait_s"]) + float(model_wait_s), 2)
        bucket["prompt_chars"] = int(bucket["prompt_chars"]) + int(prompt_chars)
        bucket["output_chars"] = int(bucket["output_chars"]) + int(output_chars)
        bucket["llm_calls"] = int(bucket["llm_calls"]) + int(llm_calls)
        bucket["cache_hits"] = int(bucket["cache_hits"]) + int(cache_hits)
        bucket["web_hits"] = int(bucket["web_hits"]) + int(web_hits)

    def _stage_metrics(self, stage: str, *, wall_time_s: float | None = None) -> dict[str, float | int | bool]:
        bucket = self._profile_bucket(stage)
        effective_wall = float(bucket["wall_time_s"])
        if wall_time_s is not None:
            effective_wall = max(effective_wall, float(wall_time_s))
            bucket["wall_time_s"] = round(effective_wall, 2)
        return {
            "time_s": round(effective_wall, 2),
            "wall_time_s": round(effective_wall, 2),
            "model_wait_s": round(float(bucket["model_wait_s"]), 2),
            "prompt_chars": int(bucket["prompt_chars"]),
            "output_chars": int(bucket["output_chars"]),
            "llm_calls": int(bucket["llm_calls"]),
            "cache_hit": bool(int(bucket["cache_hits"]) > 0),
            "cache_hits": int(bucket["cache_hits"]),
            "web_hit": bool(int(bucket["web_hits"]) > 0),
            "web_hits": int(bucket["web_hits"]),
        }

    def _record_gateway_metrics(self, stage: str, metrics: dict[str, Any] | None) -> None:
        payload = dict(metrics or {})
        self._record_stage_profile(
            stage=stage,
            model_wait_s=float(payload.get("time_s", 0.0) or 0.0),
            prompt_chars=int(payload.get("prompt_tokens", 0) or 0),
            output_chars=int(payload.get("completion_tokens", 0) or 0),
            llm_calls=1 if payload else 0,
        )

    def _build_patient_input_snapshot(self) -> PatientInput:
        return PatientInput(
            case_id=str(getattr(self.state, "patient_id", "") or "interactive-case"),
            patient_text=str(self.state.raw_context or ""),
            execution_mode=self.requested_execution_mode or PatientInput.model_fields["execution_mode"].default,
            source="state_machine",
        )

    def _typed_findings(self) -> StructuredFindings:
        findings = self.typed_case_bundle.get("findings")
        return findings if isinstance(findings, StructuredFindings) else StructuredFindings()

    def _typed_fact_graph(self) -> TypedFactGraph:
        fact_graph = self.typed_case_bundle.get("fact_graph")
        return fact_graph if isinstance(fact_graph, TypedFactGraph) else TypedFactGraph()

    def _typed_interventions(self) -> InterventionSet:
        interventions = self.typed_case_bundle.get("interventions")
        return interventions if isinstance(interventions, InterventionSet) else InterventionSet()

    def _typed_evidence(self) -> EvidenceBundle:
        evidence = self.typed_case_bundle.get("evidence")
        return evidence if isinstance(evidence, EvidenceBundle) else EvidenceBundle()

    def _journal_labels(self, kinds: list[SignalKind], *, limit: int = 24) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for kind in kinds:
            for label in self.signal_journal.latest_labels(kind, limit=limit):
                key = label.lower()
                if not label or key in seen:
                    continue
                seen.add(key)
                labels.append(label)
                if len(labels) >= limit:
                    return labels
        return labels

    def _record_findings_signals(self, findings: StructuredFindings, interventions: InterventionSet) -> None:
        signals: list[CaseSignal] = []
        for item in findings.positive_findings[:32]:
            signals.append(CaseSignal(kind=SignalKind.FINDING, label=str(item), canonical_label=normalize_candidate_label(str(item)) or "", source_stage="intake", raw_span=str(item), confidence=0.9))
        for item in findings.negative_findings[:24]:
            text = str(item or "").strip()
            if not text:
                continue
            signals.append(
                CaseSignal(
                    kind=SignalKind.FINDING,
                    label=text,
                    canonical_label=normalize_candidate_label(text) or "",
                    source_stage="intake",
                    raw_span=text,
                    confidence=0.72,
                )
            )
        for item in findings.timeline[:24]:
            text = str(item or "").strip()
            if not text:
                continue
            signals.append(
                CaseSignal(
                    kind=SignalKind.FINDING,
                    label=text,
                    canonical_label=normalize_candidate_label(text) or "",
                    source_stage="intake",
                    raw_span=text,
                    confidence=0.74,
                )
            )
        for lane, values in list((findings.context_lanes or {}).items())[:10]:
            for item in list(values or [])[:8]:
                text = str(item or "").strip()
                if not text:
                    continue
                signals.append(
                    CaseSignal(
                        kind=SignalKind.FINDING,
                        label=text,
                        canonical_label=normalize_candidate_label(text) or "",
                        source_stage=f"intake:{lane}",
                        raw_span=text,
                        confidence=0.68,
                    )
                )
        for item in findings.red_flags[:16]:
            signals.append(CaseSignal(kind=SignalKind.RED_FLAG, label=str(item), canonical_label=normalize_candidate_label(str(item)) or "", source_stage="intake", raw_span=str(item), confidence=0.95))
        for item in findings.exposures[:16]:
            signals.append(CaseSignal(kind=SignalKind.EXPOSURE, label=str(item), canonical_label=normalize_candidate_label(str(item)) or "", source_stage="intake", raw_span=str(item), confidence=0.88))
        for key, value in list((findings.derived_vitals or {}).items())[:12]:
            vital_value = str(value or "").strip()
            if not vital_value:
                continue
            signals.append(
                CaseSignal(
                    kind=SignalKind.VITAL,
                    label=str(key),
                    canonical_label=normalize_candidate_label(str(key)) or "",
                    value=vital_value,
                    source_stage="intake",
                    raw_span=f"{key}={vital_value}",
                    confidence=1.0,
                )
            )
        for item in interventions.items[:12]:
            label = str(item.label or "").strip()
            if not label:
                continue
            signals.append(CaseSignal(kind=SignalKind.INTERVENTION, label=label, canonical_label=normalize_candidate_label(label) or "", source_stage="intake", raw_span=label, confidence=float(item.confidence or 0.0)))
        if signals:
            self.signal_journal.extend(signals)

    def _provisional_frontier(self) -> HypothesisFrontier:
        frontier = self.typed_case_bundle.get("frontier")
        if isinstance(frontier, HypothesisFrontier) and frontier.hypotheses:
            return frontier
        from src.cdss.knowledge.registry import load_syndrome_registry

        registry = load_syndrome_registry()
        nodes: list[HypothesisNode] = []
        for index, match in enumerate(self._stable_prototype_matches()[:5], start=1):
            label = normalize_candidate_label(str(match.label or "").strip()) or str(match.label or "").strip()
            if not label:
                continue
            profile = registry.by_id(label)
            nodes.append(
                HypothesisNode(
                    node_id=f"proto_{index}",
                    label=label,
                    score=round(max(0.05, min(0.99, float(match.similarity or 0.0))), 2),
                    rank=index,
                    rationale=[f"Prototype match {float(match.similarity or 0.0):.2f}."],
                    must_not_miss=bool(getattr(profile, "must_not_miss", False)),
                    dangerous_if_missed=bool(getattr(profile, "dangerous_if_missed", False)),
                    dangerous_if_treated_as=str(getattr(profile, "dangerous_if_treated_as", "") or ""),
                    source="prototype_memory",
                    canonical_label=label,
                )
            )
        frontier = HypothesisFrontier(
            hypotheses=nodes,
            frontier_entropy=_sm_entropy([node.score for node in nodes]),
            strategy="prototype_memory",
            anchor_hypothesis=nodes[0].label if nodes else "",
            must_not_miss=[node.label for node in nodes if node.must_not_miss][:4],
            dangerous_if_treated_as=[node.dangerous_if_treated_as for node in nodes if node.dangerous_if_treated_as][:4],
        )
        if nodes:
            self.typed_case_bundle["frontier"] = frontier
        return frontier

    def _research_seed_bundle(self, frontier: HypothesisFrontier) -> EvidenceBundle:
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        interventions = self._typed_interventions()
        differential = self.typed_case_bundle.get("differential")
        if not isinstance(differential, DifferentialSet) or not differential.candidates:
            differential = DifferentialSet(
                candidates=[
                    DifferentialCandidate(
                        label=node.label,
                        score=round(float(node.score or 0.0), 2),
                        rationale=list(node.rationale[:3]),
                        status="candidate",
                    )
                    for node in frontier.hypotheses[:4]
                    if str(node.label or "").strip()
                ],
                uncertainty=frontier.frontier_entropy,
                strategy="research_seed",
            )
        return build_evidence_bundle(
            self._build_patient_input_snapshot(),
            findings,
            self._typed_risk_profile(),
            differential,
            hypothesis_frontier=frontier if frontier.hypotheses else None,
            interventions=interventions,
            fact_graph=fact_graph,
        )

    def _research_needs(self, frontier: HypothesisFrontier) -> list[EvidenceNeed]:
        return list(self._research_seed_bundle(frontier).evidence_needs[:8])

    def _sync_legacy_fact_graph_from_typed(self, findings: StructuredFindings, fact_graph: TypedFactGraph) -> None:
        nodes: list[EvidenceNode] = []
        seen_keys: set[tuple[str, str]] = set()

        def _append(node_type: str, label: str, *, confidence: float = 0.88, source: str = "typed_gateway", metadata: dict[str, str] | None = None) -> None:
            text = str(label or "").strip()
            key = (node_type, text.lower())
            if not text or key in seen_keys:
                return
            seen_keys.add(key)
            nodes.append(
                EvidenceNode(
                    id=f"{node_type}_{len(nodes) + 1}",
                    type=node_type,  # type: ignore[arg-type]
                    label=text,
                    confidence=round(float(confidence or 0.0), 2),
                    source=source,
                    metadata=dict(metadata or {}),
                )
            )

        for node in fact_graph.nodes:
            category = str(node.category or "").strip().lower()
            label = str(node.label or "").strip()
            if not label:
                continue
            _CATEGORY_SIGNAL_MAP: dict[str, str] = {
                "diagnosis": "diagnosis", "candidate": "diagnosis",
                "evidence": "evidence", "authority_claim": "evidence",
                "treatment": "treatment", "intervention": "treatment",
                "demographic": "demographic",
                "symptom": "symptom", "finding": "symptom",
                "pattern_marker": "symptom", "risk_marker": "symptom",
                "semantic_pattern": "symptom", "course_marker": "symptom",
                "laterality_marker": "symptom",
            }
            if category == "vital":
                display = f"{label}: {node.value}" if str(node.value or "").strip() else label
                _append("vital", display, confidence=float(node.confidence or 0.0), source="typed_gateway_fact_graph")
            else:
                signal_type = _CATEGORY_SIGNAL_MAP.get(category, "finding")
                _append(signal_type, label, confidence=float(node.confidence or 0.0), source="typed_gateway_fact_graph")

        for item in findings.positive_findings[:24]:
            _append("symptom", item, source="typed_gateway_findings")
        for item in findings.exposures[:12]:
            _append("symptom", item, source="typed_gateway_findings", metadata={"kind": "exposure"})
        for item in findings.timeline[:12]:
            _append("symptom", item, source="typed_gateway_findings", metadata={"kind": "timeline"})
        for item in findings.red_flags[:12]:
            _append("evidence", f"Risk: {item}", confidence=0.95, source="typed_gateway_findings")
        for item in findings.suspected_conditions[:8]:
            _append("diagnosis", item, confidence=0.55, source="typed_gateway_findings")
        for key, value in list((findings.derived_vitals or {}).items())[:12]:
            vital_value = str(value or "").strip()
            if not vital_value:
                continue
            _append("vital", f"{key}: {vital_value}", confidence=1.0, source="typed_gateway_findings")

        self.state.fact_graph.nodes = nodes
        self.state.fact_graph.edges = []

    def _typed_risk_profile(self) -> RiskProfile:
        findings = self._typed_findings()
        try:
            from src.cdss.clinical.safety import build_risk_profile

            risk = build_risk_profile(self._build_patient_input_snapshot(), findings)
            self.typed_case_bundle["risk_profile"] = risk
            return risk
        except Exception:
            return RiskProfile(urgency=UrgencyTier.ROUTINE, score=0.2)

    def _ensure_typed_findings(self) -> StructuredFindings:
        findings = self._typed_findings()
        if not (findings.summary or findings.positive_findings or findings.red_flags or findings.exposures or findings.raw_segments):
            fallback_findings = self._journal_labels([SignalKind.FINDING], limit=16) or [n.label for n in self.state.fact_graph.nodes if n.type == "symptom"][:16]
            fallback_red_flags = self._journal_labels([SignalKind.RED_FLAG], limit=8)
            fallback_exposures = self._journal_labels([SignalKind.EXPOSURE], limit=8)
            findings = StructuredFindings(
                summary=str(self.state.raw_context or "")[:320],
                positive_findings=fallback_findings[:12],
                red_flags=fallback_red_flags,
                exposures=fallback_exposures,
                raw_segments=[str(self.state.raw_context or "").strip()] if str(self.state.raw_context or "").strip() else [],
            )
        if not findings.phenotype_fingerprint.embedding_terms:
            from src.cdss.clinical.prototype_memory import build_phenotype_fingerprint
            findings = findings.model_copy(update={"phenotype_fingerprint": build_phenotype_fingerprint(findings)})
        self.typed_case_bundle["findings"] = findings
        return findings

    def _ensure_typed_fact_graph(self) -> TypedFactGraph:
        fact_graph = self._typed_fact_graph()
        if fact_graph.nodes:
            return fact_graph
        try:
            from src.cdss.app.service import _build_typed_objects_from_sm
            typed = _build_typed_objects_from_sm(self.state, self._build_patient_input_snapshot())
            fact_graph = typed["fact_graph"]
            self.typed_case_bundle["fact_graph"] = fact_graph
            if "interventions" not in self.typed_case_bundle:
                self.typed_case_bundle["interventions"] = typed["interventions"]
        except Exception:
            fact_graph = TypedFactGraph()
        return fact_graph

    def _prototype_matches(self) -> list[Any]:
        findings = self._ensure_typed_findings()
        cached = self.typed_case_bundle.get("prototype_matches")
        if isinstance(cached, list) and cached:
            return cached
        from src.cdss.clinical.prototype_memory import match_prototypes, mechanism_summaries_from_matches
        matches = match_prototypes(findings.phenotype_fingerprint, limit=6)
        self.typed_case_bundle["prototype_matches"] = matches
        self.typed_case_bundle["mechanism_summaries"] = mechanism_summaries_from_matches(findings.phenotype_fingerprint, matches, limit=3)
        return matches

    def _stable_prototype_matches(self) -> list[Any]:
        matches = list(self._prototype_matches())
        if not matches:
            return []
        top = matches[0]
        second_similarity = float(matches[1].similarity or 0.0) if len(matches) > 1 else 0.0
        top_similarity = float(top.similarity or 0.0)
        support_margin = max(0.0, top_similarity - second_similarity)
        top_quality = (
            top_similarity
            + float(top.slot_overlap or 0.0)
            + float(top.axis_overlap or 0.0)
            + float(top.token_overlap or 0.0)
        ) / 4.0
        stability_score = round((top_quality * 0.78) + (support_margin * 0.22), 2)
        if stability_score < 0.34:
            return []
        floor = max(0.18, top_similarity * 0.72)
        return [match for match in matches if float(match.similarity or 0.0) >= floor][:5]

    def _apply_adaptive_rerank(self, aggregated_hypotheses: dict[str, float], provenance: dict[str, list[str]]) -> list[str]:
        if not aggregated_hypotheses:
            return []
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        interventions = self._typed_interventions()
        evidence = self._typed_evidence()
        matches = self._prototype_matches()
        from src.cdss.clinical.explanation import build_explanation_graph
        from src.cdss.knowledge.registry import load_syndrome_registry

        # Fix-A: backfill from the parallel species-evidence channel so
        # specific-dx votes that were projected to ancestors during family
        # or genus folds still surface as candidates in the differential.
        # The buffer holds {species_label: [scores...]} accumulated by
        # `_fold_wave_into_belief` whenever a species label was emitted
        # at a non-species fold target. We promote the highest score per
        # species into `aggregated_hypotheses` if not already present at
        # a higher value.
        species_buf = self.typed_case_bundle.get("species_evidence_buffer")
        if isinstance(species_buf, dict) and species_buf:
            for species_label, samples in species_buf.items():
                if not species_label or not samples:
                    continue
                # Average then take 0.85× to keep these as supporting
                # evidence rather than overriding direct fusion winners.
                avg = sum(float(s) for s in samples) / max(1, len(samples))
                lifted = round(min(0.95, avg * 0.85), 4)
                norm = normalize_candidate_label(str(species_label)) or str(species_label).strip().lower()
                if not norm:
                    continue
                if aggregated_hypotheses.get(norm, 0.0) < lifted:
                    aggregated_hypotheses[norm] = lifted
                    provenance.setdefault(norm, []).append("hswarm:species_evidence")

        ranked_items = sorted(aggregated_hypotheses.items(), key=lambda item: item[1], reverse=True)
        registry = load_syndrome_registry()
        candidate_support = {str(match.label or ""): float(match.similarity or 0.0) for match in matches if str(match.label or "").strip()}
        candidates: list[DifferentialCandidate] = []
        nodes: list[HypothesisNode] = []
        disease_hypotheses: list[DiseaseHypothesis] = []

        # Pull original rationales from the gateway output so they aren't lost
        existing_diff = self.typed_case_bundle.get("differential", None)
        existing_rationales = {}
        if existing_diff and hasattr(existing_diff, "candidates"):
            for c in existing_diff.candidates:
                norm_key = normalize_candidate_label(c.label) or str(c.label or "").strip()
                existing_rationales[norm_key] = list(c.rationale) if c.rationale else []

        seen: set[str] = set()
        for index, (label, raw_score) in enumerate(ranked_items[:5], start=1):
            normalized = normalize_candidate_label(label) or str(label or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            profile = registry.by_id(normalized)

            # Use original rationale if available, otherwise fallback
            rationale = existing_rationales.get(normalized, [])
            if not rationale:
                rationale = [f"Adaptive swarm consensus from {', '.join(provenance.get(label, [])[:3]) or 'panel'}."]

            if profile and profile.summary:
                rationale.append(profile.summary[:180])
            if normalized in candidate_support:
                rationale.append(f"Prototype support {candidate_support[normalized]:.2f}.")
            base_score = max(0.05, min(0.99, (float(raw_score or 0.0) / max(1.0, float(ranked_items[0][1] or 1.0)))))
            candidates.append(
                DifferentialCandidate(
                    label=normalized,
                    score=round(base_score, 2),
                    rationale=rationale[:3],
                    evidence_needed=list((profile.evidence_needs if profile else [])[:3]),
                    status="candidate",
                )
            )
            nodes.append(
                HypothesisNode(
                    node_id=f"adaptive_hyp_{index}",
                    label=normalized,
                    score=round(base_score, 2),
                    rank=index,
                    rationale=rationale[:3],
                    must_not_miss=bool(getattr(profile, "must_not_miss", False)),
                    dangerous_if_missed=bool(getattr(profile, "dangerous_if_missed", False)),
                    dangerous_if_treated_as=str(getattr(profile, "dangerous_if_treated_as", "") or ""),
                    source="adaptive_swarm",
                    canonical_label=normalized,
                    mapped=(normalized != label),
                )
            )
            disease_hypotheses.append(
                DiseaseHypothesis(
                    label=normalized,
                    score=round(base_score, 2),
                    rationale=rationale[:3],
                    evidence_needed=list((profile.evidence_needs if profile else [])[:3]),
                    unsafe_interventions=list((profile.unsafe_interventions if profile else [])[:3]),
                    status="candidate",
                )
            )

        if not candidates:
            return []

        # 3a: enforce minimum differential breadth (>=3 candidates) when
        # prototype matches are available. Prevents premature closure when the
        # swarm collapses to 1-2 hypotheses. Prototype-fill candidates carry a
        # capped score (max 0.40) so they cannot outrank consensus picks.
        if len(candidates) < 3 and matches:
            for match in matches:
                if len(candidates) >= 3:
                    break
                m_label = normalize_candidate_label(match.label) or str(match.label or "").strip()
                if not m_label or m_label in seen:
                    continue
                seen.add(m_label)
                m_profile = registry.by_id(m_label)
                m_sim = float(match.similarity or 0.0)
                m_score = round(max(0.05, min(0.40, m_sim)), 2)
                m_rationale = [
                    f"Prototype-only candidate (similarity {m_sim:.2f}); included to maintain differential breadth."
                ]
                if m_profile and m_profile.summary:
                    m_rationale.append(m_profile.summary[:180])
                next_rank = len(candidates) + 1
                candidates.append(
                    DifferentialCandidate(
                        label=m_label,
                        score=m_score,
                        rationale=m_rationale[:3],
                        evidence_needed=list((m_profile.evidence_needs if m_profile else [])[:3]),
                        status="candidate",
                    )
                )
                nodes.append(
                    HypothesisNode(
                        node_id=f"adaptive_hyp_{next_rank}",
                        label=m_label,
                        score=m_score,
                        rank=next_rank,
                        rationale=m_rationale[:3],
                        must_not_miss=bool(getattr(m_profile, "must_not_miss", False)),
                        dangerous_if_missed=bool(getattr(m_profile, "dangerous_if_missed", False)),
                        dangerous_if_treated_as=str(getattr(m_profile, "dangerous_if_treated_as", "") or ""),
                        source="prototype_breadth_fill",
                        canonical_label=m_label,
                        mapped=(m_label != str(match.label or "")),
                    )
                )
                disease_hypotheses.append(
                    DiseaseHypothesis(
                        label=m_label,
                        score=m_score,
                        rationale=m_rationale[:3],
                        evidence_needed=list((m_profile.evidence_needs if m_profile else [])[:3]),
                        unsafe_interventions=list((m_profile.unsafe_interventions if m_profile else [])[:3]),
                        status="candidate",
                    )
                )

        # Q2: collapse near-duplicate sibling labels (typo/suffix variants) before
        # building the frontier so they do not split score mass downstream.
        candidates = _dedup_differential_candidates(candidates)

        # Inline grounding gate: drop or demote candidates whose rationale
        # cites findings that are not traceable to the patient narrative.
        # Hallucination control happens here, before the frontier is built,
        # so the verifier never receives a hallucinated anchor.
        try:
            from src.cdss.clinical.inline_grounding import gate_differential as _gate_differential
            _temp_set = DifferentialSet(candidates=candidates)
            _temp_set, _grounding_verdicts, _grounding_pass_rate = _gate_differential(_temp_set, findings)
            candidates = list(_temp_set.candidates)
            self.typed_case_bundle["inline_grounding_pass_rate"] = _grounding_pass_rate
            self.typed_case_bundle["inline_grounding_verdicts"] = {
                label: verdict.model_dump() for label, verdict in _grounding_verdicts.items()
            }
        except Exception:  # noqa: BLE001 — never let grounding crash the pipeline
            pass

        # Re-sync nodes / disease_hypotheses to the surviving label set so the
        # frontier's anchor and must-not-miss markers track the deduped + grounded set.
        surviving_labels = {c.label for c in candidates}
        nodes = [n for n in nodes if n.label in surviving_labels]
        disease_hypotheses = [d for d in disease_hypotheses if d.label in surviving_labels]

        frontier = HypothesisFrontier(
            hypotheses=nodes,
            frontier_entropy=_sm_entropy([item.score for item in candidates]),
            strategy="adaptive_swarm",
            anchor_hypothesis=nodes[0].label if nodes else "",
            must_not_miss=[node.label for node in nodes if node.must_not_miss][:4],
            dangerous_if_treated_as=[node.dangerous_if_treated_as for node in nodes if node.dangerous_if_treated_as][:4],
        )
        differential = DifferentialSet(candidates=candidates, uncertainty=frontier.frontier_entropy, strategy="adaptive_swarm")
        explanation_graph = build_explanation_graph(findings, frontier, interventions, fact_graph)
        model_support = ModelSupportSignals(
            candidate_support={candidate.label: candidate_support.get(candidate.label, float(candidate.score or 0.0)) for candidate in candidates},
            calibrated_prob={candidate.label: candidate_support.get(candidate.label, float(candidate.score or 0.0)) for candidate in candidates},
            support_reason_tags=[axis for axis, _score in sorted(findings.phenotype_fingerprint.axis_weights.items(), key=lambda item: item[1], reverse=True)[:3]],
            effective_support_influence=0.22,
        )
        retrieval_stats = RetrievalRankingStats(
            retrieved_count=max(len(matches), len(evidence.items)),
            reranked_count=max(min(len(matches), len(candidates)), min(len(evidence.items), len(candidates))),
            specificity_gain=max((float(match.similarity or 0.0) for match in matches[:3]), default=0.0),
            query_hygiene_score=1.0 if not evidence.retrieval_intents else round(min(1.0, 0.65 + (len(evidence.retrieval_intents) * 0.08)), 2),
        )
        # Inject case embedding into arbitration for neural cognitive learning
        _case_emb = self.typed_case_bundle.get("case_embedding")
        if _case_emb is None and differential.candidates:
            try:
                from src.cdss.learning.cognitive_engine import get_cognitive_engine
                _top = differential.candidates[0].label
                _case_emb = get_cognitive_engine().encode_case(findings.summary or "", _top)
                self.typed_case_bundle["case_embedding"] = _case_emb
            except Exception:
                _case_emb = None
        self._arbitration._current_case_embedding = _case_emb

        patch = self._arbitration.apply(
            frontier=frontier,
            differential=differential,
            model_support=model_support,
            evidence=evidence,
            retrieval_stats=retrieval_stats,
            intervention_safety=self.typed_case_bundle.get("intervention_safety") if isinstance(self.typed_case_bundle.get("intervention_safety"), InterventionSafetyAssessment) else InterventionSafetyAssessment(),
            findings=findings,
            fact_graph=fact_graph,
            interventions=interventions,
            explanation_graph=explanation_graph,
        )
        final_frontier = patch.hypothesis_frontier or frontier
        final_differential = patch.differential or differential
        final_hypotheses: list[DiseaseHypothesis] = []
        for candidate in final_differential.candidates[:4]:
            profile = registry.by_id(candidate.label)
            final_hypotheses.append(
                DiseaseHypothesis(
                    label=candidate.label,
                    score=round(float(candidate.score or 0.0), 2),
                    rationale=list(candidate.rationale[:3]),
                    evidence_needed=list((profile.evidence_needs if profile else [])[:3]),
                    unsafe_interventions=list((profile.unsafe_interventions if profile else [])[:3]),
                    status="candidate",
                )
            )
        self.typed_case_bundle.update(
            {
                "frontier": final_frontier,
                "differential": final_differential,
                "disease_hypotheses": final_hypotheses or disease_hypotheses,
                "model_support": patch.model_support or model_support,
                "retrieval_stats": patch.retrieval_stats or retrieval_stats,
                "explanation_graph": patch.explanation_graph or explanation_graph,
            }
        )
        self.signal_journal.extend(
            [
                CaseSignal(
                    kind=SignalKind.HYPOTHESIS,
                    label=candidate.label,
                    canonical_label=normalize_candidate_label(candidate.label) or candidate.label,
                    source_stage="adaptive_rerank",
                    confidence=float(candidate.score or 0.0),
                    raw_span=" | ".join(candidate.rationale[:2]),
                )
                for candidate in final_differential.candidates[:4]
            ]
        )
        return [candidate.label for candidate in final_differential.candidates[:3]]

    def _rank_labels_adaptively(self, labels: list[str], *, source: str) -> list[str]:
        aggregated: dict[str, float] = {}
        provenance: dict[str, list[str]] = {}
        for index, raw_label in enumerate(labels[:5], start=1):
            normalized = normalize_candidate_label(raw_label) or str(raw_label or "").strip()
            if not normalized:
                continue
            aggregated[normalized] = max(0.12, 1.0 - ((index - 1) * 0.14))
            provenance[normalized] = [source]
        return self._apply_adaptive_rerank(aggregated, provenance)

    def _adaptive_label_support(self, label: str) -> float:
        normalized = normalize_candidate_label(label)
        if not normalized or self._is_generic_process_label(normalized):
            return 0.0
        support = 0.0
        for match in self._stable_prototype_matches()[:6]:
            if normalized == str(match.label or "").strip():
                support = max(support, float(match.similarity or 0.0))
        differential = self.typed_case_bundle.get("differential")
        if isinstance(differential, DifferentialSet):
            for candidate in differential.candidates[:6]:
                if normalized == str(candidate.label or "").strip():
                    support = max(support, float(candidate.score or 0.0))
                    break
        return round(min(0.99, support), 2)

    def _is_generic_process_label(self, label: str) -> bool:
        normalized = normalize_candidate_label(label)
        return not normalized or normalized == "undifferentiated_critical_process" or normalized.endswith("_process")

    async def _recover_specific_differential(self, reason: str) -> bool:
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        if not (findings.positive_findings or findings.red_flags or fact_graph.nodes):
            return False
        result = await self._gateway.induce_differential(findings, self._typed_risk_profile(), fact_graph)
        self._record_gateway_metrics("differential", result.metrics)
        usable_candidates = [candidate for candidate in result.differential.candidates if not self._is_generic_process_label(candidate.label)]
        if not usable_candidates:
            return False
        self.typed_case_bundle.update(
            {
                "differential": result.differential.model_copy(update={"candidates": usable_candidates[:4]}),
                "frontier": result.frontier.model_copy(
                    update={
                        "hypotheses": [node for node in result.frontier.hypotheses if not self._is_generic_process_label(node.label)][:4],
                        "anchor_hypothesis": usable_candidates[0].label,
                    }
                ),
                "disease_hypotheses": [item for item in result.disease_hypotheses if not self._is_generic_process_label(item.label)][:4],
            }
        )
        self.state.working_hypotheses = [candidate.label for candidate in usable_candidates[:3]]
        existing_diagnoses = {str(node.label or "").strip().lower() for node in self.state.fact_graph.nodes if node.type == "diagnosis"}
        new_signals: list[CaseSignal] = []
        for idx, candidate in enumerate(usable_candidates[:3], start=1):
            key = str(candidate.label or "").strip().lower()
            if key and key not in existing_diagnoses:
                self.state.fact_graph.nodes.append(
                    EvidenceNode(
                        id=f"gw_d_{idx}",
                        type="diagnosis",
                        label=candidate.label,
                        confidence=round(float(candidate.score or 0.0), 2),
                        source="gateway_induction",
                    )
                )
            new_signals.append(
                CaseSignal(
                    kind=SignalKind.HYPOTHESIS,
                    label=candidate.label,
                    canonical_label=normalize_candidate_label(candidate.label) or candidate.label,
                    source_stage=f"gateway_recovery:{reason}",
                    confidence=float(candidate.score or 0.0),
                    raw_span=" | ".join(candidate.rationale[:2]),
                )
            )
        if new_signals:
            self.signal_journal.extend(new_signals)
        await self._emit_trace(
            "verification",
            f"Typed gateway recovered a more specific differential from shared state ({reason}).",
            {
                "reason": reason,
                "recovered_hypotheses": self.state.working_hypotheses,
                "metrics": self._stage_metrics("differential"),
            },
        )
        return True

    async def _run_differential_calibrator(self, risk_profile: RiskProfile) -> None:
        """Post-swarm epidemiological calibration.

        When a catastrophic/must-not-miss diagnosis is top-1 for a non-emergency case,
        performs an internet search for prevalence data and asks the LLM to re-rank using
        Occam's Razor + base-rate epidemiology. Modifies working_hypotheses in place.
        """
        differential = self.typed_case_bundle.get("differential")
        if not differential or not differential.candidates or len(differential.candidates) < 2:
            return

        top1 = differential.candidates[0]
        top_labels = [c.label for c in differential.candidates[:3]]

        # Only fire if top-1 is a must-not-miss/dangerous diagnosis
        try:
            from src.cdss.knowledge.registry import load_syndrome_registry
            registry = load_syndrome_registry()
            top1_profile = registry.by_id(top1.label) if registry else None
            is_catastrophic = bool(top1_profile and (top1_profile.must_not_miss or top1_profile.dangerous_if_missed))
        except Exception:
            is_catastrophic = False

        if not is_catastrophic:
            return  # Top-1 is already a benign/common diagnosis — no calibration needed

        findings = self._ensure_typed_findings()
        await self._emit_trace("hypothesis_generation", f"Calibrator triggered: {top1.label} is catastrophic top-1 for non-emergency case. Running prevalence check.", {"top1": top1.label})

        # Internet search for prevalence context
        epidemiology_context = ""
        try:
            from src.tools.web_search_tool import web_search
            search_query = f"{top1.label.replace('_', ' ')} incidence prevalence primary care outpatient"
            results = await web_search(query=search_query, max_results=2)
            snippets = []
            for r in _normalize_web_search_results(results):
                snippet = str(r.get("content", "") or r.get("snippet", "") or "").strip()[:300]
                if snippet:
                    snippets.append(snippet)
            epidemiology_context = " | ".join(snippets)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        # Ask LLM to re-rank using epidemiology
        try:
            reordered = await self._gateway.calibrate_differential(
                patient_summary=findings.summary or "",
                top_candidates=top_labels,
                epidemiology_context=epidemiology_context,
            )
        except Exception:
            return

        if not reordered or reordered == top_labels:
            return  # Calibrator agreed with current ranking — no change

        new_top1 = reordered[0]
        if new_top1 == top1.label:
            return  # Same top-1

        # Apply the new ranking to working_hypotheses and differential
        await self._emit_trace("hypothesis_generation", f"Calibrator re-ranked: {top1.label} → {new_top1} as top-1.", {"old_top1": top1.label, "new_top1": new_top1, "reordered": reordered})

        # Reorder working_hypotheses
        self.state.working_hypotheses = reordered[:3] + [h for h in self.state.working_hypotheses if h not in reordered][:1]

        # Reorder differential candidates to match
        label_to_candidate = {c.label: c for c in differential.candidates}
        new_candidates = [label_to_candidate[label] for label in reordered if label in label_to_candidate]
        remaining = [c for c in differential.candidates if c.label not in reordered]
        new_candidates = new_candidates + remaining
        if new_candidates:
            self.typed_case_bundle["differential"] = differential.model_copy(update={"candidates": new_candidates[:5]})

    def _lightweight_rerank_with_signals(
        self,
        *,
        epi_prior: Any,
        specificity_judgments: dict | None,
    ) -> None:
        """Re-run arbitration with epi_prior and specificity_judgments already computed.

        Does NOT rebuild candidates from scratch — reads current state from typed_case_bundle
        and re-runs only the arbitration scoring pass with the new signals.
        """
        from src.cdss.clinical.explanation import build_explanation_graph

        frontier = self.typed_case_bundle.get("frontier")
        differential = self.typed_case_bundle.get("differential")
        if not frontier or not differential or not differential.candidates:
            return

        model_support = self.typed_case_bundle.get("model_support") or ModelSupportSignals()
        evidence = self._typed_evidence()
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        interventions = self._typed_interventions()
        retrieval_stats = self.typed_case_bundle.get("retrieval_stats") or RetrievalRankingStats()
        intervention_safety = (
            self.typed_case_bundle.get("intervention_safety")
            if isinstance(self.typed_case_bundle.get("intervention_safety"), InterventionSafetyAssessment)
            else InterventionSafetyAssessment()
        )
        explanation_graph = self.typed_case_bundle.get("explanation_graph")
        if explanation_graph is None:
            explanation_graph = build_explanation_graph(findings, frontier, interventions, fact_graph)

        try:
            # Inject case embedding into arbitration for neural cognitive learning
            _case_emb_v = self.typed_case_bundle.get("case_embedding")
            if _case_emb_v is None and differential.candidates:
                try:
                    from src.cdss.learning.cognitive_engine import get_cognitive_engine
                    _top_v = differential.candidates[0].label
                    _case_emb_v = get_cognitive_engine().encode_case(findings.summary or "", _top_v)
                    self.typed_case_bundle["case_embedding"] = _case_emb_v
                except Exception:
                    _case_emb_v = None
            self._arbitration._current_case_embedding = _case_emb_v

            patch = self._arbitration.apply(
                frontier=frontier,
                differential=differential,
                model_support=model_support,
                evidence=evidence,
                retrieval_stats=retrieval_stats,
                intervention_safety=intervention_safety,
                findings=findings,
                fact_graph=fact_graph,
                interventions=interventions,
                explanation_graph=explanation_graph,
                epi_prior=epi_prior,
                specificity_judgments=specificity_judgments,
                rule_out_labels=set(frontier.must_not_miss) if frontier.must_not_miss else None,
            )
            if patch.differential and patch.differential.candidates:
                self.typed_case_bundle["differential"] = patch.differential
                if patch.hypothesis_frontier:
                    self.typed_case_bundle["frontier"] = patch.hypothesis_frontier
                if patch.model_support:
                    self.typed_case_bundle["model_support"] = patch.model_support
                self.state.working_hypotheses = [
                    c.label for c in patch.differential.candidates[:3]
                ]
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

    async def _run_epi_spec_calibration(self) -> None:
        """Compute epidemiological priors and specificity judgments, then re-rank.

        Called after _run_differential_calibrator. Computes:
        1. EpiPriorResult — LLM-assessed prevalence tiers for each candidate
        2. SpecificityJudgments — LLM-assessed root-vs-complication for each candidate

        Both are stored in typed_case_bundle and fed into a lightweight re-arbitration pass.
        This is gated by policy.epi_prior_enabled (default=True).
        """
        if not getattr(self.runtime_policy, "epi_prior_enabled", True):
            return

        differential = self.typed_case_bundle.get("differential")
        if not differential or not differential.candidates:
            return

        findings = self._ensure_typed_findings()
        risk_profile = self._typed_risk_profile()
        candidates = [c.label for c in differential.candidates[:5]]

        # Build a thin async LLM adapter that wraps the bridge's chat() interface
        bridge = self._gateway.bridge

        class _BridgeAdapter:
            """Adapts bridge._client().chat() to the complete(prompt) interface."""

            def __init__(self, bridge_ref: Any) -> None:
                self._bridge = bridge_ref

            async def complete(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
                import asyncio
                try:
                    client = self._bridge._client()
                    messages = [{"role": "user", "content": prompt}]
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.chat(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stage="epi",
                        ),
                    )
                    return str(response.content or "")
                except Exception:
                    return ""

            def complete_sync(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
                try:
                    client = self._bridge._client()
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stage="epi",
                    )
                    return str(response.content or "")
                except Exception:
                    return ""

        adapter = _BridgeAdapter(bridge)

        # Step 1: Epidemiological prior reasoning (async)
        epi_prior = None
        try:
            from src.cdss.reasoning.clinical_scoring import assess_epidemiological_priors

            # Collect prevalence snippets from evidence (relation_type="prevalence")
            pubmed_snippets: list[str] = []
            if getattr(self.runtime_policy, "epi_prior_pubmed_enabled", False):
                evidence = self._typed_evidence()
                for item in (evidence.items or []):
                    if str(item.relation_type or "").strip().lower() == "prevalence":
                        snippet = str(item.summary or item.content or "").strip()
                        if snippet:
                            pubmed_snippets.append(snippet[:200])

            epi_prior = await assess_epidemiological_priors(
                candidates,
                findings=findings,
                risk_profile=risk_profile,
                pubmed_snippets=pubmed_snippets or None,
                llm_client=adapter,
                policy=self.runtime_policy,
            )
            self.typed_case_bundle["epi_prior_result"] = epi_prior
            await self._emit_trace(
                "hypothesis_generation",
                "Epidemiological prior calibration complete.",
                {
                    "tiers": {t.label: t.tier for t in (epi_prior.tiers or [])},
                    "pubmed_grounded": epi_prior.pubmed_grounded,
                },
            )
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        # Step 2: Specificity calibration (fully async — no executor wrapping)
        specificity_judgments: dict | None = None
        if getattr(self.runtime_policy, "specificity_calibrator_enabled", True):
            try:
                from src.cdss.reasoning.specificity_calibrator import calibrate_specificity_async

                fact_graph = self._ensure_typed_fact_graph()
                specificity_judgments = await calibrate_specificity_async(
                    candidates,
                    findings=findings,
                    fact_graph=fact_graph,
                    llm_client=adapter,
                    policy=self.runtime_policy,
                )
                self.typed_case_bundle["specificity_judgments"] = specificity_judgments
                await self._emit_trace(
                    "hypothesis_generation",
                    "Specificity calibration complete.",
                    {
                        "judgments": {
                            label: {
                                "role": j.role,
                                "evidence_supports": j.evidence_supports_specificity,
                                "score": j.calibrated_specificity_score,
                            }
                            for label, j in (specificity_judgments or {}).items()
                        }
                    },
                )
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass

        # Step 2b: Inject root/parent diseases that are implied by subtype candidates.
        # When the LLM generates a specific subtype (e.g. acute_cystitis) without its
        # parent (urinary_tract_infection), the parent is never scored.  extract_parent_injections
        # reads SpecificityJudgment.parent_label for unsupported subtypes/complications and
        # returns their parent labels.  We inject them at child_score - 0.05 so they enter
        # arbitration and can outscore the child when epi prior favours the root disease.
        if specificity_judgments:
            try:
                from src.cdss.reasoning.specificity_calibrator import extract_parent_injections

                _differential = self.typed_case_bundle.get("differential")
                if _differential and _differential.candidates:
                    _existing_labels = {c.label for c in _differential.candidates}
                    _injected_count = 0
                    for _parent_label in extract_parent_injections(specificity_judgments):
                        if _parent_label in _existing_labels or _injected_count >= 2:
                            continue
                        # Find the child candidate that owns this parent
                        _child = next(
                            (
                                c for c in _differential.candidates
                                if specificity_judgments.get(c.label) is not None
                                and specificity_judgments[c.label].parent_label == _parent_label
                            ),
                            None,
                        )
                        if _child is None:
                            continue
                        # When evidence does NOT support the child's specificity level,
                        # inject the parent ABOVE the child score so it can win in arbitration
                        # after the child receives its -0.14 unsupported-subtype penalty there.
                        _child_judgment = specificity_judgments.get(_child.label)
                        _evidence_supports = (
                            _child_judgment.evidence_supports_specificity
                            if _child_judgment is not None
                            else True
                        )
                        _parent_boost = -0.05 if _evidence_supports else 0.06
                        _injected = DifferentialCandidate(
                            label=_parent_label,
                            score=max(0.10, round(_child.score + _parent_boost, 2)),
                            rationale=["injected_parent_from_specificity_calibration"],
                            status="injected",
                        )
                        _differential.candidates.append(_injected)
                        _existing_labels.add(_parent_label)
                        _injected_count += 1
                    if _injected_count > 0:
                        self.typed_case_bundle["differential"] = _differential
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass

        # Step 3: Re-run arbitration with new signals
        if epi_prior is not None or specificity_judgments:
            self._lightweight_rerank_with_signals(
                epi_prior=epi_prior,
                specificity_judgments=specificity_judgments,
            )

    def _load_knowledge_graph(self) -> Dict[str, Any]:
        """Loads semantic associations generated by past swarm interactions."""
        if os.path.exists(MEMORY_GRAPH_PATH):
            try:
                with open(MEMORY_GRAPH_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"lessons": [], "correlations": {}}
        return {"lessons": [], "correlations": {}}

    def _save_knowledge_graph(self):
        """Persists the dynamic semantic network for future few-shot grounding."""
        os.makedirs(os.path.dirname(MEMORY_GRAPH_PATH), exist_ok=True)
        with open(MEMORY_GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_graph, f, indent=2, ensure_ascii=False)

    async def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> tuple[str, float]:
        """Asynchronous HTTP LLM caller with per-request mode selection and env fallback."""
        start_t = time.time()
        current_mode = _execution_mode_to_llm_mode(self.requested_execution_mode) or _get_llm_mode()
        prompt_chars = len(system_prompt or "") + len(user_prompt or "")
        try:
            client = await self._get_http_client()
            if current_mode == "GEMINI":
                api_key = _get_gemini_api_key()
                if not api_key:
                    print("[SWARM] GEMINI mode but no API key found (checked GEMINI_API_KEY and GOOGLE_API_KEY). Falling back to LOCAL.")
                    current_mode = "LOCAL"
                else:
                    self.active_llm_mode = "GEMINI"
                    self.active_engine_model = "gemini-2.5-flash-preview-04-17"
                    endpoint = _get_gemini_endpoint()
                    payload = {
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
                    }
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    elapsed = time.time() - start_t
                    self._record_stage_profile(
                        model_wait_s=elapsed,
                        wall_time_s=elapsed,
                        prompt_chars=prompt_chars,
                        output_chars=len(content),
                        llm_calls=1,
                    )
                    return content, elapsed
            # LOCAL mode (or fallback)
            self.active_llm_mode = "LOCAL"
            self.active_engine_model = "qwen"
            payload = {
                "model": "qwen",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": ["<|im_end|>"]
            }
            response = await client.post(QWEN_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            content = re.sub(r'(?:\*+\s*)+(?:Review|Draft|Refine|Correction|Revised Draft|Wait|Revised Plan|Constraint|Interpretation).*', '', content, flags=re.IGNORECASE|re.DOTALL)
            content = content.strip()
            elapsed = time.time() - start_t
            self._record_stage_profile(
                model_wait_s=elapsed,
                wall_time_s=elapsed,
                prompt_chars=prompt_chars,
                output_chars=len(content),
                llm_calls=1,
            )
            return content, elapsed
        except Exception as e:
            logger.error("[SWARM LLM ERROR] mode=%s error=%s", current_mode, e, exc_info=True)
            return "", time.time() - start_t
    async def run(self):
        _PHASE_HANDLERS = {
            CdssPhase.INTAKE:               self._run_intake_node,
            CdssPhase.R2:                   self._run_r2_research,
            CdssPhase.DIFFERENTIAL:         self._run_expert_swarm,
            CdssPhase.COGNITIVE_LOOP:       self._run_cognitive_loop,
            CdssPhase.BACKWARD_SIMULATION:  self._run_backward_causality_simulation,
            CdssPhase.OUTCOME_SIMULATION:   self._run_test_time_simulation,
            CdssPhase.VERIFICATION:         self._run_chief_consilium,
            CdssPhase.ACTION_PLAN:          self._run_action_plan,
        }
        try:
            while self.state.phase not in {CdssPhase.DONE, CdssPhase.FAILED}:
                handler = _PHASE_HANDLERS.get(self.state.phase)
                if handler is not None:
                    await handler()
                else:
                    self.state.transition(CdssPhase.FAILED)
            return self.state
        finally:
            await self._close_http_client()

    async def _run_intake_node(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("intake")
        patient_input = self._build_patient_input_snapshot()
        try:
            intake_result = await self._gateway.extract_case(patient_input)
        except Exception as e:
            logger.error("[INTAKE] gateway.extract_case failed: %s", e, exc_info=True)
            intake_result = None
        if intake_result and (
            intake_result.findings.positive_findings
            or intake_result.findings.red_flags
            or intake_result.findings.exposures
            or intake_result.fact_graph.nodes
        ):
            self.active_llm_mode = "GEMINI" if self._gateway.active_mode == "cloud" else "LOCAL"
            self.active_engine_model = self._gateway.active_engine_model or self.active_engine_model
            self.typed_case_bundle.update(
                {
                    "findings": intake_result.findings,
                    "fact_graph": intake_result.fact_graph,
                    "interventions": intake_result.interventions,
                }
            )
            self._record_findings_signals(intake_result.findings, intake_result.interventions)
            self._sync_legacy_fact_graph_from_typed(intake_result.findings, intake_result.fact_graph)
            self._record_gateway_metrics("intake", intake_result.metrics)
            self._record_stage_profile(stage="intake", wall_time_s=time.perf_counter() - stage_started)
            await self._emit_trace(
                "extractor",
                f"Typed intake populated the signal journal with {len(self.signal_journal.signals)} clinical atoms.",
                {
                    "count": len(self.state.fact_graph.nodes),
                    "signals": len(self.signal_journal.signals),
                    "metrics": self._stage_metrics("intake", wall_time_s=time.perf_counter() - stage_started),
                },
            )
            self.state.transition(CdssPhase.R2)
            return
        print(f"[{time.strftime('%H:%M:%S')}] [ğŸ“¡ Intake Stream] Extracting multivariable clinical signals...")
        narrative = str(self.state.raw_context or "").strip()
        self.typed_case_bundle["findings"] = StructuredFindings(
            summary=narrative[:320],
            raw_segments=[narrative] if narrative else [],
        )
        if narrative:
            self.signal_journal.extend(
                [
                    CaseSignal(
                        kind=SignalKind.FINDING,
                        label=narrative[:160],
                        source_stage="intake_fallback",
                        raw_span=narrative,
                        confidence=0.15,
                    )
                ]
            )

        self._record_stage_profile(stage="intake", wall_time_s=time.perf_counter() - stage_started)
        await self._emit_trace("extractor", "Typed intake was unavailable; preserving only the raw narrative envelope for downstream abstention-safe reasoning.", {
            "count": 0,
            "metrics": self._stage_metrics("intake", wall_time_s=time.perf_counter() - stage_started)
        })
        self.state.transition(CdssPhase.R2)

    async def _run_r2_research(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("r2")
        print(f"[{time.strftime('%H:%M:%S')}] [R2 Research] Activating Medical Researcher Agent for evidence gathering...")

        await self._emit_trace("r1_reasoned", "Parsing temporal events to avoid contradictions...", {})
        await self._emit_trace("evidence_planning", "Coordinating specialized agents and researching medical databases...", {})

        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        frontier = self._provisional_frontier()
        seed_bundle = self._research_seed_bundle(frontier)
        needs = list(seed_bundle.evidence_needs[:8])

        # 3. Dynamic Priority (RAG): If high risk demographic is present, bias search towards atypical presentations
        risk_flags = " ".join([str(item) for item in findings.positive_findings + findings.input_context + findings.suspected_conditions]).lower()

        if "elderly" in risk_flags or "old" in risk_flags or "diabet" in risk_flags or "hypertension" in risk_flags:
            needs.append(
                EvidenceNeed(
                    objective="atypical cardiovascular or silent metabolic presentation in elderly diabetic presenting with vague gastrointestinal symptoms",
                    rationale="Dynamic heuristic: Elderly diabetic presentation is prone to atypical critical illnesses like silent MI."
                )
            )

        # Get loop iteration for ATOM modes
        loop_iteration = self.typed_case_bundle.get("deep_reasoning_loop", 0)

        # Phase 2 (parallelism): run planner + memory recall concurrently.
        # Memory recall is a local O(N) scan; the planner is an LLM call.
        # Both are independent — saves ~5-10s per R2 pass.
        _memory_limit = 4 if self._typed_risk_profile().urgency != UrgencyTier.ROUTINE else 2
        _memory_validated_only = (self.runtime_policy.memory_inference_policy == "validated_only")

        async def _recall_memory_async():
            try:
                return await asyncio.to_thread(
                    self._evidence_memory.recall,
                    summary=findings.summary,
                    findings=findings,
                    case_id=str(self.state.patient_id or ""),
                    hypotheses=[item.label for item in frontier.hypotheses[:4]],
                    discriminators=[
                        str(item.desired_discriminator or item.objective or "").strip()
                        for item in needs[:8]
                        if str(item.desired_discriminator or item.objective or "").strip()
                    ],
                    limit=_memory_limit,
                    validated_only=_memory_validated_only,
                )
            except Exception:
                return []

        plan, _r2_precomputed_memory_hits = await asyncio.gather(
            self._gateway.plan_research(findings, fact_graph, frontier, needs, iteration=loop_iteration),
            _recall_memory_async(),
        )
        self._record_gateway_metrics("r2", plan.metrics)

        domains = list(plan.active_hypotheses[:3]) or [
            str(intent.target_candidate or "").strip()
            for intent in seed_bundle.retrieval_intents[:3]
            if str(intent.target_candidate or "").strip()
        ] or [str(match.label or "") for match in self._stable_prototype_matches()[:3] if str(match.label or "").strip()]
        self.typed_case_bundle["domains"] = domains[:3]
        await self._emit_trace(
            "evidence_planning",
            f"Domains prioritized for swarm focus: {', '.join(domains) if domains else 'adaptive differential research'}",
            {
                "domains": domains[:3],
                "queries": [query for _, query in sorted(plan.queries.items())[:3]],
                "metrics": self._stage_metrics("r2"),
            },
        )

        from src.tools.web_search_tool import web_search

        if plan.error:
            self.typed_case_bundle["research_error"] = dict(plan.error)
            await self._emit_trace(
                "evidence_planning",
                "Research planner returned no direct structured output; open-web retrieval will remain blocked until the planner emits stable queries.",
                {"error": plan.error, "metrics": self._stage_metrics("r2")},
            )

        # Hard Web-Search Override for ATOM Deep Reasoning mode (first-pass compatible)
        # Previously required loop_iteration > 0; now fires on first pass when queries are empty.
        if self.deep_thinking and not plan.queries:
            top_hypo = frontier.anchor_hypothesis or "unknown condition"
            forced_query = f"latest guidelines pathognomonic signs '{top_hypo}' vs mimics"
            plan.queries = {"1": forced_query}
            await self._emit_trace(
                "evidence_planning",
                "⚕️ ATOM DEEP REASONING: Planner provided no queries, but ATOM forces a mandatory external literature scan for edge cases.",
                {"forced_query": forced_query}
            )

        # Entropy-gated fallback: when planner is blocked AND case complexity warrants retrieval,
        # auto-generate queries from the anchor hypothesis + blocked actions (fully dynamic, no hardcoded terms).
        if not plan.queries:
            _fallback_entropy = _sm_entropy([h.score for h in frontier.hypotheses])
            _fallback_risk = self._typed_risk_profile()
            _fallback_complexity = _compute_case_complexity(_fallback_entropy, _fallback_risk)
            if _fallback_complexity in ("elevated", "critical"):
                anchor_text = (frontier.anchor_hypothesis or "unknown").replace("_", " ")
                blocked = [str(a) for a in getattr(_fallback_risk, "blocked_actions", [])[:3]]
                fallback_queries: dict[str, str] = {
                    "1": f"{anchor_text} diagnosis criteria treatment guidelines",
                    "2": f"{anchor_text} dangerous treatment contraindications iatrogenic risk",
                }
                if blocked:
                    fallback_queries["3"] = f"{anchor_text} {' '.join(blocked[:2])} complication risk"
                plan.queries = fallback_queries
                await self._emit_trace(
                    "evidence_planning",
                    f"Entropy-gated fallback: planner blocked, auto-generated {len(fallback_queries)} queries (complexity={_fallback_complexity}, H={_fallback_entropy:.2f}).",
                    {"fallback_queries": list(fallback_queries.values()), "entropy": _fallback_entropy, "complexity": _fallback_complexity},
                )

        query_items = [(int(index), str(query).strip()) for index, query in sorted(plan.queries.items()) if str(query).strip()]
        if not query_items:
            await self._emit_trace(
                "evidence_planning",
                "Research planner did not return direct structured queries; skipping heuristic web fallback and keeping retrieval memory-first.",
                {"metrics": self._stage_metrics("r2"), "memory_only": True},
            )
        else:
            await self._emit_trace("r1_reasoned", "Searching internet for clinical guidelines...", {})
        self.state.verification_queries = [query for _, query in query_items[:6]]

        evidence_items: list[EvidenceItem] = []
        retrieval_intents: list[RetrievalIntent] = []
        evidence_signals: list[CaseSignal] = []
        seen_sources: set[str] = set()
        total_cache_hits = 0
        total_web_hits = 0
        # Reuse memory recall precomputed in parallel with planner above (Phase 2 speedup).
        memory_hits = _r2_precomputed_memory_hits if isinstance(_r2_precomputed_memory_hits, list) else []
        for hit in memory_hits:
            evidence_refs = list(hit.get("evidence_refs", []) or [])
            first_ref = dict(evidence_refs[0] or {}) if evidence_refs else {}
            title = str(first_ref.get("title", "") or (hit.get("evidence_titles") or [""])[0] or "").strip()
            if not title:
                continue
            source_key = f"memory::{str(hit.get('case_id', '') or '')}::{title.lower()}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            linked_hypotheses = list(first_ref.get("linked_hypotheses", []) or hit.get("linked_hypotheses", []) or [])
            relation_type = str(first_ref.get("relation_type", "") or hit.get("relation_type", "") or "support").strip().lower()
            evidence_items.append(
                EvidenceItem(
                    source="memory",
                    title=title,
                    citation=str(first_ref.get("citation", "") or f"memory://{hit.get('case_id', 'case')}").strip(),
                    excerpt=str(hit.get("summary", "") or "")[:400],
                    trust_score=0.46 if bool(hit.get("validated", False)) else 0.34,
                    pmid=str(first_ref.get("pmid", "") or ""),
                    doi=str(first_ref.get("doi", "") or ""),
                    origin_query=str((hit.get("queries") or [""])[0] or ""),
                    linked_hypotheses=linked_hypotheses[:4],
                    relation_type=relation_type or "support",
                    verification_status="verified" if bool(hit.get("validated", False)) else "unverified",
                )
            )
            self.state.fact_graph.nodes.append(
                EvidenceNode(
                    id=f"m2_{len(self.state.fact_graph.nodes) + 1}",
                    type="evidence",
                    label=title,
                    confidence=0.72 if bool(hit.get("validated", False)) else 0.42,
                    source="r2_memory",
                )
            )
            evidence_signals.append(
                CaseSignal(
                    kind=SignalKind.EVIDENCE,
                    label=title,
                    source_stage="r2_memory",
                    raw_span=str(hit.get("summary", "") or "")[:400] or title,
                    confidence=0.72 if bool(hit.get("validated", False)) else 0.42,
                )
            )
        # Phase 1: build RetrievalIntent metadata (no I/O — pure computation)
        _pending_searches: list[tuple[int, str, Any]] = []
        for index, query in query_items[:3]:
            need = needs[index - 1] if 0 <= index - 1 < len(needs) else None
            retrieval_intents.append(
                RetrievalIntent(
                    objective=str(need.objective if need else f"query_{index}"),
                    decision_target=str(need.decision_target if need else "confirm"),
                    query_hint=query,
                    target_candidate=str(need.hypothesis_label if need else frontier.anchor_hypothesis),
                    active_state=str(need.active_state if need else frontier.anchor_hypothesis),
                    unresolved_critical_finding=str(need.unresolved_critical_finding if need else ""),
                    rival_mechanism=str(need.rival_mechanism if need else ""),
                    action_hazard=str(need.action_hazard if need else ""),
                    desired_discriminator=str(need.desired_discriminator if need else ""),
                    priority=float(need.priority if need else 0.45),
                    decision_relevance=float(need.decision_relevance if need else 0.45),
                    expected_value=float(need.expected_information_gain if need else 0.35),
                )
            )
            _pending_searches.append((index, query, need))

        # Phase 2: fire all web searches concurrently (3 independent HTTP calls → asyncio.gather)
        _search_results_list = await asyncio.gather(
            *[web_search(query=q, max_results=3) for _, q, _ in _pending_searches],
            return_exceptions=True,
        )

        # Phase 3: process results — cache accounting + EvidenceItem/EvidenceNode/CaseSignal construction
        for (_index, query, need), results in zip(_pending_searches, _search_results_list):
            if isinstance(results, BaseException):
                results = {}
            cache_hit = _web_search_result_from_cache(results)
            total_cache_hits += 1 if cache_hit else 0
            total_web_hits += 0 if cache_hit else 1
            normalized_results = _normalize_web_search_results(results)
            for result in normalized_results[:3]:
                url = str(result.get("url", "") or "").strip()
                source_key = url or str(result.get("title", "") or "").strip().lower()
                if source_key and source_key in seen_sources:
                    continue
                if source_key:
                    seen_sources.add(source_key)
                title = str(result.get("title", "") or result.get("source", "") or query).strip()
                excerpt = str(result.get("content", "") or result.get("snippet", "") or "").strip()
                if not (title or excerpt):
                    continue
                evidence_items.append(
                    EvidenceItem(
                        source=url or "web_search",
                        title=title or query,
                        citation=url,
                        excerpt=excerpt[:600],
                        trust_score=0.58 if url else 0.42,
                        origin_query=query,
                        linked_hypotheses=[str(need.hypothesis_label)] if need and str(need.hypothesis_label).strip() else list(frontier.must_not_miss[:1]),
                        relation_type="discriminator" if need and need.decision_target == "discriminate" else "support",
                        verification_status="retrieved",
                    )
                )
                label = title or excerpt[:140]
                self.state.fact_graph.nodes.append(
                    EvidenceNode(
                        id=f"r2_{len(self.state.fact_graph.nodes) + 1}",
                        type="evidence",
                        label=label,
                        confidence=0.78 if url else 0.55,
                        source="r2_research",
                    )
                )
                evidence_signals.append(
                    CaseSignal(
                        kind=SignalKind.EVIDENCE,
                        label=label,
                        source_stage="r2_research",
                        raw_span=excerpt[:600] or label,
                        confidence=0.78 if url else 0.55,
                    )
                )

        self._record_stage_profile(stage="r2", cache_hits=total_cache_hits, web_hits=total_web_hits)
        query_quality = summarize_query_quality(retrieval_intents[:6])
        coverage = round(
            max(
                float(seed_bundle.coverage or 0.0),
                min(
                    1.0,
                    float(seed_bundle.coverage or 0.0)
                    + (len(evidence_items[:12]) * 0.04)
                    + (float(query_quality.get("pairwise_discriminator_coverage", 0.0) or 0.0) * 0.08),
                ),
            ),
            2,
        )
        contradiction_mass = round(
            max(
                float(seed_bundle.contradiction_mass or 0.0),
                0.22 if bool(query_quality.get("hazard_leakage_detected", False)) else 0.0,
            ),
            2,
        )
        evidence_bundle = seed_bundle.model_copy(
            update={
                "items": evidence_items[:12],
                "coverage": coverage,
                "contradiction_mass": contradiction_mass,
                "retrieval_intents": retrieval_intents[:6],
                "evidence_needs": needs[:8],
            }
        )
        self.typed_case_bundle["evidence"] = evidence_bundle
        self.typed_case_bundle["retrieval_stats"] = RetrievalRankingStats(
            retrieved_count=len(evidence_bundle.items),
            reranked_count=min(len(evidence_bundle.items), max(1, len(frontier.hypotheses))),
            specificity_gain=max((float(match.similarity or 0.0) for match in self._stable_prototype_matches()[:3]), default=0.0),
            novelty_gain=float(query_quality.get("novelty_gain", 0.0) or 0.0),
            repeated_query_ratio=float(query_quality.get("repeated_query_ratio", 0.0) or 0.0),
            semantic_dedup_ratio=float(query_quality.get("semantic_dedup_ratio", 0.0) or 0.0),
            pairwise_discriminator_coverage=float(query_quality.get("pairwise_discriminator_coverage", 0.0) or 0.0),
            marginal_utility_score=float(query_quality.get("marginal_utility_score", 0.0) or 0.0),
            query_hygiene_score=float(query_quality.get("query_hygiene_score", 0.0) or 0.0),
            hazard_leakage_rate=float(query_quality.get("hazard_leakage_rate", 0.0) or 0.0),
            hazard_leakage_detected=bool(query_quality.get("hazard_leakage_detected", False)),
            research_iterations=1,
        )
        if evidence_signals:
            self.signal_journal.extend(evidence_signals[:16])

        self._record_stage_profile(stage="r2", wall_time_s=time.perf_counter() - stage_started)
        await self._emit_trace("evidence_planning", f"Adaptive retrieval attached {len(evidence_bundle.items)} evidence items and {len(retrieval_intents)} typed intents.", {
            "metrics": self._stage_metrics("r2", wall_time_s=time.perf_counter() - stage_started),
            "domains": domains[:3],
            "cache_hit": bool(total_cache_hits),
            "memory_hits": len(memory_hits),
        })
        self.state.transition(CdssPhase.DIFFERENTIAL)

    def _fold_swarm_into_belief(
        self,
        *,
        induced_results: list,
        aggregated_hypotheses: dict[str, float],
    ) -> None:
        """Project swarm worker outputs into the DiagnosticBelief (W1 Module B.3).

        Each worker contributes a per-candidate score in [0,1]; aggregated as
        mean across workers. Treated as a single pseudo-likelihood finding
        (`swarm_aggregate`) and folded via log-space Bayes update against a
        uniform-seeded species α. Result parked under
        `typed_case_bundle['diagnostic_belief']`.

        Feature-gated on `bayesian_posterior_enabled`. No-op otherwise.
        """
        policy = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
        if policy is None:
            try:
                from src.cdss.runtime.policy import load_runtime_policy
                policy = load_runtime_policy()
            except Exception:
                return
        if not bool(getattr(policy, "bayesian_posterior_enabled", False)):
            return

        # Lazy imports to keep cold-start fast.
        try:
            from src.cdss.contracts.models import DiagnosticBelief
            from src.cdss.reasoning.bayes_update import update_posterior
            from src.cdss.reasoning.belief_propagation import build_parent_maps
            from src.cdss.reasoning.likelihood_ensemble import LikelihoodEstimate
        except Exception as exc:
            _log = logging.getLogger("rrrie-cdss")
            _log.warning("[BELIEF] belief modules unavailable: %s", exc)
            return

        # Per-candidate sample collection across workers.
        # W6.2 J.2 — when a worker carries ensemble_stats (n>1 temperature
        # samples), we extend the sample list with all temperature samples
        # rather than just the merged mean. This grows the variance signal
        # naturally inside `_method_of_moments_beta`.
        per_hyp_samples: dict[str, list[float]] = {}
        for r in induced_results or []:
            try:
                ens = dict(getattr(r, "ensemble_stats", {}) or {})
                for cand in (r.differential.candidates or [])[:10]:
                    label = normalize_candidate_label(str(cand.label or "")) or str(cand.label or "").strip().lower()
                    if not label:
                        continue
                    score = max(0.0, min(1.0, float(cand.score or 0.0)))
                    # Prefer per-temperature samples when ensemble fired.
                    raw_samples = ens.get(label, {}).get("samples") if ens.get(label) else None
                    if raw_samples:
                        for s in raw_samples:
                            try:
                                per_hyp_samples.setdefault(label, []).append(max(0.0, min(1.0, float(s))))
                            except Exception:  # noqa: BLE001
                                logger.debug("swallowed exception", exc_info=True)
                                continue
                    else:
                        per_hyp_samples.setdefault(label, []).append(score)
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                continue

        if not per_hyp_samples:
            return

        # Build one aggregate LikelihoodEstimate per hypothesis.
        likelihood_row: dict[str, LikelihoodEstimate] = {}
        for h, samples in per_hyp_samples.items():
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            var = (
                sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)
                if len(samples) >= 2 else 0.0
            )
            mean = max(1e-4, min(1.0 - 1e-4, mean))
            likelihood_row[h] = LikelihoodEstimate(
                hypothesis=h,
                finding="swarm_aggregate",
                mean=mean,
                variance=max(0.0, var),
                alpha=mean * 3.0 + 1.0,
                beta=(1.0 - mean) * 3.0 + 1.0,
                samples=list(samples),
                grounded=True,
            )

        if not likelihood_row:
            return

        # Load or seed belief from bundle.
        existing = self.typed_case_bundle.get("diagnostic_belief")
        if isinstance(existing, DiagnosticBelief):
            belief = existing
        elif isinstance(existing, dict):
            try:
                belief = DiagnosticBelief.model_validate(existing)
            except Exception:
                belief = DiagnosticBelief()
        else:
            belief = DiagnosticBelief()

        # Seed species_alpha uniformly across candidates if empty.
        if not belief.species_alpha:
            belief = belief.model_copy(update={
                "species_alpha": {h: 1.0 for h in likelihood_row.keys()},
            })

        # Parent maps for down-propagation (may be empty if hierarchy not shipped).
        try:
            registry = load_syndrome_registry()
            parent_of_genus, parent_of_species = build_parent_maps(registry.profiles)
        except Exception:
            parent_of_genus, parent_of_species = {}, {}

        try:
            updated = update_posterior(
                belief,
                level=2,
                likelihoods={"swarm_aggregate": likelihood_row},
                parent_of_genus=parent_of_genus,
                parent_of_species=parent_of_species,
                source="swarm",
            )
        except Exception as exc:
            _log = logging.getLogger("rrrie-cdss")
            _log.warning("[BELIEF] update_posterior failed: %s", exc)
            return

        # Persist (dict form for JSON serialization later).
        self.typed_case_bundle["diagnostic_belief"] = updated

    # ------------------------------------------------------------------
    # W2 Module E — Dempster-Shafer fusion across correlated swarm agents
    # ------------------------------------------------------------------

    def _fold_ds_into_belief(
        self,
        *,
        induced_results: list,
        level: int,
    ) -> None:
        """Build per-worker mass functions, combine, fuse with Bayes posterior.

        Captures epistemic uncertainty Bayes alone cannot — workers share a
        prompt so their votes are NOT independent; the orthogonal sum rule
        surfaces conflict mass K explicitly. When K ≥ CONFLICT_REJECT_THRESHOLD
        the combination is skipped and we log a premise-conflict warning.

        κ=0.6 Bayes + 0.4 pignistic-D-S per Module E plan spec.
        Gated on `dempster_shafer_fusion_enabled`.
        """
        try:
            pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                pol = load_runtime_policy()
            if not bool(getattr(pol, "dempster_shafer_fusion_enabled", False)):
                return
        except Exception:
            return

        try:
            from src.cdss.contracts.models import DiagnosticBelief
            from src.cdss.reasoning.dempster_shafer import (
                CONFLICT_REJECT_THRESHOLD,
                combine_many,
                fuse_with_bayes,
                mass_from_swarm_vote,
            )
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[DS] modules unavailable: %s", exc)
            return

        # Level-strict allowed set: DS masses must live in the same hypothesis
        # space as the Bayesian posterior we're fusing into, else κ-mix mixes
        # apples with oranges.
        try:
            _reg_ds = load_syndrome_registry()
            _ds_allowed = {
                normalize_candidate_label(str(p.id))
                for p in _reg_ds.by_level(int(level)) if p.id
            } if _reg_ds.by_level(0) else None
        except Exception:
            _ds_allowed = None

        # Per-worker candidate rows.
        masses: list[dict[str, float]] = []
        for r in induced_results or []:
            row: dict[str, float] = {}
            try:
                for cand in (r.differential.candidates or [])[:10]:
                    label = normalize_candidate_label(str(cand.label or "")) or str(cand.label or "").strip().lower()
                    if not label:
                        continue
                    if _ds_allowed is not None and label not in _ds_allowed:
                        continue
                    row[label] = max(row.get(label, 0.0), float(cand.score or 0.0))
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                continue
            if row:
                # Agreement proxy: top-1 confidence (sharp row → confident source).
                agreement = max(row.values())
                masses.append(mass_from_swarm_vote(row, agreement=agreement))

        if len(masses) < 2:
            return  # D-S needs ≥2 sources to be meaningful

        fused, k_conflict = combine_many(masses)
        if k_conflict >= CONFLICT_REJECT_THRESHOLD:
            logging.getLogger("rrrie-cdss").warning(
                "[DS] premise-conflict detected K=%.2f ≥ %.2f — skipping fusion at L%d",
                k_conflict, CONFLICT_REJECT_THRESHOLD, level,
            )
            # Still persist the mass for audit/UI; do NOT fold into posterior.
            existing = self.typed_case_bundle.get("diagnostic_belief")
            if isinstance(existing, DiagnosticBelief):
                belief = existing.model_copy(update={"ds_mass": fused})
                self.typed_case_bundle["diagnostic_belief"] = belief
            return

        existing = self.typed_case_bundle.get("diagnostic_belief")
        if isinstance(existing, DiagnosticBelief):
            belief = existing
        elif isinstance(existing, dict):
            try:
                belief = DiagnosticBelief.model_validate(existing)
            except Exception:
                belief = DiagnosticBelief()
        else:
            belief = DiagnosticBelief()

        post_field = {0: "family_posterior", 1: "genus_posterior", 2: "species_posterior"}[int(level)]
        bayes_post = dict(getattr(belief, post_field) or {})
        mixed = fuse_with_bayes(bayes_post, fused, kappa=0.6) if bayes_post else bayes_post

        updates: dict[str, Any] = {"ds_mass": fused}
        if mixed:
            updates[post_field] = mixed
        self.typed_case_bundle["diagnostic_belief"] = belief.model_copy(update=updates)

    # ------------------------------------------------------------------
    # W2 Module C — Coarse-to-fine hierarchical swarm
    # ------------------------------------------------------------------

    _family_specialist_map_cache: dict[str, Any] | None = None

    def _load_family_specialist_map(self) -> dict[str, Any]:
        """Load+cache data/cdss/knowledge/family_specialist_map.json."""
        cached = getattr(self.__class__, "_family_specialist_map_cache", None)
        if isinstance(cached, dict) and cached:
            return cached
        path = Path(__file__).resolve().parents[3] / "data" / "cdss" / "knowledge" / "family_specialist_map.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"default_panel": ["general", "common_case", "sherlock"], "families": {}}
        self.__class__._family_specialist_map_cache = data
        return data

    def _hierarchical_panel(self, family_id: str | None) -> list[str]:
        """Resolve worker profile panel for a given family anchor.

        Falls back to default_panel when family is unknown or no mapping.
        """
        spec = self._load_family_specialist_map()
        default = list(spec.get("default_panel", ["general", "common_case", "sherlock"]))
        if not family_id:
            return default
        fam_panel = list((spec.get("families", {}) or {}).get(str(family_id), []))
        if not fam_panel:
            return default
        # Merge: default core + family-specific heads, dedup preserving order.
        merged: list[str] = []
        for p in default + fam_panel:
            if p and p not in merged:
                merged.append(p)
        return merged[:5]  # cap fan-out to respect VRAM semaphore

    def _shared_belief_summary(self, belief: Any, *, max_per_level: int = 3) -> str:
        """Compose a compact belief-state prefix broadcast to all workers.

        Implements W2 Module C.4 shared mental model: top-k per level plus
        entropy, projected into the worker's user message.
        """
        if belief is None:
            return ""
        lines: list[str] = []
        try:
            from src.cdss.reasoning.belief_propagation import entropy as _h
        except Exception:
            _h = None  # type: ignore[assignment]

        def _topk(post: dict[str, float]) -> list[tuple[str, float]]:
            items = [(str(k), float(v)) for k, v in (post or {}).items()]
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:max_per_level]

        for level_name, post in (
            ("FAMILY", getattr(belief, "family_posterior", {}) or {}),
            ("GENUS", getattr(belief, "genus_posterior", {}) or {}),
            ("SPECIES", getattr(belief, "species_posterior", {}) or {}),
        ):
            if not post:
                continue
            top = _topk(post)
            if not top:
                continue
            pairs = ", ".join(f"{k}={v:.2f}" for k, v in top)
            if _h is not None:
                try:
                    h_val = float(_h(post))
                    lines.append(f"  {level_name} (H={h_val:.2f}): {pairs}")
                except Exception:
                    lines.append(f"  {level_name}: {pairs}")
            else:
                lines.append(f"  {level_name}: {pairs}")
        if not lines:
            return ""
        return "Diagnostic belief so far (top candidates per hierarchy level):\n" + "\n".join(lines)

    def _fold_wave_into_belief(
        self,
        *,
        induced_results: list,
        level: int,
        candidate_ids: list[str] | None = None,
    ) -> Any:
        """Per-level fold — writes family/genus/species posterior at `level`.

        Shares the aggregate-likelihood math with `_fold_swarm_into_belief`
        but targets the correct Dirichlet field via `update_posterior(level=N)`.
        Returns the updated DiagnosticBelief (also persisted to the bundle).
        """
        try:
            from src.cdss.contracts.models import DiagnosticBelief
            from src.cdss.reasoning.bayes_update import update_posterior
            from src.cdss.reasoning.belief_propagation import build_parent_maps
            from src.cdss.reasoning.likelihood_ensemble import LikelihoodEstimate
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[HSWARM] belief modules unavailable: %s", exc)
            return None

        # Strict level gate: only accept labels that resolve to a registry
        # profile at the target level. Kills:
        #   (a) family/genus labels leaking into species_posterior,
        #   (b) LLM-hallucinated symptom-concat ids that don't exist in registry.
        # Falls back to open acceptance when hierarchy asset is unshipped so
        # non-hierarchical runs keep working.
        try:
            _registry = load_syndrome_registry()
            _hierarchy_live = bool(_registry.by_level(0))
        except Exception:
            _registry = None
            _hierarchy_live = False
        _allowed: set[str] | None = None
        if _hierarchy_live and _registry is not None:
            # Build the allowed set from registry at target level.
            _allowed = {
                normalize_candidate_label(str(p.id)) for p in _registry.by_level(int(level)) if p.id
            }
            # Intersect with candidate_ids when caller constrained it.
            if candidate_ids:
                ci = {normalize_candidate_label(str(c)) for c in candidate_ids if c}
                _allowed = (_allowed & ci) if _allowed else ci

        per_hyp_samples: dict[str, list[float]] = {}
        _rejected: dict[str, int] = {}
        for r in induced_results or []:
            try:
                for cand in (r.differential.candidates or [])[:10]:
                    label = normalize_candidate_label(str(cand.label or "")) or str(cand.label or "").strip().lower()
                    if not label:
                        continue
                    if _allowed is not None and label not in _allowed:
                        # Try: is this an ancestor/descendant we can project onto level?
                        projected = None
                        original_species: str | None = None
                        if _registry is not None:
                            prof = _registry.by_id(label)
                            if prof is not None:
                                try:
                                    prof_level = int(prof.level)
                                except Exception:
                                    prof_level = -1
                                if prof_level > int(level):
                                    # Descendant — project up to ancestor at target level.
                                    # Fix-A: continue search across the full ancestor
                                    # chain rather than break on first level match,
                                    # so an ancestor that *isn't* in `_allowed`
                                    # doesn't silently shadow a deeper one that is.
                                    for anc in _registry.ancestors(label):
                                        if int(anc.level) == int(level):
                                            cand_label = normalize_candidate_label(str(anc.id))
                                            if cand_label in _allowed:
                                                projected = cand_label
                                                break
                                            # else: keep looking — same level, but
                                            # not allowed (filtered candidate set).
                                    # Preserve the original species label for the
                                    # parallel species-evidence channel below so
                                    # specific-dx votes at family/genus folds are
                                    # not lost to the projection step.
                                    if int(level) < 2 and prof_level == 2:
                                        original_species = label
                                elif prof_level < int(level):
                                    # Ancestor emitted instead of species — drop (can't invent child).
                                    projected = None
                        if projected is None:
                            _rejected[label] = _rejected.get(label, 0) + 1
                            continue
                        # Fix-A: parallel species-evidence channel. When a
                        # species label is projected up to an ancestor for the
                        # current fold, also stamp the *species* label into the
                        # bundle so downstream candidate generation can preserve
                        # specific-dx voting even when the current fold target
                        # is family or genus.
                        if original_species is not None:
                            score_for_species = max(0.0, min(1.0, float(cand.score or 0.0)))
                            buf = self.typed_case_bundle.setdefault("species_evidence_buffer", {})
                            if isinstance(buf, dict):
                                buf.setdefault(original_species, []).append(score_for_species)
                        label = projected
                    score = max(0.0, min(1.0, float(cand.score or 0.0)))
                    per_hyp_samples.setdefault(label, []).append(score)
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                continue
        if _rejected:
            top_rej = sorted(_rejected.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logging.getLogger("rrrie-cdss").info(
                "[HSWARM] L%d rejected %d off-level labels (top: %s)",
                int(level), sum(_rejected.values()), top_rej,
            )
        if not per_hyp_samples:
            return None

        likelihood_row: dict[str, LikelihoodEstimate] = {}
        for h, samples in per_hyp_samples.items():
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            var = (
                sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)
                if len(samples) >= 2 else 0.0
            )
            mean = max(1e-4, min(1.0 - 1e-4, mean))
            likelihood_row[h] = LikelihoodEstimate(
                hypothesis=h,
                finding=f"swarm_L{int(level)}",
                mean=mean,
                variance=max(0.0, var),
                alpha=mean * 3.0 + 1.0,
                beta=(1.0 - mean) * 3.0 + 1.0,
                samples=list(samples),
                grounded=True,
            )

        existing = self.typed_case_bundle.get("diagnostic_belief")
        if isinstance(existing, DiagnosticBelief):
            belief = existing
        elif isinstance(existing, dict):
            try:
                belief = DiagnosticBelief.model_validate(existing)
            except Exception:
                belief = DiagnosticBelief()
        else:
            belief = DiagnosticBelief()

        # Seed α uniformly across candidates on first touch of this level.
        alpha_field = {0: "family_alpha", 1: "genus_alpha", 2: "species_alpha"}[int(level)]
        if not getattr(belief, alpha_field):
            belief = belief.model_copy(update={alpha_field: {h: 1.0 for h in likelihood_row.keys()}})

        try:
            registry = load_syndrome_registry()
            parent_of_genus, parent_of_species = build_parent_maps(registry.profiles)
        except Exception:
            parent_of_genus, parent_of_species = {}, {}

        try:
            updated = update_posterior(
                belief,
                level=int(level),
                likelihoods={f"swarm_L{int(level)}": likelihood_row},
                parent_of_genus=parent_of_genus,
                parent_of_species=parent_of_species,
                source=f"hswarm_L{int(level)}",
            )
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[HSWARM] update_posterior L%d failed: %s", level, exc)
            return belief

        self.typed_case_bundle["diagnostic_belief"] = updated
        return updated

    # -----------------------------------------------------------------
    # W4 Module F — causal-do mechanism verifier integration
    # -----------------------------------------------------------------

    _pathway_edges_cache: list[dict[str, Any]] | None = None

    def _load_pathway_edges(self) -> list[dict[str, Any]]:
        """Load+cache data/cdss/knowledge/pathway_edges.json (seed ~200 edges)."""
        cached = getattr(self.__class__, "_pathway_edges_cache", None)
        if isinstance(cached, list):
            return cached
        path = Path(__file__).resolve().parents[3] / "data" / "cdss" / "knowledge" / "pathway_edges.json"
        edges: list[dict[str, Any]] = []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            edges = list(data.get("edges", []) or [])
        except Exception:
            edges = []
        self.__class__._pathway_edges_cache = edges
        return edges

    def _collect_findings_keys(self) -> list[str]:
        """Best-effort extraction of normalized finding tokens for graph reachability.

        Pulls from StructuredFindings.positive_findings / critical_findings and
        FactGraph fact labels, normalized with the same case-folding used in
        mechanism frames. Returns a deduped list (order preserved by first
        occurrence).
        """
        out: list[str] = []
        seen: set[str] = set()

        def _add(tok: Any) -> None:
            s = str(tok or "").strip().lower().replace(" ", "_")
            if not s or s in seen:
                return
            seen.add(s)
            out.append(s)

        try:
            findings = self._ensure_typed_findings()
            for src_list in (
                getattr(findings, "positive_findings", None) or [],
                getattr(findings, "critical_findings", None) or [],
                getattr(findings, "negative_findings", None) or [],
            ):
                for tok in src_list:
                    _add(tok)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass
        try:
            fg = self._ensure_typed_fact_graph()
            for fact in getattr(fg, "facts", None) or []:
                _add(getattr(fact, "label", "") or getattr(fact, "text", ""))
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass
        return out[:80]

    def _fold_causal_do_into_belief(
        self,
        *,
        mechanism_frames: list[MechanismFrame],
        top_k: int = 5,
    ) -> Any:
        """Build MechanismGraph, run do-probes per top-k species, fold robustness.

        Robustness score ∈ [0, 1] per hypothesis becomes a pseudo-likelihood
        against the synthetic finding `causal_do`. Feeds `update_posterior` at
        species level (level=2) so the Bayesian layer sees a downweight when a
        mechanism cannot explain the observed findings without brittle reliance
        on a single edge.
        """
        try:
            from src.cdss.contracts.models import DiagnosticBelief
            from src.cdss.reasoning.bayes_update import update_posterior
            from src.cdss.reasoning.belief_propagation import build_parent_maps
            from src.cdss.reasoning.causal_do import build_graph_from_edges, robustness_score, explain_coverage
            from src.cdss.reasoning.likelihood_ensemble import LikelihoodEstimate
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[CAUSAL_DO] modules unavailable: %s", exc)
            return None

        existing = self.typed_case_bundle.get("diagnostic_belief")
        if isinstance(existing, DiagnosticBelief):
            belief = existing
        elif isinstance(existing, dict):
            try:
                belief = DiagnosticBelief.model_validate(existing)
            except Exception:
                return None
        else:
            return None

        species_post = dict(belief.species_posterior or {})
        if not species_post:
            return belief

        top_items = sorted(species_post.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
        top_hyps = [h for h, _ in top_items]

        curated = self._load_pathway_edges()
        frame_edges: list[dict[str, Any]] = []
        for fr in mechanism_frames or []:
            for e in getattr(fr, "causal_edges", None) or []:
                try:
                    frame_edges.append({
                        "from": e.from_node,
                        "to": e.to_node,
                        "weight": float(getattr(e, "weight", 1.0)),
                        "pathway_ref": getattr(e, "pathway_ref", ""),
                    })
                except Exception:  # noqa: BLE001
                    logger.debug("swallowed exception", exc_info=True)
                    continue
        all_edges = list(curated) + frame_edges
        if not all_edges:
            return belief

        findings_keys = self._collect_findings_keys()
        if not findings_keys:
            return belief

        graph = build_graph_from_edges(
            hypotheses=top_hyps,
            findings=findings_keys,
            edges=all_edges,
        )

        # Per-hyp robustness + coverage in one pass (no LLM, pure BFS).
        likelihood_row: dict[str, LikelihoodEstimate] = {}
        probes: list[dict[str, Any]] = []
        for h in top_hyps:
            try:
                cov = float(explain_coverage(graph, h, findings_keys))
                rob = float(robustness_score(graph, h, findings_keys))
            except Exception:
                cov, rob = 0.0, 0.0
            # Blend: coverage = does mechanism explain the findings; robustness =
            # does explanation survive ablation. Product penalizes brittle stories.
            score = max(0.05, min(0.95, 0.5 * cov + 0.5 * rob))
            likelihood_row[h] = LikelihoodEstimate(
                hypothesis=h,
                finding="causal_do",
                mean=score,
                variance=0.04,
                alpha=score * 4.0 + 1.0,
                beta=(1.0 - score) * 4.0 + 1.0,
                samples=[score],
                grounded=True,
            )
            probes.append({"hypothesis": h, "coverage": round(cov, 4), "robustness": round(rob, 4)})

        self.typed_case_bundle["causal_do_probes"] = probes

        try:
            registry = load_syndrome_registry()
            parent_of_genus, parent_of_species = build_parent_maps(registry.profiles)
        except Exception:
            parent_of_genus, parent_of_species = {}, {}

        try:
            updated = update_posterior(
                belief,
                level=2,
                likelihoods={"causal_do": likelihood_row},
                parent_of_genus=parent_of_genus,
                parent_of_species=parent_of_species,
                source="causal_do_verifier",
            )
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[CAUSAL_DO] update_posterior failed: %s", exc)
            return belief

        self.typed_case_bundle["diagnostic_belief"] = updated
        return updated

    async def _run_hierarchical_swarm(
        self,
        *,
        findings,
        risk_profile,
        fact_graph,
        evidence,
    ) -> list:
        """Coarse-to-fine swarm: L0 family → L1 genus → L2 species (W2 Module C).

        Returns the list of GatewayDifferentialResult from the *final* (L2) wave,
        so the existing merge/prototype/evidence flow in `_run_expert_swarm`
        keeps working unmodified. Side-effect: updates `DiagnosticBelief` in the
        typed bundle at every level.

        Early terminates to L2 when family posterior is sharp
        (max > 0.9 and H < 0.3).
        """
        from src.cdss.reasoning.belief_propagation import entropy as _h
        try:
            registry = load_syndrome_registry()
        except Exception:
            return []
        families = registry.by_level(0)
        if not families:
            return []

        _swarm_limit = int(os.getenv("CDSS_SWARM_CONCURRENCY_LIMIT", "4"))
        sem = asyncio.Semaphore(_swarm_limit)

        # W6.3 J.4 / J.5 policy snapshot — used inside _run_panel.
        try:
            _two_phase_on = bool(getattr(self.runtime_policy, "swarm_two_phase_broadcast_enabled", False))
            _round2_entropy = float(getattr(self.runtime_policy, "two_phase_round2_entropy_threshold", 0.5))
            _early_drop_on = bool(getattr(self.runtime_policy, "swarm_early_dropout_enabled", False))
            _early_H = float(getattr(self.runtime_policy, "swarm_early_stop_entropy", 0.4))
            _early_top1 = float(getattr(self.runtime_policy, "swarm_early_stop_top1_mass", 0.7))
            _early_min = int(getattr(self.runtime_policy, "swarm_early_stop_min_workers", 3))
            _ens_on_h = bool(getattr(self.runtime_policy, "swarm_temperature_ensemble_enabled", False))
        except Exception:
            _two_phase_on = False
            _round2_entropy = 0.5
            _early_drop_on = False
            _early_H = 0.4
            _early_top1 = 0.7
            _early_min = 3
            _ens_on_h = False

        async def _run_panel(level: int, candidate_set: list[str], panel: list[str], shared: str) -> list:
            async def _worker(profile: str, override_shared: str):
                worker_deep_thinking = self.deep_thinking if profile == "general" else False
                async with sem:
                    return await self._gateway.induce_differential(
                        findings, risk_profile, fact_graph,
                        evidence=evidence,
                        focus_profile=profile,
                        deep_thinking=worker_deep_thinking,
                        hierarchy_level=level,
                        candidate_set=candidate_set,
                        shared_belief=override_shared,
                        temperature_ensemble=_ens_on_h,
                    )

            # W6.3 J.4 — Two-phase broadcast.
            # Round 1: every worker runs with NO belief prefix (truly
            # independent likelihood estimators — repairs the Bayes-independence
            # violation that artificially flattens DS conflict K).
            # Round 2: only workers whose Round-1 top-1 disagrees with the
            # cross-worker mode get re-fired *with* the belief prefix.
            if _two_phase_on:
                # Round 1 (independent).
                round1 = await asyncio.gather(*[_worker(p, "") for p in panel], return_exceptions=False)
                # Compute Round-1 top-1 distribution and entropy.
                top1_counts: dict[str, int] = {}
                top1_per_idx: list[str] = []
                for r in round1:
                    lab = ""
                    try:
                        if r and r.differential and r.differential.candidates:
                            lab = normalize_candidate_label(r.differential.candidates[0].label) or ""
                    except Exception:
                        lab = ""
                    top1_per_idx.append(lab)
                    if lab:
                        top1_counts[lab] = top1_counts.get(lab, 0) + 1
                if top1_counts:
                    n = sum(top1_counts.values())
                    p_top = max(top1_counts.values()) / n
                    H = 0.0
                    for c in top1_counts.values():
                        p = c / n
                        if p > 0:
                            H -= p * math.log2(p)
                else:
                    p_top, H = 0.0, 0.0

                # Skip round 2 on clear consensus (entropy below threshold AND
                # top-1 mass dominates). Otherwise refine dissenters only.
                if H < _round2_entropy or not top1_counts or not shared:
                    if H < _round2_entropy:
                        logging.getLogger("rrrie-cdss").info(
                            "[J4] Round-2 skipped at L%d (H=%.2f < %.2f, top1_mass=%.2f) — Round 1 consensus.",
                            level, H, _round2_entropy, p_top,
                        )
                    return list(round1)

                mode_label = max(top1_counts.items(), key=lambda kv: kv[1])[0]
                dissent_idx = [i for i, lab in enumerate(top1_per_idx) if lab and lab != mode_label]
                if not dissent_idx:
                    return list(round1)
                dissent_panel = [panel[i] for i in dissent_idx]
                logging.getLogger("rrrie-cdss").info(
                    "[J4] Round 2 at L%d (H=%.2f) — refining %d dissenters out of %d (mode=%s).",
                    level, H, len(dissent_idx), len(panel), mode_label,
                )
                round2 = await asyncio.gather(*[_worker(p, shared) for p in dissent_panel], return_exceptions=False)
                refined: list = list(round1)
                for slot, refined_r in zip(dissent_idx, round2):
                    refined[slot] = refined_r
                return refined

            # Default single-phase: every worker gets the shared belief prefix.
            tasks = [_worker(p, shared) for p in panel]

            if not _early_drop_on:
                return await asyncio.gather(*tasks, return_exceptions=False)

            # W6.3 J.5 — SID-style early worker dropout.
            # Once `_early_min` workers complete, compute partial RRF + entropy
            # over their top-3s. If consensus is sharp, cancel the rest.
            from src.cdss.reasoning.rank_fusion import reciprocal_rank_fusion as _rrf
            futures = [asyncio.ensure_future(t) for t in tasks]
            done_results: list = [None] * len(futures)
            completed = 0
            try:
                while completed < len(futures):
                    done_set, _ = await asyncio.wait(
                        [f for f in futures if not f.done()],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for f in done_set:
                        try:
                            idx = futures.index(f)
                        except ValueError:
                            continue
                        try:
                            done_results[idx] = f.result()
                        except Exception:
                            done_results[idx] = None
                        completed += 1
                    if completed < _early_min or completed >= len(futures):
                        continue
                    # Build partial rankings and check sharpness.
                    rankings: list[list[str]] = []
                    for r in done_results:
                        if r is None:
                            continue
                        try:
                            ids = []
                            for cand in (r.differential.candidates or [])[:3]:
                                nid = normalize_candidate_label(cand.label) or ""
                                if nid:
                                    ids.append(nid)
                            if ids:
                                rankings.append(ids)
                        except Exception:  # noqa: BLE001
                            logger.debug("swallowed exception", exc_info=True)
                            continue
                    if not rankings:
                        continue
                    rrf_partial = _rrf(rankings, k=60, normalize=True)
                    if not rrf_partial:
                        continue
                    total = sum(rrf_partial.values())
                    if total <= 0:
                        continue
                    p_norm = [v / total for v in rrf_partial.values()]
                    H_partial = 0.0
                    for p in p_norm:
                        if p > 0:
                            H_partial -= p * math.log2(p)
                    top1_mass = max(p_norm)
                    if H_partial < _early_H and top1_mass > _early_top1:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        logging.getLogger("rrrie-cdss").info(
                            "[J5] Early dropout L%d (H=%.2f<%.2f, top1=%.2f>%.2f) — %d/%d done.",
                            level, H_partial, _early_H, top1_mass, _early_top1, completed, len(futures),
                        )
                        break
            finally:
                # Drain any cancelled futures.
                for f in futures:
                    if not f.done():
                        try:
                            await f
                        except Exception:  # noqa: BLE001
                            logger.debug("swallowed exception", exc_info=True)
                            pass
            # Filter out None / cancelled.
            return [r for r in done_results if r is not None]

        # ---- L0 wave: identify family ------------------------------
        l0_candidates = [f.id for f in families]
        l0_panel = self._hierarchical_panel(None)
        belief = self.typed_case_bundle.get("diagnostic_belief")
        shared_prefix = self._shared_belief_summary(belief)
        try:
            l0_results = await _run_panel(0, l0_candidates, l0_panel, shared_prefix)
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[HSWARM] L0 failed: %s", exc)
            l0_results = []
        if l0_results:
            belief = self._fold_wave_into_belief(induced_results=l0_results, level=0, candidate_ids=l0_candidates)

        # ---- Early termination on confident family (C.3) ----------
        fam_post = dict(getattr(belief, "family_posterior", {}) or {}) if belief else {}
        skip_l1 = False
        if fam_post:
            max_f = max(fam_post.values())
            h_f = float(_h(fam_post))
            if max_f > 0.9 and h_f < 0.3:
                skip_l1 = True
                logging.getLogger("rrrie-cdss").info(
                    "[HSWARM] L0 confident (max=%.2f H=%.2f) — skipping L1 to L2", max_f, h_f
                )

        # Top-1 family drives specialist panel from here on.
        top_family = max(fam_post.items(), key=lambda x: x[1])[0] if fam_post else None
        specialist_panel = self._hierarchical_panel(top_family)

        # ---- L1 wave (unless skipped): genus, conditioned on top-5 families ----
        if not skip_l1:
            top_k_fams = [fid for fid, _p in sorted(fam_post.items(), key=lambda x: x[1], reverse=True)[:5]]
            l1_set: list[str] = []
            for fid in top_k_fams:
                for ch in registry.descendants(fid, level=1):
                    if ch.id and ch.id not in l1_set:
                        l1_set.append(ch.id)
            if l1_set:
                shared_prefix = self._shared_belief_summary(belief)
                try:
                    l1_results = await _run_panel(1, l1_set, specialist_panel, shared_prefix)
                except Exception as exc:
                    logging.getLogger("rrrie-cdss").warning("[HSWARM] L1 failed: %s", exc)
                    l1_results = []
                if l1_results:
                    belief = self._fold_wave_into_belief(induced_results=l1_results, level=1, candidate_ids=l1_set)

        # ---- L2 wave: species, conditioned on top-8 genera (or direct from family if no genera) ----
        gen_post = dict(getattr(belief, "genus_posterior", {}) or {}) if belief else {}
        top_genera = [gid for gid, _p in sorted(gen_post.items(), key=lambda x: x[1], reverse=True)[:8]]
        l2_set: list[str] = []
        if top_genera:
            for gid in top_genera:
                for ch in registry.descendants(gid, level=2):
                    if ch.id and ch.id not in l2_set:
                        l2_set.append(ch.id)
        if not l2_set:
            # No genus layer populated — fall back to descendants(top_family, level=2).
            top_k_fams = [fid for fid, _p in sorted(fam_post.items(), key=lambda x: x[1], reverse=True)[:3]]
            for fid in top_k_fams:
                for ch in registry.descendants(fid, level=2):
                    if ch.id and ch.id not in l2_set:
                        l2_set.append(ch.id)

        shared_prefix = self._shared_belief_summary(belief)
        try:
            l2_results = await _run_panel(2, l2_set, specialist_panel, shared_prefix)
            # W6.1 J.1 — publish the L2 panel so the flat-path consumer in
            # `_run_expert_swarm` can align rank-fusion profiles to results.
            try:
                self.typed_case_bundle["last_swarm_panel"] = list(specialist_panel or [])
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[HSWARM] L2 failed: %s", exc)
            l2_results = []
        if l2_results:
            self._fold_wave_into_belief(induced_results=l2_results, level=2, candidate_ids=l2_set)
            # W2 Module E — Dempster-Shafer fusion over L2 agent votes.
            try:
                self._fold_ds_into_belief(induced_results=l2_results, level=2)
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[DS] L2 fold failed: %s", exc)

        return l2_results

    async def _run_expert_swarm(self):
        """
        Phase 1 of the Futuristic Vision: Mixture of Agents (MoA).
        Instead of 1 prompt, we fire off parallel LLM requests simulating different medical specialties.
        """
        print(f"[{time.strftime('%H:%M:%S')}] [Swarm Panel] Launching adaptive specialty panel...")
        await self._emit_trace("hypothesis_generation", "Launching adaptive swarm panel over shared typed state...", {})

        stage_started = time.perf_counter()
        self._set_profile_stage("differential")
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        risk_profile = self._typed_risk_profile()
        evidence = self._typed_evidence()
        prototype_matches = self._stable_prototype_matches()
        await self._emit_trace(
            "hypothesis_generation",
            "Adaptive swarm panel focus derived from prototype memory and typed evidence.",
            {"prototype_matches": [str(match.label or "") for match in prototype_matches[:3]]},
        )
        aggregated_hypotheses: dict[str, float] = {}
        provenance: dict[str, list[str]] = {}

        # 5. Parallel Hypothesis Cross-Examination with Concurrency Protection
        # Fetch domains prioritized from planning
        case_domains = self.typed_case_bundle.get("domains", [])
        specialties = []
        for d in case_domains:
            token = str(d).strip().upper()
            if token and token not in specialties:
                specialties.append(token)

        # Core diagnostic panel. ``red_team_fatal`` is mandatory on EVERY case
        # regardless of urgency — anchoring (and therefore mis-triage) happens
        # upstream of urgency assignment, so suppressing it on routine tier
        # silently drops the only worker that ignores patient self-framing.
        # See plan item 3 (cross-case engineering improvements 2026-04-26).
        _speed_profile = str(getattr(self.runtime_policy, "runtime_speed_profile", "") or "").strip().lower()
        _fast_profile = _speed_profile == "fast"
        profiles_to_run = (
            ["general", "common_case", "red_team_fatal"]
            if _fast_profile
            else ["general", "common_case", "sherlock", "red_team_fatal"]
        )

        # zebra_hunter remains acuity-gated — its role is rare-disease
        # enumeration, not adversarial de-anchoring.
        _is_emergency = risk_profile.urgency == UrgencyTier.EMERGENCY
        _is_urgent = risk_profile.urgency == UrgencyTier.URGENT
        if _is_emergency or _is_urgent:
            profiles_to_run.append("zebra_hunter")

        # Dynamic Polypharmacy / Medication Analysis
        # Autonomously construct a targeted pharmacologist role based on actual medications, avoiding static strings.
        if findings.medications and (not _fast_profile or _is_emergency or _is_urgent):
            sanitized_meds = [m.replace(" ", "_").upper() for m in findings.medications[:4]]
            dynamic_pharmacology_profile = f"CLINICS_AND_TOXICOLOGY_ON_{'_AND_'.join(sanitized_meds)}"
            profiles_to_run.append(f"specialist:{dynamic_pharmacology_profile}")

        # Map inferred domains to broad specialty experts dynamically.
        specialty_limit = 1 if _fast_profile else 2
        for sp in specialties[:specialty_limit]:
            # Quick sanitization to avoid hyper-specific anchoring
            sp_clean = sp.replace("_", " ")
            if len(sp_clean.split()) <= 2 and (not _fast_profile or _is_emergency or _is_urgent):  # Broad categories usually short
                profiles_to_run.append(f"specialist:{sp_clean}")

        # Hardware-protection concurrency limit (KV Cache memory saturation prevention)
        # Configurable via settings.swarm_concurrency_limit (default 4; increase for >6GB VRAM)
        _swarm_limit = int(os.getenv("CDSS_SWARM_CONCURRENCY_LIMIT", "3" if _fast_profile else "4"))
        sem = asyncio.Semaphore(_swarm_limit)

        # W2 Module C — coarse-to-fine hierarchical swarm. Runs 3 waves
        # (family → genus → species) when `hierarchical_swarm_enabled` is set.
        # On success its L2 wave results replace the flat panel below; on
        # failure we transparently fall back.
        induced_results = None
        _hsw_on = False
        try:
            _pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if _pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                _pol = load_runtime_policy()
            _hsw_on = bool(getattr(_pol, "hierarchical_swarm_enabled", False))
        except Exception:
            _hsw_on = False
        if _hsw_on:
            try:
                hsw_results = await self._run_hierarchical_swarm(
                    findings=findings,
                    risk_profile=risk_profile,
                    fact_graph=fact_graph,
                    evidence=evidence,
                )
                if hsw_results:
                    induced_results = hsw_results
                    await self._emit_trace(
                        "hypothesis_generation",
                        "Hierarchical coarse-to-fine swarm (L0→L1→L2) completed.",
                        {"waves": 3, "profiles_used": profiles_to_run},
                    )
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[HSWARM] fell back to flat panel: %s", exc)
                induced_results = None

        # W6.2 J.2 — per-worker temperature ensemble. When enabled, each worker
        # is fired n times in parallel at distinct temperatures and merged
        # into mean ± variance per candidate (Beta posterior via method-of-moments).
        try:
            _ens_on = bool(getattr(self.runtime_policy, "swarm_temperature_ensemble_enabled", False))
        except Exception:
            _ens_on = False

        async def _run_worker(profile: str):
            # Only the Master ("general") uses deep thinking, and only if ATOM mode (self.deep_thinking) is active.
            # All other specialist/hunter workers run in fast zero-shot mode regardless of ATOM mode.
            worker_deep_thinking = self.deep_thinking if profile == "general" else False
            async with sem:
                return await self._gateway.induce_differential(
                    findings, risk_profile, fact_graph, evidence=evidence,
                    focus_profile=profile, deep_thinking=worker_deep_thinking,
                    temperature_ensemble=_ens_on,
                )

        # Track which profile produced each induced_results[i] so rank-fusion
        # (W6.1 J.1) can pull per-worker trust + hierarchical weights. Set in
        # both paths so the alignment never drifts.
        _active_panel: list[str] = []
        if induced_results is None:
            tasks = [_run_worker(p) for p in profiles_to_run]
            induced_results = await asyncio.gather(*tasks)
            _active_panel = list(profiles_to_run)
        else:
            # Hierarchical path: panel is the specialist L2 panel; recovered
            # from the bundle if `_run_hierarchical_swarm` published it,
            # otherwise leave empty (fusion falls back to MAX for safety).
            _active_panel = list(self.typed_case_bundle.get("last_swarm_panel", []) or [])

        # W6.2 J.3 — publish per-profile top-3 ids so worker_trust can score
        # each profile against the verifier's committed top-1 at close. Stored
        # as `{profile: [normalized_id, ...]}`. Cheap; ~N×3 strings.
        try:
            _per_profile_top3: dict[str, list[str]] = {}
            for prof, r in zip(_active_panel, induced_results):
                if not prof or not getattr(r, "differential", None):
                    continue
                ids = []
                for cand in r.differential.candidates[:3]:
                    nid = normalize_candidate_label(cand.label) or str(cand.label or "").strip()
                    if nid:
                        ids.append(nid)
                if ids:
                    _per_profile_top3[str(prof)] = ids
            if _per_profile_top3:
                self.typed_case_bundle["last_swarm_per_profile_top3"] = _per_profile_top3
                self.typed_case_bundle["last_swarm_panel"] = list(_active_panel)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        # Merge all metrics and errors
        merged_metrics = {}
        merged_error = {}
        for r in induced_results:
            if r.metrics:
                merged_metrics.update(r.metrics)
            if r.error:
                merged_error.update(r.error)

        self._record_gateway_metrics("differential", merged_metrics)
        if merged_error:
            self.typed_case_bundle["differential_error"] = dict(merged_error)
            await self._emit_trace(
                "hypothesis_generation",
                "Differential inducer returned no stable structured output; suppressing weak autonomous closure and keeping the case open.",
                {"error": merged_error, "metrics": self._stage_metrics("differential")},
            )

        # Baseline induction is the general result
        induced = induced_results[0]

        self.typed_case_bundle.update(
            {
                "differential": induced.differential,
                "frontier": induced.frontier,
                "disease_hypotheses": induced.disease_hypotheses,
            }
        )

        # Merge candidates from all swarm tracks.
        # W6.1 J.1 — when `swarm_rank_fusion_enabled`, replace per-candidate
        # `max(score)` with Reciprocal-Rank-Fusion + Trust-Weighted Borda
        # convex mixture (rank_fusion.aggregate_worker_outputs). Otherwise
        # keep legacy MAX as before (back-compat).
        _rf_on = False
        try:
            _rf_pol = self.runtime_policy
            _rf_on = bool(getattr(_rf_pol, "swarm_rank_fusion_enabled", False)) and bool(_active_panel)
        except Exception:
            _rf_on = False

        if _rf_on:
            try:
                from src.cdss.reasoning.rank_fusion import aggregate_worker_outputs
                worker_results: list[list[tuple[str, float]]] = []
                for r in induced_results:
                    items: list[tuple[str, float]] = []
                    for candidate in r.differential.candidates[:8]:
                        normalized = normalize_candidate_label(candidate.label) or str(candidate.label or "").strip()
                        if not normalized:
                            continue
                        items.append((normalized, float(candidate.score or 0.0)))
                    worker_results.append(items)
                # Per-(profile,family) trust + family-specialist hierarchical
                # boost are J.3 (worker_reliability.json) — until shipped, the
                # rank_fusion module uses cold-start defaults (τ=0.8, w_hier=1.0).
                _trust_map: dict[str, float] | None = None
                _hier_map: dict[str, float] | None = None
                try:
                    from src.cdss.reasoning.worker_trust import load_worker_trust  # type: ignore
                    _top_fam = ""
                    try:
                        _belief = self.typed_case_bundle.get("diagnostic_belief")
                        if _belief is not None and getattr(_belief, "family_posterior", None):
                            _top_fam = max(_belief.family_posterior.items(), key=lambda kv: kv[1])[0]
                    except Exception:
                        _top_fam = ""
                    _trust_map = load_worker_trust(family=_top_fam) if _top_fam else None
                except Exception:
                    _trust_map = None
                fused = aggregate_worker_outputs(
                    worker_results=worker_results,
                    profiles=_active_panel,
                    trust=_trust_map,
                    hier_weights=_hier_map,
                    alpha=float(getattr(_rf_pol, "rank_fusion_alpha", 0.4)),
                    beta=float(getattr(_rf_pol, "rank_fusion_beta", 0.4)),
                    gamma=float(getattr(_rf_pol, "rank_fusion_gamma", 0.2)),
                    rrf_k=int(getattr(_rf_pol, "rank_fusion_rrf_k", 60)),
                )
                # Plan Item 4 — multiplicative miss-cost prior on the fused
                # posterior. Pure posterior ranking ignores asymmetric harm of
                # missing a deadly dx; this lifts severity-weighted candidates
                # one tier without altering equal-severity ordering.
                if bool(getattr(_rf_pol, "severity_rank_enabled", True)) and float(getattr(_rf_pol, "severity_rank_weight", 0.0)) > 0:
                    try:
                        from src.cdss.reasoning.rank_fusion import severity_weighted as _sev_rerank
                        from src.cdss.reasoning.bma_utility import load_mortality_priors as _load_priors
                        _priors = _load_priors()
                        _severity_map: dict[str, float] = {
                            label: float(getattr(row, "harm_if_untreated", 0.3) or 0.3)
                            for label, row in _priors.items()
                        }
                        if _severity_map:
                            fused = _sev_rerank(
                                fused,
                                _severity_map,
                                delta=float(getattr(_rf_pol, "severity_rank_weight", 0.25)),
                            )
                    except Exception as _sev_exc:
                        logging.getLogger("rrrie-cdss").warning(
                            "[SEVERITY_RERANK] skipped (%s)", _sev_exc
                        )
                # Plan Item 7 — anchoring detector. If the top-1 fused
                # candidate verbatim-echoes a patient-narrative phrase, apply
                # `anchor_penalty` so the rank reflects objective findings
                # rather than the patient's self-diagnosis.
                if bool(getattr(_rf_pol, "anchor_detector_enabled", True)) and fused:
                    try:
                        from src.cdss.clinical.anchoring import detect_anchoring as _detect_anchor
                        _top_label = max(fused.items(), key=lambda kv: kv[1])[0]
                        _anchor_rep = _detect_anchor(
                            findings,
                            _top_label,
                            cosine_threshold=float(getattr(_rf_pol, "anchor_cosine_threshold", 0.78)),
                        )
                        self.typed_case_bundle["anchoring_report"] = _anchor_rep
                        if _anchor_rep.is_anchored:
                            penalty = float(getattr(_rf_pol, "anchor_penalty", 0.7))
                            penalty = max(0.0, min(1.0, penalty))
                            # 3d: proactive anchoring — when the alternative is
                            # close (top1 - top2 < 0.15) the anchored top is
                            # treated as much weaker, so the penalty doubles
                            # toward 0 (clipped at penalty/2). Prevents a thin
                            # margin lock-in on a self-diagnosis echo.
                            close_race = False
                            try:
                                _sorted = sorted(fused.values(), reverse=True)
                                if len(_sorted) >= 2:
                                    close_race = (float(_sorted[0]) - float(_sorted[1])) < 0.15
                            except Exception:
                                close_race = False
                            applied_penalty = penalty / 2.0 if close_race else penalty
                            fused[_top_label] = float(fused[_top_label]) * applied_penalty
                            await self._emit_trace(
                                "ie_override",
                                f"Anchoring detected on top candidate '{_top_label}' "
                                f"(cosine={_anchor_rep.score:.2f}); fused score downweighted by ×{applied_penalty:.2f}"
                                + (" [close-race adaptive]" if close_race else ""),
                                {
                                    "top_label": _top_label,
                                    "anchor_score": _anchor_rep.score,
                                    "matched_phrase": _anchor_rep.matched_phrase,
                                    "penalty": applied_penalty,
                                    "close_race": close_race,
                                },
                            )
                    except Exception as _anchor_exc:
                        logging.getLogger("rrrie-cdss").warning(
                            "[ANCHOR_DETECTOR] skipped (%s)", _anchor_exc
                        )
                for normalized, score in fused.items():
                    aggregated_hypotheses[normalized] = max(
                        aggregated_hypotheses.get(normalized, 0.0), float(score)
                    )
                    provenance.setdefault(normalized, []).append("typed_gateway:rrf_twb")
            except Exception as _rf_exc:
                logging.getLogger("rrrie-cdss").warning(
                    "[RANK_FUSION] aggregate_worker_outputs raised, falling back to MAX: %s", _rf_exc
                )
                _rf_on = False

        if not _rf_on:
            for r in induced_results:
                for candidate in r.differential.candidates[:5]:
                    normalized = normalize_candidate_label(candidate.label) or str(candidate.label or "").strip()
                    if not normalized:
                        continue
                    # Take highest score across tracks
                    aggregated_hypotheses[normalized] = max(aggregated_hypotheses.get(normalized, 0.0), float(candidate.score or 0.0))
                    provenance.setdefault(normalized, []).append("typed_gateway")

        for match in prototype_matches[:4]:
            normalized = normalize_candidate_label(str(match.label or "").strip()) or str(match.label or "").strip()
            if not normalized:
                continue
            aggregated_hypotheses[normalized] = max(aggregated_hypotheses.get(normalized, 0.0), float(match.similarity or 0.0))
            provenance.setdefault(normalized, []).append("prototype_memory")
        for item in evidence.items[:12]:
            relation = str(item.relation_type or "").strip().lower()
            if relation not in {"support", "supports", "discriminator", "retrieved"}:
                continue
            linked_labels = [
                normalize_candidate_label(str(label or "").strip()) or str(label or "").strip()
                for label in list(item.linked_hypotheses or [])[:4]
            ]
            linked_labels = [label for label in linked_labels if label]
            if not linked_labels:
                continue
            support_score = 0.18 + min(0.26, float(item.trust_score or 0.0) * 0.5)
            if str(item.verification_status or "").strip().lower() == "verified":
                support_score += 0.12
            if str(item.source or "").strip().lower() == "memory":
                support_score += 0.08
            for normalized in linked_labels:
                aggregated_hypotheses[normalized] = max(aggregated_hypotheses.get(normalized, 0.0), round(min(0.82, support_score), 2))
                provenance.setdefault(normalized, []).append(f"evidence:{str(item.source or 'evidence').strip().lower()}")
        total = sum(max(score, 0.01) for score in aggregated_hypotheses.values())
        entropy = -sum((max(score, 0.01) / total) * math.log2(max(score, 0.01) / total) for score in aggregated_hypotheses.values()) if total > 0 else 0

        # Coarse-to-fine Bayesian belief fold (W1 Module B.3). No-op unless
        # `bayesian_posterior_enabled` is on. Projects swarm outputs onto
        # DiagnosticBelief.species_posterior so downstream modules (MCTS,
        # Dempster-Shafer, conformal) can consume typed state.
        try:
            self._fold_swarm_into_belief(
                induced_results=induced_results,
                aggregated_hypotheses=aggregated_hypotheses,
            )
        except Exception as _belief_exc:
            logging.getLogger("rrrie-cdss").warning(
                "[BELIEF] fold_swarm_into_belief raised: %s", _belief_exc
            )

        # W2 Module E — D-S fusion (flat path also benefits when flag on).
        try:
            self._fold_ds_into_belief(induced_results=induced_results, level=2)
        except Exception as _ds_exc:
            logging.getLogger("rrrie-cdss").warning(
                "[DS] fold_ds_into_belief raised: %s", _ds_exc
            )

        if aggregated_hypotheses:
            reranked = self._apply_adaptive_rerank(aggregated_hypotheses, provenance)
            self.state.working_hypotheses = reranked[:3] or [label for label, _ in sorted(aggregated_hypotheses.items(), key=lambda item: item[1], reverse=True)[:3]]
        else:
            self.state.working_hypotheses = ["undifferentiated_critical_process"]

        if (not induced.error) and (not self.state.working_hypotheses or all(self._is_generic_process_label(label) for label in self.state.working_hypotheses)):
            await self._recover_specific_differential("swarm_generic_recovery")

        # Epidemiological calibration: if a catastrophic diagnosis is top-1 for a non-emergency case,
        # ask LLM + internet to re-rank using base-rate reasoning.
        if risk_profile.urgency != UrgencyTier.EMERGENCY:
            await self._run_differential_calibrator(risk_profile)

        # Epi/spec calibration now runs in parallel with BACKWARD_SIMULATION (Fix 3).
        # Both read from typed_case_bundle["differential"] and write to disjoint keys,
        # so they are fully independent. The gather is in _run_backward_causality_simulation.

        # BUG-03 FIX: Do NOT run simulation here — it runs again in OUTCOME_SIMULATION phase.
        # P8: When divergence is high, actively surface the most specific non-generic hypothesis.
        if entropy > 1.0:
            print(f"Epistemic uncertainty detected (Entropy: {entropy:.2f}). Flagging for deeper simulator evaluation.")
            await self._emit_trace("hypothesis_generation", f"High epistemic divergence (H={entropy:.2f}); mortality simulator will run at full depth.", {"entropy": entropy})
            # Promote specific hypotheses to top of working list so downstream stages can focus
            specific = [h for h in self.state.working_hypotheses if not self._is_generic_process_label(h)]
            generic = [h for h in self.state.working_hypotheses if self._is_generic_process_label(h)]
            if specific:
                self.state.working_hypotheses = specific + generic
                # Also boost their scores in aggregated_hypotheses so rerank picks them up
                for label in specific:
                    if label in aggregated_hypotheses:
                        aggregated_hypotheses[label] = min(0.95, aggregated_hypotheses[label] + 0.12)

        # Entropy actuator: derive case complexity and promote to ATOM mode when critical.
        # This makes high-entropy cases self-escalate without user intervention.
        self._case_complexity = _compute_case_complexity(entropy, risk_profile)
        if self._case_complexity == "critical" and not self.deep_thinking:
            self.deep_thinking = True
            await self._emit_trace(
                "hypothesis_generation",
                f"Auto-promoted to deep reasoning mode (complexity=critical, H={entropy:.2f}).",
                {"entropy": entropy, "complexity": self._case_complexity},
            )

        for idx, d in enumerate(self.state.working_hypotheses):
            self.state.fact_graph.nodes.append(EvidenceNode(
                id=f"d_{idx}", type="diagnosis", label=d, confidence=0.8, source="swarm_panel"
            ))

        self._record_stage_profile(stage="differential", wall_time_s=time.perf_counter() - stage_started)
        await self._emit_trace(
            "verification",
            f"Consolidated hypotheses: {', '.join(self.state.working_hypotheses)}",
            {
                "hypotheses": self.state.working_hypotheses,
                "provenance": {label: provenance.get(label, []) for label in self.state.working_hypotheses},
                "metrics": self._stage_metrics("differential", wall_time_s=time.perf_counter() - stage_started)
            }
        )

        # W3 Module D.5 — route through cognitive loop when enabled.
        _cog_on = False
        try:
            _pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if _pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                _pol = load_runtime_policy()
            _cog_on = bool(getattr(_pol, "cognitive_controller_enabled", False))
        except Exception:
            _cog_on = False

        # 6a: early-exit gate. When the top-1 candidate is dominant (score>=0.7
        # and entropy<0.4) and the case has no red flags or must-not-miss
        # concerns, skip COGNITIVE_LOOP / BACKWARD / OUTCOME and move straight
        # to VERIFICATION. Saves ~30-60s on confident routine cases. Disable
        # via CDSS_EARLY_EXIT_ENABLED=0 to force the full pipeline.
        early_exit = False
        try:
            if os.environ.get("CDSS_EARLY_EXIT_ENABLED", "1").strip().lower() not in {"0", "false", "off"}:
                _diff_obj = self.typed_case_bundle.get("differential")
                _frontier_obj = self.typed_case_bundle.get("frontier")
                _findings_obj = self._ensure_typed_findings()
                _entropy_val = float(getattr(_diff_obj, "uncertainty", 1.0) or 1.0)
                _red_flags_present = bool(getattr(_findings_obj, "red_flags", None))
                _mnm_present = bool(getattr(_frontier_obj, "must_not_miss", None)) if _frontier_obj else False
                _top_score = 0.0
                if _diff_obj and getattr(_diff_obj, "candidates", None):
                    _top_score = float(_diff_obj.candidates[0].score or 0.0)
                if (
                    _entropy_val < 0.4
                    and _top_score >= 0.7
                    and not _red_flags_present
                    and not _mnm_present
                ):
                    early_exit = True
                    await self._emit_trace(
                        "stage_skip",
                        f"Early-exit: top-1 dominant (score={_top_score:.2f}, entropy={_entropy_val:.2f}) "
                        "and no red flags / must-not-miss — skipping COGNITIVE_LOOP / BACKWARD / OUTCOME.",
                        {
                            "top_score": round(_top_score, 2),
                            "entropy": round(_entropy_val, 2),
                            "red_flags": _red_flags_present,
                            "must_not_miss": _mnm_present,
                        },
                    )
                elif (
                    str(getattr(self.runtime_policy, "runtime_speed_profile", "") or "").strip().lower() == "fast"
                    and risk_profile.urgency == UrgencyTier.ROUTINE
                    and not _red_flags_present
                    and not _mnm_present
                    and (_top_score >= 0.45 or _entropy_val < 1.35)
                ):
                    early_exit = True
                    await self._emit_trace(
                        "stage_skip",
                        f"Fast-path: routine case with acceptable candidate signal "
                        f"(score={_top_score:.2f}, entropy={_entropy_val:.2f}) - skipping simulation stages.",
                        {
                            "top_score": round(_top_score, 2),
                            "entropy": round(_entropy_val, 2),
                            "runtime_speed_profile": "fast",
                        },
                    )
        except Exception as _ee_exc:
            logging.getLogger("rrrie-cdss").debug("[EARLY_EXIT] gate skipped (%s)", _ee_exc)
            early_exit = False

        if early_exit:
            self.state.phase = CdssPhase.VERIFICATION
        else:
            self.state.phase = CdssPhase.COGNITIVE_LOOP if _cog_on else CdssPhase.BACKWARD_SIMULATION

    # ------------------------------------------------------------------
    # W3 Module D.5 — Cognitive loop phase
    # ------------------------------------------------------------------

    async def _run_cognitive_loop(self):
        """MCTS-driven action selection loop (W3 D.5 + I.1/I.2/I.3 wiring).

        Runs up to `policy.mcts_max_ticks` ticks. Each tick:
          1. Compute/refresh DiagnosticBelief from `typed_case_bundle`.
          2. Terminal checks (should_stop / should_escalate) — short-circuit.
          3. MCTS rollouts (UCT, budget=`mcts_rollout_budget`) → chosen action.
          4. Soft-apply chosen action's rollout to belief (no real LLM yet —
             real execution dispatches are stubbed until later waves).

        Meta-controller post-hook (when `meta_controller_enabled`):
          5. self_refine LLM pass → CTX atoms appended to input_context.

        Always transitions to BACKWARD_SIMULATION on exit, even on error.
        """
        from src.cdss.contracts.models import DiagnosticBelief
        from src.cdss.runtime.mcts import tick as _mcts_tick
        from src.cdss.runtime.actions import ActionKind
        from src.cdss.reasoning.evi import simulate_rollout as _sim_rollout
        from src.cdss.reasoning.belief_propagation import entropy as _entropy
        stage_started = time.perf_counter()
        self._set_profile_stage("cognitive_loop")

        try:
            pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                pol = load_runtime_policy()
        except Exception:
            pol = None

        max_ticks = int(getattr(pol, "mcts_max_ticks", 6)) if pol else 6
        rollout_budget = int(getattr(pol, "mcts_rollout_budget", 20)) if pol else 20
        u_miss_w = float(getattr(pol, "utility_miss_risk_weight", 0.8)) if pol else 0.8
        u_tight_w = float(getattr(pol, "utility_tightness_weight", 0.2)) if pol else 0.2
        u_budget_w = float(getattr(pol, "utility_budget_weight", 0.05)) if pol else 0.05
        meta_on = bool(getattr(pol, "meta_controller_enabled", False)) if pol else False

        utility_kwargs = {
            "miss_risk_weight": u_miss_w,
            "tightness_weight": u_tight_w,
            "budget_weight": u_budget_w,
        }

        def _get_belief() -> DiagnosticBelief:
            existing = self.typed_case_bundle.get("diagnostic_belief")
            if isinstance(existing, DiagnosticBelief):
                return existing
            if isinstance(existing, dict):
                try:
                    return DiagnosticBelief.model_validate(existing)
                except Exception:
                    return DiagnosticBelief()
            return DiagnosticBelief()

        belief = _get_belief()
        if not (belief.species_posterior or belief.family_posterior):
            # Nothing to reason about — skip loop.
            await self._emit_trace(
                "cognitive_loop",
                "Skipping cognitive loop — empty DiagnosticBelief.",
                {"metrics": self._stage_metrics("cognitive_loop", wall_time_s=time.perf_counter() - stage_started)},
            )
            self.state.phase = CdssPhase.BACKWARD_SIMULATION
            return

        trace_ticks: list[dict[str, Any]] = []
        ds_K = 0.0  # TODO: surface from _fold_ds_into_belief last call once persisted per-step.

        for tick_i in range(max_ticks):
            budget_spent = min(1.0, float(tick_i) / float(max(1, max_ticks)))
            try:
                result = await _mcts_tick(
                    belief,
                    budget_spent_frac=budget_spent,
                    rollout_budget=rollout_budget,
                    parallelism=4,
                    ds_conflict_K=ds_K,
                    utility_kwargs=utility_kwargs,
                )
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[COGLOOP] tick %d failed: %s", tick_i, exc)
                break

            trace_ticks.append({
                "tick": tick_i,
                "chosen": result.chosen.kind.value,
                "target": result.chosen.target,
                "q_hat": round(result.q_hat, 4),
                "rollouts": result.total_rollouts,
                "terminated": result.terminated,
                "termination_reason": result.termination_reason,
                "escalation_reason": result.escalation_reason,
            })

            if result.terminated:
                if result.escalation_reason:
                    self.typed_case_bundle["cognitive_escalation"] = result.escalation_reason
                break

            # Apply chosen action as a soft rollout (stub dispatch — real action
            # execution is wired incrementally in later PRs; for now MCTS
            # drives belief mutation via the same rollout math used to score).
            try:
                _sim_rollout(belief, result.chosen, utility_kwargs=utility_kwargs)
                from src.cdss.reasoning.evi import _apply_soft_update, _apply_soft_downweight
                top_items = sorted((belief.species_posterior or {}).items(), key=lambda kv: kv[1], reverse=True)
                top_h = top_items[0][0] if top_items else ""
                target = result.chosen.target or top_h
                if result.chosen.kind == ActionKind.ABLATE_FINDING and top_h:
                    belief = _apply_soft_downweight(belief, top_h, decay=0.5)
                elif result.chosen.kind in (
                    ActionKind.CONSULT_SPECIALIST,
                    ActionKind.QUERY_ONTOLOGY,
                    ActionKind.SEARCH_LITERATURE,
                    ActionKind.RUN_SWARM_LEVEL,
                    ActionKind.REQUEST_TEST,
                    ActionKind.REQUEST_HISTORY,
                ) and target and target in (belief.species_posterior or {}):
                    boost = 1.6 if result.chosen.kind == ActionKind.REQUEST_TEST else 1.4
                    belief = _apply_soft_update(belief, target, boost=boost)
                # EXPAND_HIERARCHY handled inside simulate_rollout's math; no persistent change yet.

                # Record entropy progression on species layer for future MCTS utility calibration.
                species = dict(belief.species_posterior or {})
                if species:
                    belief = belief.model_copy(update={
                        "entropy_history": list(belief.entropy_history or []) + [round(float(_entropy(species)), 6)],
                        "step": int(belief.step) + 1,
                    })
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[COGLOOP] apply tick %d failed: %s", tick_i, exc)

        # Persist belief.
        self.typed_case_bundle["diagnostic_belief"] = belief

        # Meta-controller self-refine hook (Module I.3).
        if meta_on:
            try:
                from src.cdss.reasoning.self_refine import self_refine, atoms_from_ctx_lines
                findings = self._ensure_typed_findings()
                chief = (findings.summary or " ".join(findings.positive_findings[:3]) or "")[:300]
                top_species = sorted(
                    (belief.species_posterior or {}).items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                top_family = sorted(
                    (belief.family_posterior or {}).items(), key=lambda kv: kv[1], reverse=True
                )[:5]
                llm_client = getattr(self._gateway, "bridge", None)
                if llm_client is not None:
                    ctx_lines = await self_refine(
                        llm_client=llm_client,
                        chief_complaint=chief,
                        top_species=top_species,
                        top_family=top_family,
                        red_flags=list(findings.red_flags or []),
                        evidence_summary="",
                    )
                    atoms = atoms_from_ctx_lines(ctx_lines)
                    if atoms:
                        findings.input_context = list(findings.input_context or []) + atoms
                        self.typed_case_bundle["findings"] = findings
                        await self._emit_trace(
                            "cognitive_loop",
                            f"Self-refine appended {len(atoms)} clinician-gap atoms.",
                            {"atoms": atoms},
                        )
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[SELF_REFINE] failed: %s", exc)

        self._record_stage_profile(stage="cognitive_loop", wall_time_s=time.perf_counter() - stage_started)
        await self._emit_trace(
            "cognitive_loop",
            f"Cognitive loop completed after {len(trace_ticks)} tick(s).",
            {
                "ticks": trace_ticks,
                "metrics": self._stage_metrics("cognitive_loop", wall_time_s=time.perf_counter() - stage_started),
            },
        )
        self.state.phase = CdssPhase.BACKWARD_SIMULATION

    async def _run_backward_causality_simulation(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("backward_simulation")
        """
        Phase 1.5: Backward (Past) Simulation - Disease Origin & Causal Inference
        This traces biochemical and pathophysiological causal pathways backwards from current symptoms
        to structural or molecular origins, validating if the hypotheses scientifically explain the raw data.
        """
        print(f"[{time.strftime('%H:%M:%S')}] [Causal Pathophysiology] Running backward-time source simulations...")
        await self._emit_trace("governor", "Tracing causal biochemical pathways backwards from symptoms to disease origin...", {"candidates": self.state.working_hypotheses})

        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        interventions = self._typed_interventions()

        # Parallel execution: epi/spec calibration (1-2 LLM calls) runs concurrently with
        # mechanism_frames generation (1 LLM call). Both read from completed differential and
        # write to disjoint typed_case_bundle keys — no data dependency.
        _mechanism_holder: list = []

        async def _generate_mechanism_inner():
            result = await self._gateway.generate_mechanism_frames(
                findings, self._typed_risk_profile(), fact_graph, interventions,
            )
            _mechanism_holder.append(result)

        await asyncio.gather(
            self._run_epi_spec_calibration(),
            _generate_mechanism_inner(),
        )

        mechanism_result = _mechanism_holder[0] if _mechanism_holder else None
        if mechanism_result is None:
            return
        self._record_gateway_metrics("backward_simulation", mechanism_result.metrics)
        if mechanism_result.error:
            self.typed_case_bundle["mechanism_error"] = dict(mechanism_result.error)
        if mechanism_result.mechanism_frames:
            self.typed_case_bundle["mechanism_frames"] = mechanism_result.mechanism_frames
            primary = mechanism_result.mechanism_frames[0]
            for frame in mechanism_result.mechanism_frames[:3]:
                label = str(frame.active_state or frame.primary_mechanism or "").strip()
                if not label:
                    continue
                self.signal_journal.extend(
                    [
                        CaseSignal(
                            kind=SignalKind.MECHANISM,
                            label=label,
                            canonical_label=normalize_candidate_label(label) or "",
                            source_stage="backward_simulation",
                            raw_span=frame.primary_mechanism or label,
                            confidence=float(frame.confidence or 0.0),
                        )
                    ]
                )
            await self._emit_trace(
                "governor",
                "Mechanism frames attached to the shared state and aligned against the active differential.",
                {
                    "mechanism": primary.primary_mechanism or primary.active_state,
                    "active_state": primary.active_state,
                    "metrics": self._stage_metrics("backward_simulation", wall_time_s=time.perf_counter() - stage_started),
                },
            )
        else:
            await self._emit_trace(
                "governor",
                "Mechanism frame generation returned no stable override; retaining the current differential.",
                {
                    "metrics": self._stage_metrics("backward_simulation", wall_time_s=time.perf_counter() - stage_started),
                    "error": mechanism_result.error,
                },
            )

        # W4 Module F — causal-do mechanism verifier. Gated on flag; no-op when
        # off. Runs BFS probes over the curated pathway graph + frame edges, folds
        # robustness back into species posterior via Bayes update.
        try:
            pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                pol = load_runtime_policy()
        except Exception:
            pol = None
        if pol is not None and bool(getattr(pol, "causal_do_verifier_enabled", False)):
            try:
                frames = mechanism_result.mechanism_frames if mechanism_result else []
                updated = self._fold_causal_do_into_belief(mechanism_frames=frames or [])
                if updated is not None:
                    probes = self.typed_case_bundle.get("causal_do_probes") or []
                    await self._emit_trace(
                        "governor",
                        f"Causal-do verifier folded {len(probes)} probe(s) into species posterior.",
                        {"probes": probes[:8]},
                    )
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[CAUSAL_DO] fold failed: %s", exc)

        self._record_stage_profile(stage="backward_simulation", wall_time_s=time.perf_counter() - stage_started)
        self.state.transition(CdssPhase.OUTCOME_SIMULATION)
        return


    async def _run_test_time_simulation(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("outcome_simulation")
        """
        Phase 2: Outcome Simulator (The Challenger / Red Teamer for Mortality Asymmetry)
        """
        print(f"[{time.strftime('%H:%M:%S')}] [Test-Time Simulator] Running forward-time mortality simulations...")
        await self._emit_trace("governor", "Running Test-Time mortality simulations to stress-test treatment pathways...", {"candidates": self.state.working_hypotheses})
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        frontier = self.typed_case_bundle.get("frontier") if isinstance(self.typed_case_bundle.get("frontier"), HypothesisFrontier) else self._provisional_frontier()
        differential = self.typed_case_bundle.get("differential") if isinstance(self.typed_case_bundle.get("differential"), DifferentialSet) else DifferentialSet()
        evidence = self._typed_evidence()
        interventions = self._typed_interventions()
        # Auto-enable deep_thinking for critical complexity cases so forward simulation
        # always runs when epistemic entropy is high, regardless of user ATOM flag.
        _challenger_deep_thinking = getattr(self, "deep_thinking", False) or (self._case_complexity == "critical")
        policy = self.runtime_policy
        # W7.1 K.1 — always-on draft challenger with two-tier escalation.
        _two_tier_on = bool(getattr(policy, "always_on_challenger_enabled", False))
        _alt_thresh = float(getattr(policy, "challenger_escalate_alt_threshold", 0.4))
        _esc_sevs = {
            s.strip().lower()
            for s in str(getattr(policy, "challenger_escalate_severity", "high,critical")).split(",")
            if s.strip()
        }
        challenge_result = None
        draft_result = None
        if _two_tier_on:
            draft_result = await self._gateway.challenge(
                findings,
                self._typed_risk_profile(),
                fact_graph,
                frontier,
                differential,
                evidence,
                interventions,
                deep_thinking=False,
                loop_iteration=self.typed_case_bundle.get("deep_reasoning_loop", 0),
                tier="draft",
            )
            self._record_gateway_metrics("outcome_simulation_draft", draft_result.metrics)
            # Decide escalation: any severe issue OR ALT score above threshold OR error.
            severe_issue = any(
                str(getattr(it, "severity", "")).strip().lower() in _esc_sevs
                for it in (draft_result.issues or [])
            )
            high_alt = any(
                float(alt.get("score", 0.0) or 0.0) >= _alt_thresh
                for alt in (draft_result.alt_hypotheses or [])
            )
            should_escalate = bool(draft_result.error) or severe_issue or high_alt or _challenger_deep_thinking
            if should_escalate:
                challenge_result = await self._gateway.challenge(
                    findings,
                    self._typed_risk_profile(),
                    fact_graph,
                    frontier,
                    differential,
                    evidence,
                    interventions,
                    deep_thinking=_challenger_deep_thinking,
                    loop_iteration=self.typed_case_bundle.get("deep_reasoning_loop", 0),
                    tier="full",
                )
                self.typed_case_bundle["challenger_escalated"] = True
            else:
                challenge_result = draft_result
                self.typed_case_bundle["challenger_escalated"] = False
        else:
            challenge_result = await self._gateway.challenge(
                findings,
                self._typed_risk_profile(),
                fact_graph,
                frontier,
                differential,
                evidence,
                interventions,
                deep_thinking=_challenger_deep_thinking,
                loop_iteration=self.typed_case_bundle.get("deep_reasoning_loop", 0),
            )
        self._record_gateway_metrics("outcome_simulation", challenge_result.metrics)
        self.typed_case_bundle["challenge_issues"] = challenge_result.issues
        self.typed_case_bundle["loop_directive"] = challenge_result.loop_directive
        if challenge_result.error:
            self.typed_case_bundle["challenge_error"] = dict(challenge_result.error)
        aggregated = {candidate.label: float(candidate.score or 0.0) for candidate in differential.candidates[:5] if str(candidate.label or "").strip()}
        provenance = {candidate.label: ["typed_differential"] for candidate in differential.candidates[:5] if str(candidate.label or "").strip()}
        for alt in challenge_result.alt_hypotheses[:4]:
            label = normalize_candidate_label(str(alt.get("label", "") or "").strip()) or str(alt.get("label", "") or "").strip()
            if not label:
                continue
            aggregated[label] = max(float(alt.get("score", 0.0) or 0.0), aggregated.get(label, 0.0))
            provenance.setdefault(label, []).append("challenger")
        # W7.2 K.2 — steel-man top-2 protocol. Run only when we have a real top-2.
        if bool(getattr(policy, "steelman_enabled", False)) and len(differential.candidates) >= 2:
            _top1 = str(differential.candidates[0].label or "").strip()
            _top2 = str(differential.candidates[1].label or "").strip()
            _top2_rat = "; ".join(list(differential.candidates[1].rationale or [])[:2])
            if _top1 and _top2 and _top1 != _top2:
                steel = await self._gateway.bridge.steelman_top2(findings, evidence, _top1, _top2, _top2_rat)
                if isinstance(steel, dict) and not steel.get("error"):
                    self.typed_case_bundle["steelman_argument"] = {
                        "label": str(steel.get("label", "") or ""),
                        "argument_score": float(steel.get("argument_score", 0.0) or 0.0),
                        "rationale": str(steel.get("rationale", "") or ""),
                        "anchor_top1": _top1,
                    }
                    arg_score = float(steel.get("argument_score", 0.0) or 0.0)
                    arg_thresh = float(getattr(policy, "steelman_argument_threshold", 0.7))
                    steel_label = normalize_candidate_label(str(steel.get("label", "") or "").strip()) or str(steel.get("label", "") or "").strip()
                    if arg_score >= arg_thresh and steel_label:
                        # Surface for K.5 BT tournament gate (downstream wave). Boost
                        # aggregated score conservatively so rerank reflects credible swap pressure.
                        aggregated[steel_label] = max(aggregated.get(steel_label, 0.0), min(0.95, arg_score))
                        provenance.setdefault(steel_label, []).append("steelman")
                        self.typed_case_bundle["steelman_swap_flagged"] = True
                        await self._emit_trace(
                            "ie_override",
                            f"Steel-man surfaced strong defense ({arg_score:.2f}) for #2 candidate: {steel_label}",
                            {"top1": _top1, "top2": steel_label, "argument_score": arg_score},
                        )
        # W7.2 K.3 — counterfactual finding-ablation probe.
        if bool(getattr(policy, "counterfactual_ablation_enabled", False)):
            try:
                from src.cdss.contracts.models import DiagnosticBelief as _DB
                from src.cdss.reasoning.counterfactual_ablation import find_pivot_findings as _find_pivots
                _belief = self.typed_case_bundle.get("diagnostic_belief")
                if isinstance(_belief, _DB):
                    _top_n = int(getattr(policy, "counterfactual_ablation_top_n", 5))
                    _level = int(_belief.active_level if _belief.species_posterior else 0)
                    pivot_report = _find_pivots(_belief, level=_level, top_n=_top_n)
                    if pivot_report.get("pivot_findings"):
                        self.typed_case_bundle["pivot_findings"] = list(pivot_report["pivot_findings"])
                        self.typed_case_bundle["absence_dependent_diagnosis"] = bool(pivot_report.get("absence_dependent", False))
                        await self._emit_trace(
                            "governor",
                            f"Counterfactual ablation found {len(pivot_report['pivot_findings'])} pivot findings"
                            + (" (absence-dependent diagnosis)" if pivot_report.get("absence_dependent") else ""),
                            {"pivots": pivot_report["pivot_findings"][:3]},
                        )
            except Exception as _exc:
                # Probe is best-effort; never block challenger flow.
                self.typed_case_bundle["pivot_findings_error"] = str(_exc)

        # W7.3 K.4 — premise-conflict resolver. Fires on high contradiction mass
        # OR high DS conflict K (when DS fusion is enabled).
        if bool(getattr(policy, "premise_conflict_resolver_enabled", False)):
            try:
                _k_thresh = float(getattr(policy, "premise_conflict_k_threshold", 0.6))
                _c_thresh = float(getattr(policy, "premise_conflict_contradiction_threshold", 0.24))
                _ds_k = float(self.typed_case_bundle.get("ds_conflict_k", 0.0) or 0.0)
                _contradiction = float(getattr(evidence, "contradiction_mass", 0.0) or 0.0)
                if _ds_k >= _k_thresh or _contradiction >= _c_thresh:
                    # Build conflicting premise list from contradicting atoms / authority claims.
                    _premises: list[str] = []
                    for node in fact_graph.nodes:
                        if str(getattr(node, "category", "")).lower() in {"authority_claim", "contradiction"}:
                            lbl = str(getattr(node, "label", "") or "").strip()
                            if lbl:
                                _premises.append(lbl)
                    # Fallback: pull the strongest contradicting evidence atoms.
                    if not _premises and getattr(evidence, "atoms", None):
                        for atom in list(evidence.atoms)[:6]:
                            txt = str(getattr(atom, "text", "") or getattr(atom, "label", "") or "").strip()
                            if txt:
                                _premises.append(txt)
                    if _premises:
                        resolution = await self._gateway.bridge.resolve_premise_conflict(
                            findings, evidence, _premises, _ds_k
                        )
                        if isinstance(resolution, dict) and not resolution.get("error"):
                            self.typed_case_bundle["premise_conflict_resolution"] = {
                                "ds_conflict_k": _ds_k,
                                "contradiction_mass": _contradiction,
                                "artifact_premise": resolution.get("artifact_premise", ""),
                                "discriminator": resolution.get("discriminator", ""),
                                "decision": resolution.get("decision", ""),
                                "decision_reason": resolution.get("decision_reason", ""),
                            }
                            await self._emit_trace(
                                "governor",
                                f"Premise-conflict resolver: decision={resolution.get('decision', '')}, "
                                f"discriminator={resolution.get('discriminator', '')}",
                                {"k": _ds_k, "contradiction": _contradiction},
                            )
            except Exception as _exc:
                self.typed_case_bundle["premise_conflict_error"] = str(_exc)

        # W7.3 K.5 — Bradley-Terry pairwise tournament. Fires on high entropy
        # over the top-K candidates OR when steelman flagged a swap.
        if bool(getattr(policy, "bradley_terry_tournament_enabled", False)) and len(differential.candidates) >= 2:
            try:
                from src.cdss.reasoning.bradley_terry import (
                    bt_mle as _bt_mle,
                    tournament_rank as _bt_rank,
                )
                import math as _math
                _k_top = int(getattr(policy, "bt_tournament_top_k", 3))
                _ent_trig = float(getattr(policy, "bt_tournament_entropy_trigger", 0.8))
                _judges = int(getattr(policy, "bt_tournament_judges_per_pair", 3))
                top_cands = [c for c in differential.candidates[:_k_top] if str(c.label or "").strip()]
                # Shannon entropy over normalized top-K scores.
                scores = [max(1e-9, float(c.score or 0.0)) for c in top_cands]
                z = sum(scores) or 1.0
                norm = [s / z for s in scores]
                _ent = -sum(p * _math.log(p) for p in norm if p > 0.0)
                _swap_flag = bool(self.typed_case_bundle.get("steelman_swap_flagged", False))
                if (_ent >= _ent_trig or _swap_flag) and len(top_cands) >= 2:
                    # Fire pairwise judges across all (i,j) for i<j; framings rotate.
                    framings = ["bayesian", "mechanism", "epi"]
                    judge_tasks = []
                    pair_meta: list[tuple[str, str, str]] = []
                    for i in range(len(top_cands)):
                        for j in range(i + 1, len(top_cands)):
                            a = top_cands[i].label
                            b = top_cands[j].label
                            for k_idx in range(min(_judges, len(framings))):
                                framing = framings[k_idx % len(framings)]
                                temp = 0.0 + (k_idx * 0.2)
                                judge_tasks.append(self._gateway.bridge.pairwise_judge(findings, evidence, a, b, framing, temp))
                                pair_meta.append((a, b, framing))
                    if judge_tasks:
                        results = await asyncio.gather(*judge_tasks, return_exceptions=True)
                        # Aggregate votes: each judge's WIN goes to winner with weight = prob.
                        votes: dict[tuple[str, str], float] = {}
                        for res, (a, b, _) in zip(results, pair_meta):
                            if isinstance(res, Exception) or not isinstance(res, dict) or res.get("error"):
                                continue
                            winner = str(res.get("winner", "") or "").strip()
                            prob = float(res.get("prob", 0.5) or 0.5)
                            if winner == a:
                                votes[(a, b)] = votes.get((a, b), 0.0) + prob
                                votes[(b, a)] = votes.get((b, a), 0.0) + (1.0 - prob)
                            elif winner == b:
                                votes[(b, a)] = votes.get((b, a), 0.0) + prob
                                votes[(a, b)] = votes.get((a, b), 0.0) + (1.0 - prob)
                        if votes:
                            strengths = _bt_mle(votes, iters=80)
                            ranked = _bt_rank(strengths)
                            self.typed_case_bundle["bt_strengths"] = dict(strengths)
                            self.typed_case_bundle["bt_ranked"] = list(ranked)
                            # If BT top-1 differs from current top-1, surface for swap.
                            current_top1 = top_cands[0].label
                            bt_top1 = ranked[0][0] if ranked else ""
                            if bt_top1 and bt_top1 != current_top1:
                                # Boost BT winner in aggregated map so rerank reflects pairwise verdict.
                                aggregated[bt_top1] = max(aggregated.get(bt_top1, 0.0), float(strengths.get(bt_top1, 0.5)))
                                provenance.setdefault(bt_top1, []).append("bradley_terry")
                                self.typed_case_bundle["bt_swap_top1"] = bt_top1
                                await self._emit_trace(
                                    "ie_override",
                                    f"Bradley-Terry tournament swapped top-1: {current_top1} → {bt_top1}",
                                    {"strengths": dict(strengths), "entropy": _ent},
                                )
            except Exception as _exc:
                self.typed_case_bundle["bt_tournament_error"] = str(_exc)

        if aggregated:
            reranked = self._apply_adaptive_rerank(aggregated, provenance)
            if reranked:
                self.state.working_hypotheses = reranked[:3]
        if challenge_result.error:
            await self._emit_trace(
                "governor",
                "Typed challenger returned no stable structured output; catastrophic override remains unresolved and closure stays safety-blocked.",
                {
                    "error": challenge_result.error,
                    "metrics": self._stage_metrics("outcome_simulation", wall_time_s=time.perf_counter() - stage_started),
                },
            )
        elif challenge_result.alt_hypotheses:
            lead = self.state.working_hypotheses[0] if self.state.working_hypotheses else str(challenge_result.alt_hypotheses[0].get("label", "") or "")
            reason = challenge_result.alt_hypotheses[0].get("rationale", "") if challenge_result.alt_hypotheses else ""
            await self._emit_trace("ie_override", f"Adaptive challenger escalated the anchor to: {lead}", {
                "trap": reason,
                "loop_action": challenge_result.loop_directive.action,
                "metrics": self._stage_metrics("outcome_simulation", wall_time_s=time.perf_counter() - stage_started),
            })
        else:
            await self._emit_trace("governor", "Typed challenger found no catastrophic override. Proceeding to verification.", {
                "loop_action": challenge_result.loop_directive.action,
                "metrics": self._stage_metrics("outcome_simulation", wall_time_s=time.perf_counter() - stage_started),
            })

        loop_count = self.typed_case_bundle.get("deep_reasoning_loop", 0)

        # In ATOM mode, we FORCE at least 1 deliberate counter-factual loop even if the model feels confident,
        # to ensure deep broad exploration of alternative or rare edge cases.
        force_deep_loop = False
        action_reason = "Test-Time Simulator found asymmetric risks"

        if self.deep_thinking:
            if challenge_result.loop_directive and challenge_result.loop_directive.action in {"request_targeted_retrieval", "revise_hypotheses"}:
                force_deep_loop = True
                action_reason = f"Test-Time Simulator found asymmetric risks ({challenge_result.loop_directive.action})"
            elif loop_count == 0:
                # Force first loop in deep thinking unconditionally
                force_deep_loop = True
                action_reason = "Unconditional counter-factual exploration for potential rare edge cases"

        # Plan Item 5 — budget-aware depth gate. Track cumulative outcome-
        # simulation wall time and skip the next deep loop if we have already
        # spent the per-case budget. Stops 10-call mortality-sim explosions
        # from blowing the case latency, regardless of entropy signal.
        _mortality_elapsed = float(self.typed_case_bundle.get("_mortality_sim_wall_time_s", 0.0)) + (
            time.perf_counter() - stage_started
        )
        self.typed_case_bundle["_mortality_sim_wall_time_s"] = _mortality_elapsed
        _budget_s = float(getattr(policy, "mortality_sim_per_case_budget_s", 60.0))
        _budget_exceeded = _budget_s > 0 and _mortality_elapsed >= _budget_s

        if force_deep_loop and loop_count < 2 and not _budget_exceeded:
            self.typed_case_bundle["deep_reasoning_loop"] = loop_count + 1
            await self._emit_trace(
                "governor",
                f"⚕️ ATOM DEEP REASONING ACTIVATED: {action_reason}. Redirecting to Swarm (Loop {loop_count + 1}/2).",
                {"loop_action": getattr(challenge_result.loop_directive, "action", "force_deep_retrieval") if challenge_result.loop_directive else "force_deep_retrieval", "loop": loop_count + 1}
            )
            self._record_stage_profile(stage="outcome_simulation", wall_time_s=time.perf_counter() - stage_started)
            self.state.transition(CdssPhase.R2)
            return
        if force_deep_loop and _budget_exceeded:
            await self._emit_trace(
                "governor",
                f"Mortality-sim budget exhausted ({_mortality_elapsed:.1f}s ≥ {_budget_s:.0f}s); skipping additional deep loop.",
                {"elapsed_s": _mortality_elapsed, "budget_s": _budget_s, "loop": loop_count},
            )

        # W4 Module G — Bayesian-model-averaged utility over disposition actions.
        # Runs after challenger but before VERIFICATION so the verifier sees the
        # disposition recommendation as one more signal. Gated on policy flag.
        try:
            pol = self._policy_snapshot() if hasattr(self, "_policy_snapshot") else None
            if pol is None:
                from src.cdss.runtime.policy import load_runtime_policy
                pol = load_runtime_policy()
        except Exception:
            pol = None
        if pol is not None and bool(getattr(pol, "bma_outcome_enabled", False)):
            try:
                self._run_bma_outcome_analysis()
                bma = self.typed_case_bundle.get("bma_outcome") or {}
                chosen = bma.get("chosen") or {}
                if chosen:
                    await self._emit_trace(
                        "governor",
                        f"BMA disposition → {chosen.get('action','?')} (E[U]={chosen.get('e_utility',0):.2f}).",
                        {"bma": chosen, "reason": bma.get("chosen_reason", "")},
                    )
            except Exception as exc:
                logging.getLogger("rrrie-cdss").warning("[BMA] analysis failed: %s", exc)

        self._record_stage_profile(stage="outcome_simulation", wall_time_s=time.perf_counter() - stage_started)
        self.state.transition(CdssPhase.VERIFICATION)
        return

    # -----------------------------------------------------------------
    # W4 Module G — BMA disposition analysis
    # -----------------------------------------------------------------

    def _run_bma_outcome_analysis(self) -> None:
        """Compute BMA disposition recommendation from current belief + priors.

        Writes `typed_case_bundle['bma_outcome']` with `{chosen, chosen_reason,
        ranked, counterfactual_discharge}`. No LLM — pure table lookup + BMA
        math. Safe to call when posterior empty (returns early).
        """
        try:
            from src.cdss.contracts.models import DiagnosticBelief
            from src.cdss.reasoning.bma_utility import (
                decide_disposition,
                counterfactual_discharge_harm,
                load_mortality_priors,
            )
        except Exception as exc:
            logging.getLogger("rrrie-cdss").warning("[BMA] modules unavailable: %s", exc)
            return

        existing = self.typed_case_bundle.get("diagnostic_belief")
        if isinstance(existing, DiagnosticBelief):
            belief = existing
        elif isinstance(existing, dict):
            try:
                belief = DiagnosticBelief.model_validate(existing)
            except Exception:
                return
        else:
            # Fall back to a flat posterior from the current differential so the
            # module still emits a recommendation on legacy non-belief runs.
            differential = self.typed_case_bundle.get("differential")
            if not isinstance(differential, DifferentialSet) or not differential.candidates:
                return
            posterior: dict[str, float] = {}
            for cand in differential.candidates[:10]:
                label = normalize_candidate_label(str(cand.label or "")) or str(cand.label or "").strip().lower()
                if not label:
                    continue
                posterior[label] = max(posterior.get(label, 0.0), float(cand.score or 0.0))
            if not posterior:
                return
            total = sum(posterior.values()) or 1.0
            posterior = {k: v / total for k, v in posterior.items()}
            belief = DiagnosticBelief(species_posterior=posterior)

        species_post = dict(belief.species_posterior or {})
        if not species_post:
            return

        priors = load_mortality_priors()
        # Build tier map from prior rows so unfamiliar hypotheses fall back to
        # tier-based defaults.
        tier_of: dict[str, str] = {h: row.tier for h, row in priors.items()}
        # Severity multiplier: EMERGENCY 1.5, URGENT 1.1, else 1.0.
        sev_map: dict[str, float] = {}
        for h in species_post.keys():
            t = tier_of.get(h, "UNKNOWN")
            if t == "EMERGENCY":
                sev_map[h] = 1.5
            elif t == "URGENT":
                sev_map[h] = 1.1
            else:
                sev_map[h] = 1.0

        decision = decide_disposition(
            species_post,
            priors,
            severity_factor=sev_map,
            tier_of=tier_of,
            top_k=5,
        )
        cf = counterfactual_discharge_harm(
            species_post,
            priors,
            tier_of=tier_of,
            top_k=5,
        )
        self.typed_case_bundle["bma_outcome"] = {
            **decision,
            "counterfactual_discharge": cf,
        }


    async def _run_chief_consilium(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("verification")
        """
        Phase 3: The Chief of Staff resolves swarm conflicts and produces a ranked final differential.
        Uses a real LLM call to synthesize evidence, validate consistency, and select the best diagnosis.
        """
        print(f"[{time.strftime('%H:%M:%S')}] [Chief of Staff] Synthesizing swarm consensus...")
        await self._emit_trace("verification", "Chief of Staff synthesizing final differential from swarm evidence...", {})
        findings = self._ensure_typed_findings()
        fact_graph = self._ensure_typed_fact_graph()
        frontier = self.typed_case_bundle.get("frontier") if isinstance(self.typed_case_bundle.get("frontier"), HypothesisFrontier) else self._provisional_frontier()
        differential = self.typed_case_bundle.get("differential") if isinstance(self.typed_case_bundle.get("differential"), DifferentialSet) else DifferentialSet()
        evidence = self._typed_evidence()
        interventions = self._typed_interventions()

        # Compute differential-aware intervention safety now that we have the final frontier.
        # This cross-checks the diagnosis-specific unsafe_interventions (from the registry) against
        # both the proposed intervention list AND the patient's own reported medications.
        try:
            from src.cdss.clinical.intervention_safety import build_intervention_safety_assessment
            intervention_safety = build_intervention_safety_assessment(
                findings,
                self._typed_risk_profile(),
                frontier,
                interventions,
                fact_graph=fact_graph,
            )
            self.typed_case_bundle["intervention_safety"] = intervention_safety
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        verification_result = await self._gateway.verify(
            findings,
            self._typed_risk_profile(),
            fact_graph,
            frontier,
            interventions,
            differential,
            evidence,
        )
        self._record_gateway_metrics("verification", verification_result.metrics)
        verification_issues = list(verification_result.issues)
        if verification_result.error:
            verification_issues.append(
                VerificationIssue(
                    severity="major",
                    issue_type="verifier_unavailable",
                    detail="Structured verifier output was unavailable, so autonomous closure remains blocked.",
                )
            )
        self.typed_case_bundle["verification_issues"] = verification_issues
        self.typed_case_bundle["verification_hint"] = verification_result.decision_hint or ("abstain" if verification_result.error else "")

        # 4. Safety State Feedback: Penalize anchor when the DIAGNOSIS is likely wrong.
        # Three trigger families:
        #   (a) deterministic must-not-miss / missed-critical-risk issue types,
        #   (b) candidate_conflict / unresolved_hazard / pregnancy_context_gap from
        #       the deterministic verifier — these directly imply the anchor is unsafe,
        #   (c) LLM-side anchor-mismatch signals where the verifier explicitly
        #       names a better-fitting alternative in free text. We parse those
        #       so the verifier's clinical judgment actually flows into ranking,
        #       which the previous trigger set missed (benchmark cases 1/4/7).
        issue_types = {issue.issue_type for issue in verification_issues}
        deterministic_anchor_unsafe = issue_types.intersection({
            "missed_critical_risk",
            "must_not_miss_gap",
            "candidate_conflict",
            "unresolved_hazard",
            "pregnancy_context_gap",
            "contraindicated_intervention",
        })
        anchor_mismatch_signal = self._detect_anchor_mismatch_signal(verification_issues)
        if deterministic_anchor_unsafe or anchor_mismatch_signal:
            self._apply_safety_penalty(
                differential,
                frontier,
                issue_types,
                anchor_mismatch=anchor_mismatch_signal,
            )

        if verification_result.error:
            self.typed_case_bundle["verification_error"] = dict(verification_result.error)
            await self._emit_trace(
                "verification",
                "Typed verifier returned no stable structured output; autonomous closure is blocked until objective confirmation resolves the case.",
                {
                    "error": verification_result.error,
                    "metrics": self._stage_metrics("verification", wall_time_s=time.perf_counter() - stage_started),
                },
            )
        elif verification_result.issues:
            await self._emit_trace(
                "verification",
                "Typed verifier attached safety issues to the shared state before closure.",
                {
                    "issues": [issue.model_dump(mode="json") for issue in verification_result.issues[:4]],
                    "decision_hint": verification_result.decision_hint,
                    "metrics": self._stage_metrics("verification", wall_time_s=time.perf_counter() - stage_started),
                },
            )
        else:
            await self._emit_trace(
                "verification",
                "Typed verifier found no new closure blockers; retaining the current adaptive differential.",
                {
                    "decision_hint": verification_result.decision_hint,
                    "metrics": self._stage_metrics("verification", wall_time_s=time.perf_counter() - stage_started),
                },
            )
        if (not verification_result.error) and verification_result.decision_hint in {"revise", "urgent_escalation", "abstain"}:

            loop_count = self._deep_reasoning_loop_count
            if self.deep_thinking and loop_count < 2 and verification_result.decision_hint in {"revise", "abstain", "urgent_escalation"}:
                self._deep_reasoning_loop_count += 1
                await self._emit_trace(
                    "verification",
                    f"⚕️ ATOM DEEP REASONING ACTIVATED: Diagnostic certainty low ({verification_result.decision_hint}). Expanding evidence search and forcing Swarm re-evaluation (Loop {loop_count + 1}/2).",
                    {"decision_hint": verification_result.decision_hint, "loop": loop_count + 1}
                )
                self._record_stage_profile(stage="verification", wall_time_s=time.perf_counter() - stage_started)
                self.state.transition(CdssPhase.R2)
                return

            # Only recover from gateway if we lost our specific hypotheses.
            # If we still have specific hypotheses (e.g. from the Swarm), we should retain them
            # rather than letting the gateway overwrite them with generic labels.
            has_specific = any(not self._is_generic_process_label(h) for h in self.state.working_hypotheses)
            if not has_specific:
                await self._recover_specific_differential(f"verifier_{verification_result.decision_hint}")
            else:
                await self._emit_trace(
                    "verification",
                    f"Retaining specific swarm hypotheses despite {verification_result.decision_hint} hint (avoiding generic downgrade).",
                    {"decision_hint": verification_result.decision_hint}
                )
        # W6.2 J.3 — persist worker trust EMA on confident, error-free closure.
        # Quality signal is "self-supervised": each worker scored against the
        # verifier-confirmed top-1 since gold labels are not available at runtime.
        try:
            if (
                bool(getattr(self.runtime_policy, "worker_trust_enabled", False))
                and not verification_result.error
                and verification_result.decision_hint not in {"revise", "urgent_escalation", "abstain"}
            ):
                _committed = ""
                _diff_now = self.typed_case_bundle.get("differential")
                if isinstance(_diff_now, DifferentialSet) and _diff_now.candidates:
                    _top = _diff_now.candidates[0]
                    if float(getattr(_top, "score", 0.0) or 0.0) >= 0.55:
                        _committed = normalize_candidate_label(_top.label) or str(_top.label or "").strip()
                if _committed:
                    panel = list(self.typed_case_bundle.get("last_swarm_panel", []) or [])
                    per_profile = dict(self.typed_case_bundle.get("last_swarm_per_profile_top3", {}) or {})
                    if panel and per_profile:
                        # Top family for the (profile, family)-keyed trust block.
                        _belief = self.typed_case_bundle.get("diagnostic_belief")
                        _top_fam = ""
                        try:
                            if _belief is not None and getattr(_belief, "family_posterior", None):
                                _top_fam = max(_belief.family_posterior.items(), key=lambda kv: kv[1])[0]
                        except Exception:
                            _top_fam = ""
                        from src.cdss.reasoning.worker_trust import quality_score, update_panel
                        q_per = {
                            prof: quality_score(per_profile.get(prof, []), _committed)
                            for prof in panel
                        }
                        try:
                            update_panel(panel, _top_fam or "_global", q_per)
                        except Exception as _wt_exc:
                            logging.getLogger("rrrie-cdss").warning(
                                "[WORKER_TRUST] update_panel failed: %s", _wt_exc
                            )
        except Exception as _wt_outer:
            logging.getLogger("rrrie-cdss").debug("[WORKER_TRUST] skipped: %s", _wt_outer)

        self._record_stage_profile(stage="verification", wall_time_s=time.perf_counter() - stage_started)
        self.state.transition(CdssPhase.ACTION_PLAN)
        return

    def _detect_anchor_mismatch_signal(self, verification_issues) -> dict | None:
        """Q1: scan verifier free-text for anchor-mismatch phrasing and extract
        the named alternative diagnosis. Returns None if no signal is found.

        The verifier LLM tends to write detail strings like:
          "anchor matches patient self-label; alternative <label> better explains ..."
          "<symptom> not explained by [<anchor>] - consider [<alternative>] ..."
        These carry strong clinical judgment that previously did not flow into
        ranking. The parser is conservative: it requires either explicit
        "alternative" / "consider" / "better explains" phrasing AND a parseable
        candidate label. Returns dict with `alternative_label` and `severity`.
        """
        import re as _re

        if not verification_issues:
            return None
        anchor_kw = (
            "anchor matches patient self-label",
            "anchor matches patient",
            "patient self-label",
            "anchor diagnosis",
            "self-diagnosis",
        )
        better_kw = ("better explains", "better explain")
        not_explained_kw = ("not explained by", "fails to explain")
        # Prefer explicit "consider [X]" / "alternative [X] better explains" forms.
        # These are unambiguous: the bracketed term is the named alternative.
        consider_alt_re = _re.compile(
            r"(?:consider|alternative(?:\s+diagnosis)?)\s*\[?\s*([a-zA-Z][\w \-/_]{3,80}?)\s*\]?(?:\s+(?:better|which|as|that|to)\b|[.,;:])",
            _re.IGNORECASE,
        )
        # Anchor-side phrases describe the WRONG candidate (skip these brackets).
        anchor_side_re = _re.compile(
            r"(?:not\s+explained\s+by|fails\s+to\s+explain|self[\- ]label|anchor(?:\s+diagnosis)?|patient[\-' s]+self|suspected\s+\w+)\s*\[?\s*([^\]\.;]+?)\s*\]?\s*(?:[\-—–.,;:]|\bconsider\b|\balternative\b)",
            _re.IGNORECASE,
        )
        bracket_alt_re = _re.compile(r"\[([a-zA-Z][\w \-/():,]{3,160})\]", _re.IGNORECASE)
        best: dict | None = None
        for issue in verification_issues:
            detail = str(getattr(issue, "detail", "") or "").strip()
            if not detail:
                continue
            low = detail.lower()
            anchor_hit = any(kw in low for kw in anchor_kw)
            better_hit = any(kw in low for kw in better_kw)
            not_explained_hit = any(kw in low for kw in not_explained_kw)
            if not (anchor_hit or better_hit or not_explained_hit):
                continue
            # Collect anchor-side phrases first so we can filter them out of bracket matches.
            anchor_side_terms = {
                str(m or "").strip().lower().strip(".,;:")
                for m in anchor_side_re.findall(detail)
            }
            generic_alt_phrases = {
                "alternative diagnosis", "alternative", "diagnosis", "differential",
                "the alternative", "another diagnosis",
            }
            def _strip_alt_prefix(text: str) -> str:
                t = str(text or "").strip().strip(".,;:")
                tl = t.lower()
                for pref in ("alternative diagnosis:", "alternative:", "diagnosis:", "consider:"):
                    if tl.startswith(pref):
                        t = t[len(pref):].strip().strip(".,;:")
                        break
                # Remove parenthetical explanations like "lateral medullary syndrome (Wallenberg syndrome)".
                t = _re.sub(r"\s*\([^)]{1,60}\)", "", t).strip()
                # Truncate at "or" / "/" disjunctions — pick the first listed alternative.
                for sep in (" or ", " / ", "/", ";"):
                    if sep in t.lower():
                        t = t.split(sep, 1)[0] if sep in t else t.lower().split(sep, 1)[0]
                        break
                return t.strip().strip(".,;:")
            alternative_label = ""
            for match in consider_alt_re.findall(detail):
                cand = _strip_alt_prefix(match)
                cl = cand.lower()
                if cand and 4 <= len(cand) <= 80 and cl not in anchor_side_terms and cl not in generic_alt_phrases:
                    alternative_label = cand
                    break
            if not alternative_label:
                # Fallback: pick the LAST bracketed term that is NOT an anchor-side
                # phrase (LLM convention: "not explained by [anchor] - consider [alt]").
                bracket_hits = [_strip_alt_prefix(b) for b in bracket_alt_re.findall(detail)]
                bracket_hits = [
                    b for b in bracket_hits
                    if b
                    and 4 <= len(b) <= 80
                    and "anchor" not in b.lower()
                    and b.lower() not in generic_alt_phrases
                    and not any(
                        side and (side in b.lower() or b.lower() in side)
                        for side in anchor_side_terms
                    )
                    and not any(
                        kw in b.lower() for kw in ("suspected", "self-label", "patient", "self diagnosis")
                    )
                ]
                if bracket_hits:
                    alternative_label = bracket_hits[-1]
            if not alternative_label:
                continue
            severity = str(getattr(issue, "severity", "") or "").lower()
            score = 2 if severity in {"major", "critical"} else 1
            if better_hit:
                score += 1
            if not_explained_hit:
                score += 1
            # Fix-F: collect *all* bracketed alternatives the verifier
            # named — not just the highest-scoring single one. When the
            # verifier writes "consider [epidural_hematoma] or
            # [diffuse_axonal_injury]" or surfaces multiple sibling
            # must-not-miss diagnoses across several issues, every named
            # candidate carries clinical signal that should flow into
            # ranking. Old behavior promoted at most one; the rest were
            # buried as warning text.
            secondary_pool: list[str] = []
            for raw in bracket_alt_re.findall(detail):
                cand = _strip_alt_prefix(raw)
                cl = cand.lower()
                if (
                    cand
                    and 4 <= len(cand) <= 80
                    and "anchor" not in cl
                    and cl not in generic_alt_phrases
                    and cl != alternative_label.lower()
                    and not any(
                        side and (side in cl or cl in side)
                        for side in anchor_side_terms
                    )
                    and not any(
                        kw in cl for kw in ("suspected", "self-label", "patient", "self diagnosis")
                    )
                ):
                    secondary_pool.append(cand)
            if best is None or score > int(best.get("score", 0) or 0):
                best = {
                    "alternative_label": alternative_label,
                    "severity": severity,
                    "score": score,
                    "detail": detail[:240],
                    "issue_type": str(getattr(issue, "issue_type", "") or ""),
                    "secondary_alternatives": [],
                }
            # Merge secondaries into the *current* best (de-duped).
            if best is not None:
                seen_secondaries = {
                    s.lower() for s in best.get("secondary_alternatives", [])
                }
                seen_secondaries.add(best["alternative_label"].lower())
                for s in secondary_pool:
                    if s.lower() not in seen_secondaries:
                        best.setdefault("secondary_alternatives", []).append(s)
                        seen_secondaries.add(s.lower())
        # Cap to top-3 secondary alternatives so we never explode the
        # differential beyond [:5] truncation room.
        if best is not None:
            best["secondary_alternatives"] = best.get("secondary_alternatives", [])[:3]
        return best

    def _apply_safety_penalty(
        self,
        differential: DifferentialSet,
        frontier: HypothesisFrontier,
        issue_types: set[str],
        *,
        anchor_mismatch: dict | None = None,
    ) -> None:
        """Step 4: Safety State Feedback - lower the anchor's score and elevate
        the safer alternative when the verifier flags an unsafe closure or
        names a better-fitting hypothesis. ``anchor_mismatch`` carries the
        verifier-named alternative when free-text parsing succeeded."""
        if not frontier.hypotheses or not differential.candidates:
            return

        anchor = frontier.anchor_hypothesis or differential.candidates[0].label
        # Magnitude scales with severity: must-not-miss > anchor mismatch > soft.
        if issue_types.intersection({"missed_critical_risk", "must_not_miss_gap"}):
            penalty_value = 0.50
        elif anchor_mismatch:
            penalty_value = 0.40
        else:
            penalty_value = 0.25

        replacement = None
        if anchor_mismatch:
            target = normalize_candidate_label(anchor_mismatch.get("alternative_label", "")) or ""
            if target:
                replacement = next(
                    (c for c in differential.candidates if normalize_candidate_label(c.label) == target and c.label != anchor),
                    None,
                )
                if replacement is None:
                    for c in differential.candidates:
                        if c.label == anchor:
                            continue
                        if target in normalize_candidate_label(c.label) or normalize_candidate_label(c.label) in target:
                            replacement = c
                            break

        if replacement is None:
            # Fall back to the original heuristic: prefer an explicit
            # must-not-miss alternative, else any candidate >= 0.25.
            replacement = next(
                (c for c in differential.candidates if c.label != anchor and (c.label in frontier.must_not_miss or c.score >= 0.25)),
                None
            )

        anchor_penalty_msg = (
            f"Penalized: verifier flagged anchor mismatch — alternative '{anchor_mismatch['alternative_label']}' better explains the symptom cluster."
            if anchor_mismatch
            else "Penalized due to critical safety blockers."
        )
        elevation_msg = (
            f"Elevated by verifier: explicitly named as the better-fitting alternative to '{anchor}'."
            if anchor_mismatch
            else "Elevated due to pending must-not-miss exclusion."
        )
        elevation_boost = 0.20 if anchor_mismatch else 0.15

        if replacement is None:
            # No safe alternative found — still penalize the unsafe anchor and flag for human review.
            # Absence of a replacement is NOT a reason to leave a safety-flagged anchor unpenalized.
            new_candidates = []
            anchor_post_penalty_score = 0.0
            for c in differential.candidates:
                if c.label == anchor:
                    anchor_post_penalty_score = max(0.10, c.score - penalty_value)
                    c = c.model_copy(update={
                        "score": anchor_post_penalty_score,
                        "rationale": c.rationale + [anchor_penalty_msg + " No safe alternative present in the differential — manual review required."],
                    })
                new_candidates.append(c)
            # Fix-D: when the verifier names a label that is NOT yet on the
            # slate, insert it at a score that GUARANTEES promotion above the
            # demoted anchor (clinical judgment from the verifier wins). Was
            # a fixed 0.30, which silently lost when the anchor stayed above
            # 0.30 even after penalty.
            if anchor_mismatch and not any(
                normalize_candidate_label(c.label) == normalize_candidate_label(anchor_mismatch["alternative_label"]) for c in new_candidates
            ):
                inserted_label = normalize_candidate_label(anchor_mismatch["alternative_label"]) or anchor_mismatch["alternative_label"]
                # Promotion floor: at least 0.55, and at least 5 % above the
                # demoted anchor so the verifier-named alternative is the new
                # leader after sort. Capped at 0.95 to stay below absolute
                # confidence.
                inserted_score = round(min(0.95, max(0.55, anchor_post_penalty_score * 1.05 + 0.20)), 4)
                new_candidates.append(
                    DifferentialCandidate(
                        label=inserted_label,
                        score=inserted_score,
                        rationale=[
                            f"Inserted by verifier feedback as the better-fitting alternative to anchor '{anchor}'. Promoted above demoted anchor so the verifier's clinical judgment becomes the leader rather than a buried warning.",
                            anchor_mismatch.get("detail", "")[:180],
                        ],
                        evidence_needed=[],
                        status="candidate",
                    )
                )
            # Fix-F: also insert any secondary alternatives the verifier
            # named, at descending scores below the primary insertion but
            # above the demoted anchor. Caps to 3 secondaries so the
            # truncation budget stays reasonable.
            if anchor_mismatch:
                secondaries = list(anchor_mismatch.get("secondary_alternatives", []) or [])[:3]
                existing_labels = {normalize_candidate_label(c.label) or c.label for c in new_candidates}
                step = 0.06
                for i, sec in enumerate(secondaries, start=1):
                    sec_norm = normalize_candidate_label(sec) or sec
                    if not sec_norm or sec_norm in existing_labels:
                        continue
                    sec_score = round(min(0.92, max(0.40, anchor_post_penalty_score * 1.05 + 0.20 - step * i)), 4)
                    new_candidates.append(
                        DifferentialCandidate(
                            label=sec_norm,
                            score=sec_score,
                            rationale=[
                                f"Inserted by verifier feedback as a secondary must-not-miss alternative to anchor '{anchor}'.",
                                anchor_mismatch.get("detail", "")[:180],
                            ],
                            evidence_needed=[],
                            status="candidate",
                        )
                    )
                    existing_labels.add(sec_norm)
            # Q2: dedup before storing so the inserted alternative does not
            # collide with an existing near-duplicate label.
            new_candidates = _dedup_differential_candidates(new_candidates)
            new_candidates = sorted(new_candidates, key=lambda x: x.score, reverse=True)
            self.typed_case_bundle["differential"] = differential.model_copy(update={"candidates": new_candidates[:5]})
            return

        # Fix-D: when verifier explicitly names an alternative
        # (anchor_mismatch is set), guarantee the elevated replacement
        # ends up above the demoted anchor. The original additive
        # boost (anchor−0.40, replacement+0.20) could leave the
        # replacement still below the anchor when the anchor started
        # high and the replacement low — exactly the scenario that
        # produced the rendered "anchor wins" outcome despite a
        # correct verifier judgment.
        anchor_post_penalty_score = max(0.01, float(differential.candidates[0].score or 0.0) - penalty_value)
        new_candidates = []
        for c in differential.candidates:
            if c.label == anchor:
                anchor_post_penalty_score = max(0.01, c.score - penalty_value)
                c = c.model_copy(update={"score": anchor_post_penalty_score, "rationale": c.rationale + [anchor_penalty_msg]})
            elif c.label == replacement.label:
                if anchor_mismatch:
                    # Verifier-named alternative: elevation floor =
                    # demoted-anchor × 1.10 + 0.05, never below 0.55.
                    elevated = round(min(0.99, max(0.55, anchor_post_penalty_score * 1.10 + 0.05, c.score + elevation_boost)), 4)
                else:
                    elevated = min(0.99, c.score + elevation_boost)
                c = c.model_copy(update={"score": elevated, "rationale": c.rationale + [elevation_msg]})
            new_candidates.append(c)

        # Fix-F: replicate secondary-alternative injection in the
        # replacement-found path. The primary replacement has already been
        # elevated above the anchor; secondaries go in below it but
        # still above the demoted anchor so the verifier's full clinical
        # picture (e.g. "consider epidural OR DAI OR ICH") all surface as
        # candidates.
        if anchor_mismatch:
            secondaries = list(anchor_mismatch.get("secondary_alternatives", []) or [])[:3]
            existing_labels = {normalize_candidate_label(c.label) or c.label for c in new_candidates}
            step = 0.06
            for i, sec in enumerate(secondaries, start=1):
                sec_norm = normalize_candidate_label(sec) or sec
                if not sec_norm or sec_norm in existing_labels:
                    continue
                sec_score = round(min(0.90, max(0.40, anchor_post_penalty_score * 1.05 + 0.10 - step * i)), 4)
                new_candidates.append(
                    DifferentialCandidate(
                        label=sec_norm,
                        score=sec_score,
                        rationale=[
                            f"Inserted by verifier feedback as a secondary must-not-miss alternative to anchor '{anchor}'.",
                            anchor_mismatch.get("detail", "")[:180],
                        ],
                        evidence_needed=[],
                        status="candidate",
                    )
                )
                existing_labels.add(sec_norm)

        new_candidates = _dedup_differential_candidates(new_candidates)
        new_candidates = sorted(new_candidates, key=lambda x: x.score, reverse=True)
        self.typed_case_bundle["differential"] = differential.model_copy(update={"candidates": new_candidates[:5]})

        # Re-sync hypothesis frontier
        new_hypotheses = []
        score_map = {c.label: c.score for c in new_candidates}
        for h in frontier.hypotheses:
            new_hypotheses.append(h.model_copy(update={"score": score_map.get(h.label, h.score)}))
        new_hypotheses = sorted(new_hypotheses, key=lambda x: x.score, reverse=True)
        for i, h in enumerate(new_hypotheses, 1): h.rank = i

        self.typed_case_bundle["frontier"] = frontier.model_copy(update={"hypotheses": new_hypotheses, "anchor_hypothesis": new_hypotheses[0].label if new_hypotheses else ""})

    def _source_disease_candidates_from_safety_issues(self) -> list[str]:
        if self.typed_case_bundle.get("source_disease_resolved", None) is not False:
            return []
        try:
            from src.cdss.knowledge.registry import load_syndrome_registry

            registry = load_syndrome_registry()
        except Exception:
            registry = None

        issue_texts: list[str] = []
        for issue in list(getattr(self.state, "safety_issues", []) or []):
            issue_type = str(getattr(issue, "type", "") or getattr(issue, "code", "") or "").strip().lower()
            if issue_type not in {"missing_discriminator", "unsafe_closure", "missed_critical_risk"}:
                continue
            detail = str(getattr(issue, "detail", "") or "").strip()
            if detail:
                issue_texts.append(detail)
        if not issue_texts:
            return []

        import re

        issue_text = " ".join(issue_texts)
        lowered = issue_text.lower()
        candidates: list[str] = []
        if registry is not None:
            scored: list[tuple[int, str]] = []
            text_tokens = set(re.findall(r"[a-z0-9]{4,}", lowered))
            for profile in list(getattr(registry, "profiles", []) or []):
                label = str(getattr(profile, "label", "") or getattr(profile, "id", "") or "").strip()
                if not label:
                    continue
                role = str(getattr(profile, "clinical_role", "") or "").strip().lower()
                resolved = getattr(profile, "source_disease_resolved", None)
                if role and role != "source_disease" and resolved is not True:
                    continue
                label_tokens = set(re.findall(r"[a-z0-9]{4,}", label.replace("_", " ").lower()))
                summary_tokens = set(re.findall(r"[a-z0-9]{4,}", str(getattr(profile, "summary", "") or "").lower()))
                overlap = len(text_tokens & (label_tokens | summary_tokens))
                if overlap > 0 and label.lower() not in lowered:
                    overlap += 1
                if overlap >= 2:
                    scored.append((overlap, label))
            scored.sort(key=lambda item: (-item[0], item[1]))
            for _, label in scored[:5]:
                if label not in candidates:
                    candidates.append(label)

        if candidates:
            return candidates[:3]

        raw_chunks = re.split(r"\bvs\.?\b|\bor\b|,|;|\(|\)", issue_text, flags=re.IGNORECASE)
        for chunk in raw_chunks:
            text = " ".join(str(chunk or "").split()).strip(" .:-")
            if not text or len(text) < 8 or len(text) > 64:
                continue
            label = normalize_candidate_label(text) or ""
            if not label:
                continue
            if label not in candidates:
                candidates.append(label)
            if len(candidates) >= 3:
                break
        return candidates[:3]

    async def _deterministic_action_summary(self) -> str:
        findings = self._ensure_typed_findings()
        differential = self.typed_case_bundle.get("differential")
        evidence = self._typed_evidence()
        risk_profile = self._typed_risk_profile()
        top_candidates: list[str] = []
        if isinstance(differential, DifferentialSet):
            top_candidates = [
                str(candidate.label or "").strip()
                for candidate in differential.candidates[:3]
                if str(candidate.label or "").strip() and not self._is_generic_process_label(candidate.label)
            ]
        if not top_candidates:
            top_candidates = [
                str(label or "").strip()
                for label in self.state.working_hypotheses[:3]
                if str(label or "").strip() and not self._is_generic_process_label(label)
            ]
        critical_signals = list(
            dict.fromkeys(
                [
                    *[str(item).strip() for item in findings.red_flags[:3] if str(item).strip()],
                    *[str(item).strip() for item in risk_profile.escalation_reasons[:3] if str(item).strip()],
                ]
            )
        )[:3]
        gateway_failures = [
            key.replace("_error", "")
            for key in ("research_error", "differential_error", "mechanism_error", "challenge_error", "verification_error")
            if isinstance(self.typed_case_bundle.get(key), dict) and self.typed_case_bundle.get(key)
        ]
        summary_parts: list[str] = []
        if gateway_failures:
            summary_parts.append(
                "Structured model output was incomplete in "
                + ", ".join(gateway_failures)
                + ", so autonomous closure remained blocked."
            )
        if top_candidates:
            lead = top_candidates[0].replace("_", " ")
            try:
                from src.cdss.runtime.canonical_state import build_canonical_state

                typed_bundle = dict(getattr(self.state, "typed_case_bundle", {}) or {})
                safety_issue_details = [
                    str(getattr(issue, "detail", issue) or "").strip()
                    for issue in (getattr(self.state, "safety_issues", None) or [])
                    if str(getattr(issue, "detail", issue) or "").strip()
                ]
                patient_text = ""
                for attr_name in ("patient_text", "patient_story", "case_text", "raw_case_text", "narrative"):
                    candidate_text = getattr(self.state, attr_name, "")
                    if candidate_text:
                        patient_text = str(candidate_text)
                        break
                source_candidates = list(typed_bundle.get("source_disease_candidates", []) or [])
                if not source_candidates and patient_text:
                    try:
                        from src.cdss.runtime.mainline_source_resolution import resolve_source_disease_mainline

                        resolution = await asyncio.wait_for(
                            asyncio.shield(asyncio.to_thread(
                                resolve_source_disease_mainline,
                                patient_text=patient_text,
                                phenotype_anchor=str(top_candidates[0] or ""),
                                phenotype_candidates=list(getattr(self.state, "differential", []) or []),
                                safety_issue_details=safety_issue_details,
                                research_evidence=list(typed_bundle.get("linked_evidence", []) or []),
                            )),
                            timeout=10.0
                        )
                        source_candidates = list(resolution.candidates or [])
                        if resolution.rationale:
                            typed_bundle["source_resolution_rationale"] = resolution.rationale
                    except Exception:
                        source_candidates = list(source_candidates or [])
                canonical_state = build_canonical_state(
                    anchor=typed_bundle.get("anchor") or top_candidates[0],
                    anchor_role=typed_bundle.get("anchor_role"),
                    source_disease_resolved=typed_bundle.get("source_disease_resolved"),
                    phenotype_candidates=list(getattr(self.state, "differential", []) or []),
                    source_disease_candidates=source_candidates,
                    linked_evidence=typed_bundle.get("linked_evidence", []),
                    required_data=typed_bundle.get("required_data", []),
                    objective_discriminators=typed_bundle.get("objective_discriminators", []),
                    must_not_miss=typed_bundle.get("must_not_miss", []),
                    contraindications=typed_bundle.get("contraindications", []),
                    urgency=typed_bundle.get("urgency", ""),
                    resolution_rationale=typed_bundle.get("source_resolution_rationale", ""),
                )
                typed_bundle["anchor"] = canonical_state.resolved_anchor or typed_bundle.get("anchor") or top_candidates[0]
                typed_bundle["anchor_role"] = canonical_state.anchor_role
                typed_bundle["source_disease_resolved"] = canonical_state.source_disease_resolved
                typed_bundle["required_data"] = list(canonical_state.required_data)
                typed_bundle["linked_evidence"] = [item.__dict__ for item in canonical_state.linked_evidence]
                typed_bundle["research_status"] = canonical_state.research_status
                typed_bundle["source_disease_candidates"] = [item.__dict__ for item in canonical_state.source_disease_candidates]
                if canonical_state.linked_evidence:
                    self.state.reasoning_trace.append(
                        "Canonical trusted research attached external evidence to the active diagnostic state."
                    )
                elif canonical_state.research_status == "blocked":
                    self.state.reasoning_trace.append(
                        "Canonical trusted research remained blocked because no structured external evidence was attached."
                    )
                self.state.typed_case_bundle = typed_bundle
                if canonical_state.resolved_anchor:
                    top_candidates[0] = canonical_state.resolved_anchor.replace(" ", "_")
                    lead = canonical_state.resolved_anchor
                    self.state.reasoning_trace.append(
                        "Canonical source-resolution stage promoted the upstream anchor before final synthesis."
                    )
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass
            try:
                from src.cdss.runtime.diagnostic_contract import build_contract_from_profile

                contract = build_contract_from_profile(top_candidates[0])
                typed_bundle = getattr(self.state, "typed_case_bundle", None)
                if isinstance(typed_bundle, dict):
                    if contract.anchor:
                        typed_bundle["anchor"] = contract.anchor
                    if contract.anchor_role is not None:
                        typed_bundle["anchor_role"] = contract.anchor_role
                        if (
                            contract.anchor_role != "source_disease"
                            and contract.source_disease_resolved is None
                        ):
                            typed_bundle["source_disease_resolved"] = False
                    if contract.source_disease_resolved is not None:
                        typed_bundle["source_disease_resolved"] = contract.source_disease_resolved
                    if contract.required_data:
                        typed_bundle["required_data"] = list(contract.required_data)
                    if "underlying_cause_resolution" in contract.required_data:
                        typed_bundle["source_disease_resolved"] = False
                    if contract.objective_discriminators:
                        typed_bundle["objective_discriminators"] = list(contract.objective_discriminators)
                    if contract.must_not_miss:
                        typed_bundle["must_not_miss"] = list(contract.must_not_miss)
                    if contract.contraindications:
                        typed_bundle["contraindications"] = list(contract.contraindications)
                    source_candidates: list[str] = []
                    try:
                        source_candidates = list(self._source_disease_candidates_from_safety_issues() or [])
                    except Exception:  # noqa: BLE001
                        logger.debug("source-disease candidate derivation failed", exc_info=True)
                        source_candidates = []
                    if source_candidates:
                        typed_bundle["source_disease_candidates"] = list(source_candidates)
                        if (
                            contract.anchor_role != "source_disease"
                            or contract.source_disease_resolved is False
                        ):
                            promoted_source = source_candidates[0]
                            typed_bundle["anchor"] = promoted_source
                            typed_bundle["anchor_role"] = "source_disease"
                            typed_bundle["source_disease_resolved"] = False
                            top_candidates = [promoted_source] + [
                                item for item in top_candidates if item != promoted_source
                            ]
                            # Bug-fix: previously referenced an undefined
                            # `frontier` local — the assignment silently
                            # raised NameError and was swallowed, so the
                            # source-disease promotion never reached the
                            # frontier anchor. Read from the typed bundle,
                            # mutate, write back.
                            try:
                                _frontier = self.typed_case_bundle.get("frontier")
                                if _frontier is not None and hasattr(_frontier, "model_copy"):
                                    self.typed_case_bundle["frontier"] = _frontier.model_copy(
                                        update={"anchor_hypothesis": promoted_source}
                                    )
                                elif _frontier is not None:
                                    setattr(_frontier, "anchor_hypothesis", promoted_source)
                            except Exception:  # noqa: BLE001
                                logger.debug("frontier anchor promotion failed", exc_info=True)
                            lead = top_candidates[0].replace("_", " ")
                    evidence_count = 0
                    for attr_name in (
                        "evidence_items",
                        "linked_evidence",
                        "retrieved_evidence",
                        "research_evidence",
                    ):
                        value = getattr(self.state, attr_name, None)
                        if isinstance(value, (list, tuple, set)):
                            evidence_count = max(evidence_count, len(value))
                    if evidence_count == 0:
                        _bundle_ev2 = typed_bundle.get("evidence")
                        if hasattr(_bundle_ev2, "items") and _bundle_ev2.items:
                            evidence_count = len(_bundle_ev2.items)
                    if contract.objective_discriminators or contract.required_data:
                        if contract.source_disease_resolved is None:
                            typed_bundle["source_disease_resolved"] = False
                        typed_bundle["research_status"] = "complete" if evidence_count else "blocked"
                        if evidence_count == 0:
                            typed_bundle["research_block_reason"] = "no_structured_trusted_research_evidence"
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass
            typed_bundle = getattr(self.state, "typed_case_bundle", None)
            if isinstance(typed_bundle, dict) and typed_bundle.get("research_status") == "blocked":
                summary_parts.append(
                    "Trusted web research is blocked until a structured planner query is produced."
                )
            anchor_role = ""
            source_disease_resolved = None
            try:
                from src.cdss.knowledge.registry import load_syndrome_registry

                anchor_profile = load_syndrome_registry().by_id(top_candidates[0])
            except Exception:
                anchor_profile = None
            if anchor_profile is not None:
                anchor_role = str(getattr(anchor_profile, "clinical_role", "") or "").strip().lower()
                source_disease_resolved = getattr(anchor_profile, "source_disease_resolved", None)
                self.typed_case_bundle["anchor_role"] = anchor_role
                self.typed_case_bundle["source_disease_resolved"] = source_disease_resolved
            if source_disease_resolved is None:
                for issue in list(getattr(self.state, "safety_issues", []) or []):
                    issue_type = str(getattr(issue, "type", "") or getattr(issue, "code", "") or "").strip().lower()
                    if issue_type == "missing_discriminator":
                        source_disease_resolved = False
                        self.typed_case_bundle["source_disease_resolved"] = False
                        break
            if source_disease_resolved is False:
                source_candidates = self._source_disease_candidates_from_safety_issues()
                if source_candidates:
                    self.typed_case_bundle["source_disease_candidates"] = source_candidates
                summary_parts.append(
                    "The leading anchor remains an unresolved surface state, so upstream source-disease resolution is still required."
                )
                if source_candidates:
                    summary_parts.append(
                        "Potential upstream source diseases remain open around: "
                        + ", ".join(item.replace("_", " ") for item in source_candidates)
                        + "."
                    )
            if len(top_candidates) > 1:
                summary_parts.append(
                    f"Current adaptive differential is led by {lead}, with {', '.join(item.replace('_', ' ') for item in top_candidates[1:3])} still requiring objective discrimination."
                )
            else:
                summary_parts.append(f"Current adaptive differential is led by {lead}, but objective confirmation is still required before safe closure.")
        elif findings.summary:
            summary_parts.append(f"The presentation remains open around this typed clinical story: {findings.summary[:220]}.")
        if critical_signals or risk_profile.urgency != UrgencyTier.ROUTINE:
            signal_text = ", ".join(critical_signals[:3]) if critical_signals else risk_profile.urgency.value.replace("_", " ")
            summary_parts.append(f"Given {signal_text}, urgent clinician review and rule-out testing are required instead of symptom-only discharge.")
        elif evidence.evidence_needs:
            next_data = [
                str(item.desired_discriminator or item.objective or "").strip().replace("_", " ")
                for item in evidence.evidence_needs[:3]
                if str(item.desired_discriminator or item.objective or "").strip()
            ]
            if next_data:
                summary_parts.append(f"Required next data: {', '.join(next_data)}.")
        return " ".join(part for part in summary_parts if part).strip() or "Adaptive state-machine review remains open and requires objective confirmation before autonomous closure."


    async def _run_action_plan(self):
        stage_started = time.perf_counter()
        self._set_profile_stage("action_plan")
        try:
            print(f"[{time.strftime('%H:%M:%S')}] [Action Agent] Outputting final typed clinical summary...\n")
            try:
                resp = await asyncio.wait_for(self._deterministic_action_summary(), timeout=45.0)
            except (asyncio.TimeoutError, Exception) as synthesis_err:
                print(f"[{time.strftime('%H:%M:%S')}] [Action Agent] Summary synthesis timed out or failed: {synthesis_err.__class__.__name__}. Using emergency ejection.")
                findings = self._ensure_typed_findings()
                risk = self._typed_risk_profile()
                resp = (findings.summary or "Clinical review required.") + (
                    f" Urgency: {risk.urgency.value.replace('_', ' ')}." if risk and risk.urgency else ""
                )

            await self._emit_trace(
                "ie_override",
                "Generating action plan and final clinical recommendations.",
                {"metrics": self._stage_metrics("action_plan", wall_time_s=time.perf_counter() - stage_started)}
            )
            self._record_stage_profile(stage="action_plan", wall_time_s=time.perf_counter() - stage_started)
            self.state.final_plan = resp

            # Neural cognitive learning: record case embedding for future outcome feedback
            _mem_written = 0
            try:
                from src.cdss.learning.cognitive_engine import get_cognitive_engine
                _engine = get_cognitive_engine()
                _findings = self._ensure_typed_findings()
                _diff = self.typed_case_bundle.get("differential")
                _top_label = (
                    _diff.candidates[0].label
                    if _diff and getattr(_diff, "candidates", None)
                    else "unknown"
                )
                _engine.record_case(
                    case_id=str(self.state.patient_id or ""),
                    findings_summary=_findings.summary or "",
                    top_candidate=_top_label,
                )
                _mem_written = 1
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass

            await self._emit_trace(
                "memory_commit",
                "Learning memory committed.",
                {"records_written": _mem_written},
            )
        finally:
            print(f"[{time.strftime('%H:%M:%S')}] [Action Agent] Transitioning to DONE phase.\n")
            self.state.transition(CdssPhase.DONE)
