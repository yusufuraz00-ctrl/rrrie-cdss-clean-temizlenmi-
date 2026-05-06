"""Canonical typed contracts for the rebuilt CDSS core."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class UrgencyTier(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class DecisionStatus(str, Enum):
    PRELIMINARY = "preliminary"
    ACCEPT = "accept"
    REVISE = "revise"
    ABSTAIN = "abstain"
    MANUAL_REVIEW = "manual_review"
    URGENT_ESCALATION = "urgent_escalation"


class ExecutionMode(str, Enum):
    LOCAL_QWEN = "local_qwen"
    CLOUD_GEMINI = "cloud_gemini"


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str = ""
    patient_text: str
    language: str = ""
    demographics: dict[str, Any] = Field(default_factory=dict)
    vitals: dict[str, Any] = Field(default_factory=dict)
    labs: dict[str, Any] = Field(default_factory=dict)
    medications: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    source: str = "interactive"
    execution_mode: str = ExecutionMode.LOCAL_QWEN.value


class ProvenanceEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_stage: str = ""
    source_type: str = "derived"
    source_texts: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    derivation: str = ""
    confidence_basis: str = ""


class FactNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str
    label: str
    category: str
    value: str = ""
    polarity: str = "positive"
    confidence: float = 0.0
    temporal: str = ""
    source_text: str = ""
    provenance: str = "narrative"
    provenance_envelope: ProvenanceEnvelope = Field(default_factory=ProvenanceEnvelope)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.0
    rationale: str = ""


class FactGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: list[FactNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    summary: str = ""


class InterventionNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str
    label: str
    status: str = "planned"
    confidence: float = 0.0
    source_text: str = ""
    class_hint: str = ""
    route: str = ""


class InterventionSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[InterventionNode] = Field(default_factory=list)


class StructuredFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    language: str = ""
    positive_findings: list[str] = Field(default_factory=list)
    negative_findings: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    exposures: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    demographics: dict[str, Any] = Field(default_factory=dict)
    derived_vitals: dict[str, Any] = Field(default_factory=dict)
    suspected_conditions: list[str] = Field(default_factory=list)
    planned_interventions: list[str] = Field(default_factory=list)
    input_context: list[str] = Field(default_factory=list)
    context_lanes: dict[str, list[str]] = Field(default_factory=dict)
    phenotype_atoms: list["PhenotypeAtom"] = Field(default_factory=list)
    phenotype_fingerprint: "PhenotypeFingerprint" = Field(default_factory=lambda: PhenotypeFingerprint())
    red_flags: list[str] = Field(default_factory=list)
    uncertainty_markers: list[str] = Field(default_factory=list)
    raw_segments: list[str] = Field(default_factory=list)
    constitutional_findings: list[str] = Field(default_factory=list)


class PhenotypeAtom(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    category: str = "phenotype"
    slot: str = ""
    value: str = ""
    normalized_value: str = ""
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    provenance: str = "heuristic"
    # Polarity of the finding. "positive" = present, "negative" = explicitly
    # denied/absent, "uncertain" = patient-uncertain or contradicted. Consumers
    # MUST respect this tag; mixing polarities silently is a correctness bug.
    polarity: str = "positive"


class PhenotypeFingerprint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slots: dict[str, list[str]] = Field(default_factory=dict)
    slot_evidence: dict[str, list[str]] = Field(default_factory=dict)
    # Slot-structured negatives. Parallel to `slots`/`slot_evidence`, populated
    # only from explicitly-denied findings so prototype matching can penalise
    # without contaminating the positive signal.
    negative_slots: dict[str, list[str]] = Field(default_factory=dict)
    negative_slot_evidence: dict[str, list[str]] = Field(default_factory=dict)
    axis_weights: dict[str, float] = Field(default_factory=dict)
    temporal_signature: list[str] = Field(default_factory=list)
    embedding_terms: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    contradiction_terms: list[str] = Field(default_factory=list)
    # Objective clinical context derived from patient demographics + vitals.
    # Consumed by match_prototypes() for data-driven demographic boost scoring.
    # Keys: sex, age, shock_index, complaint_domains (list[str]).
    clinical_context: dict = Field(default_factory=dict)


class AnchoringReport(BaseModel):
    """Detector output: did the top-1 fused candidate echo the patient's own
    self-diagnosis lane? Plan Item 7 (cross-case engineering improvements).
    """

    model_config = ConfigDict(extra="ignore")

    is_anchored: bool = False
    score: float = 0.0
    matched_phrase: str = ""
    top_label: str = ""
    rationale: str = ""


class PrototypeMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    family_label: str = ""
    source: str = "registry"
    slots: dict[str, list[str]] = Field(default_factory=dict)
    slot_evidence: dict[str, list[str]] = Field(default_factory=dict)
    axis_weights: dict[str, float] = Field(default_factory=dict)
    mechanism_signature: list[str] = Field(default_factory=list)
    discriminator_set: list[str] = Field(default_factory=list)
    wrong_treatment_risks: list[str] = Field(default_factory=list)
    embedding_terms: list[str] = Field(default_factory=list)
    canonical_examples: list[str] = Field(default_factory=list)


class PrototypeMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prototype_id: str
    label: str
    family_label: str = ""
    similarity: float = 0.0
    slot_overlap: float = 0.0
    axis_overlap: float = 0.0
    token_overlap: float = 0.0
    matched_slots: dict[str, list[str]] = Field(default_factory=dict)
    missing_slots: dict[str, list[str]] = Field(default_factory=dict)
    evidence_spans: list[str] = Field(default_factory=list)
    mechanism_paths: list[str] = Field(default_factory=list)
    discriminator_set: list[str] = Field(default_factory=list)
    wrong_treatment_risks: list[str] = Field(default_factory=list)
    source: str = "prototype_memory"


class PrototypeUpdateSuggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str = "shadow"
    target_prototype_id: str = ""
    candidate_label: str = ""
    similarity: float = 0.0
    rationale: str = ""
    fingerprint_summary: list[str] = Field(default_factory=list)
    requires_review: bool = True


class RiskProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    urgency: UrgencyTier = UrgencyTier.ROUTINE
    score: float = 0.0
    escalation_reasons: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    workflow_guards: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    vital_alerts: list[str] = Field(default_factory=list)
    manual_review_required: bool = False


class DifferentialCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    score: float = 0.0
    specificity_tier: str = "disease"
    rationale: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    status: str = "candidate"
    parent_category: str = ""
    grounding_score: float = 1.0
    grounding_unsupported: list[str] = Field(default_factory=list)
    grounding_verdict: str = "unchecked"


class GroundingVerdict(BaseModel):
    """Inline grounding verdict for a single candidate.

    `verdict` is one of: pass, demote, drop, unchecked.
    `score` is the supported-claim ratio in [0, 1].
    `unsupported_claims` are the rationale / evidence-need fragments that
    could not be traced back to the patient narrative.
    """

    model_config = ConfigDict(extra="ignore")

    verdict: str = "unchecked"
    score: float = 1.0
    unsupported_claims: list[str] = Field(default_factory=list)
    checked_claims: int = 0


class DifferentialSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[DifferentialCandidate] = Field(default_factory=list)
    rule_out_candidates: list[DifferentialCandidate] = Field(default_factory=list)
    uncertainty: float = 1.0
    strategy: str = "bootstrap"


class FamilyHypothesis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    dominant_mechanism: str = ""
    dominant_findings: list[str] = Field(default_factory=list)


class GeneratedCandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    family_label: str = ""
    summary: str = ""
    prototype_id: str = ""
    prototype_similarity: float = 0.0
    cue_lexicon: list[str] = Field(default_factory=list)
    mechanism_paths: list[str] = Field(default_factory=list)
    discriminator_set: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)
    challenge_queries: list[str] = Field(default_factory=list)
    unsafe_interventions: list[str] = Field(default_factory=list)
    contrastive_notes: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False
    dangerous_if_treated_as: str = ""


class DiseaseHypothesis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    family_label: str = ""
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    unsafe_interventions: list[str] = Field(default_factory=list)
    status: str = "candidate"


class SimulationScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    family_label: str = ""
    story_fit: float = 0.0
    trigger_fit: float = 0.0
    objective_fit: float = 0.0
    contradiction_debt: float = 0.0
    safety_debt: float = 0.0
    specificity_gain: float = 0.0
    must_not_miss_risk: float = 0.0
    overall: float = 0.0
    expected_findings: list[str] = Field(default_factory=list)
    missing_confirmers: list[str] = Field(default_factory=list)
    contrastive_rationale: list[str] = Field(default_factory=list)
    rival_differentiators: list[str] = Field(default_factory=list)


class HypothesisSlateItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False
    dangerous_if_treated_as: str = ""
    source: str = "llm_or_hybrid"
    parent_category: str = ""


class HypothesisSlate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypotheses: list[HypothesisSlateItem] = Field(default_factory=list)
    strategy: str = "free_slate"


class HypothesisNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str
    label: str
    score: float = 0.0
    rank: int = 0
    rationale: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False
    dangerous_if_treated_as: str = ""
    source: str = "llm_or_hybrid"
    canonical_label: str = ""
    mapped: bool = False
    parent_category: str = ""


class HypothesisFrontier(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypotheses: list[HypothesisNode] = Field(default_factory=list)
    frontier_entropy: float = 0.0
    strategy: str = "bootstrap"
    anchor_hypothesis: str = ""
    must_not_miss: list[str] = Field(default_factory=list)
    dangerous_if_treated_as: list[str] = Field(default_factory=list)
    loop_history: list[str] = Field(default_factory=list)


class HypothesisLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    support_count: int = 0
    refute_count: int = 0
    missing_discriminators: list[str] = Field(default_factory=list)
    elimination_reason: str = ""
    score_timeline: list[float] = Field(default_factory=list)


class HypothesisLedger(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entries: list[HypothesisLedgerEntry] = Field(default_factory=list)


class CausalEdge(BaseModel):
    """One directed causal-pathway edge in a `MechanismGraph` (W4 Module F.1).

    `from_node` → `to_node` with an optional `weight ∈ [0, 1]` and a
    `pathway_ref` tag pointing at a curated pathway asset
    (`data/cdss/knowledge/pathway_edges.json`). Edges are Pearl-style structural
    relations; causal-do interventions sever all incoming edges of the target
    when computing counterfactual reachability.
    """

    model_config = ConfigDict(extra="ignore")

    from_node: str
    to_node: str
    weight: float = 1.0
    pathway_ref: str = ""


class MechanismGraph(BaseModel):
    """Typed DAG across active hypotheses (W4 Module F.1).

    Nodes are free-form strings (hypothesis ids, intermediate mechanisms like
    `hypoperfusion`, findings, or test results). `edges` is a flat list of
    `CausalEdge`. Shared intermediates (e.g. `hypoperfusion` reached by both
    ACS and PE) are just nodes that appear as `to_node` for multiple edges —
    no special structure.

    Consumers:
      - `src.cdss.reasoning.causal_do.do(graph, intervention)` — sever incoming
        edges to intervened node; returns perturbed graph.
      - `src.cdss.reasoning.causal_do.explain_coverage(graph, hyp, findings)` —
        fraction of `findings` reachable from `hyp` over the graph.
      - robustness score = mean explain_coverage across single-finding
        ablations; feeds Bayes update in BACKWARD_SIMULATION.
    """

    model_config = ConfigDict(extra="ignore")

    nodes: list[str] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class MechanismFrame(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_state: str = ""
    organ_system: str = ""
    primary_mechanism: str = ""
    secondary_mechanism: str = ""
    critical_findings: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    hazard_context: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: str = "derived"
    # W4 Module F.1 — optional causal graph backing the frame.
    causal_edges: list[CausalEdge] = Field(default_factory=list)


class HypothesisMapperResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frontier: HypothesisFrontier = Field(default_factory=HypothesisFrontier)
    differential: DifferentialSet = Field(default_factory=DifferentialSet)
    mapper_notes: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    objective: str
    decision_target: str = "confirm"
    query_hint: str = ""
    target_candidate: str = ""
    active_state: str = ""
    unresolved_critical_finding: str = ""
    rival_mechanism: str = ""
    action_hazard: str = ""
    desired_discriminator: str = ""
    priority: float = 0.0
    decision_relevance: float = 0.0
    expected_value: float = 0.0


class EvidenceNeed(BaseModel):
    model_config = ConfigDict(extra="ignore")

    objective: str
    decision_target: str = "confirm"
    rationale: str = ""
    hypothesis_label: str = ""
    priority: float = 0.0
    query_hint: str = ""
    active_state: str = ""
    unresolved_critical_finding: str = ""
    rival_mechanism: str = ""
    action_hazard: str = ""
    desired_discriminator: str = ""
    decision_relevance: float = 0.0
    expected_information_gain: float = 0.0


class LoopDirective(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str = "none"
    target_stage: str = ""
    reason: str = ""
    priority: float = 0.0


class ClinicalScoreCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    score_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    risk_band: str = ""
    impact_on_decision: str = ""


class BlockedOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_name: str
    block_level: str = "hard_block"
    rationale: str = ""
    fatal_risk: str = ""


class MandatoryAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str
    priority: str = "urgent"
    deadline_hint: str = ""
    service_owner: str = ""


class R1Decomposition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs_used: list[str] = Field(default_factory=list)
    edge_case_decomposition: list[str] = Field(default_factory=list)
    temporal_findings: list[str] = Field(default_factory=list)
    demographic_hazards: list[str] = Field(default_factory=list)
    paradox_signals: list[str] = Field(default_factory=list)
    derived_signals: list[str] = Field(default_factory=list)
    decision_effect: str = ""
    missing_criticals: list[str] = Field(default_factory=list)


class R2EvidenceMath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs_used: list[str] = Field(default_factory=list)
    active_hypotheses: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    expected_impact: list[str] = Field(default_factory=list)
    guideline_anchors: list[str] = Field(default_factory=list)
    score_cards: list[ClinicalScoreCard] = Field(default_factory=list)
    derived_signals: list[str] = Field(default_factory=list)
    decision_effect: str = ""
    missing_criticals: list[str] = Field(default_factory=list)


class R3CausalConflict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs_used: list[str] = Field(default_factory=list)
    primary_causal_hypothesis: str = ""
    disease_vs_iatrogenic_conflict: bool = False
    contraindicated_by_causal_loop: list[str] = Field(default_factory=list)
    harm_if_follow_current_order: str = ""
    safe_alternative_pathway: list[str] = Field(default_factory=list)
    derived_signals: list[str] = Field(default_factory=list)
    decision_effect: str = ""
    missing_criticals: list[str] = Field(default_factory=list)


class IEOverride(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    mode: str = "silent"
    summary: str = ""
    critical_statement: str = ""
    blocked_orders: list[BlockedOrder] = Field(default_factory=list)
    mandatory_actions: list[MandatoryAction] = Field(default_factory=list)
    workflow_modifications: list[str] = Field(default_factory=list)
    requested_data: list[str] = Field(default_factory=list)
    inputs_used: list[str] = Field(default_factory=list)
    derived_signals: list[str] = Field(default_factory=list)
    decision_effect: str = ""
    missing_criticals: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    title: str = ""
    citation: str = ""
    excerpt: str = ""
    trust_score: float = 0.0
    pmid: str = ""
    doi: str = ""
    origin_query: str = ""
    linked_hypotheses: list[str] = Field(default_factory=list)
    relation_type: str = "neutral"
    verification_status: str = "unverified"


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[EvidenceItem] = Field(default_factory=list)
    coverage: float = 0.0
    contradiction_mass: float = 0.0
    retrieval_intents: list[RetrievalIntent] = Field(default_factory=list)
    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)


class ExplanationLink(BaseModel):
    model_config = ConfigDict(extra="ignore")

    finding: str
    explanation_type: str = "still_unexplained"
    target: str = ""
    confidence: float = 0.0
    rationale: str = ""
    critical: bool = False
    provenance_envelope: ProvenanceEnvelope = Field(default_factory=ProvenanceEnvelope)


class DetectiveAtom(BaseModel):
    model_config = ConfigDict(extra="ignore")

    atom_id: str
    label: str
    category: str
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class DetectiveEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_atom_id: str
    target_atom_id: str
    relation: str
    weight: float = 0.0
    rationale: str = ""


class ExplanationGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    links: list[ExplanationLink] = Field(default_factory=list)
    detective_atoms: list[DetectiveAtom] = Field(default_factory=list)
    detective_edges: list[DetectiveEdge] = Field(default_factory=list)
    detective_hubs: list[str] = Field(default_factory=list)
    explained_count: int = 0
    unexplained_count: int = 0
    critical_unexplained_count: int = 0
    coverage: float = 0.0
    state_frames: list[str] = Field(default_factory=list)
    unexplained_findings: list[str] = Field(default_factory=list)
    contradictory_findings: list[str] = Field(default_factory=list)
    dangerous_treatment_assumptions: list[str] = Field(default_factory=list)
    primary_process: str = ""
    secondary_processes: list[str] = Field(default_factory=list)
    epistemic_gaps: list[str] = Field(default_factory=list)
    story_summary: str = ""


class InterventionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intervention: str
    classification: str
    rationale: str = ""
    linked_hypotheses: list[str] = Field(default_factory=list)


class InterventionSafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decisions: list[InterventionDecision] = Field(default_factory=list)
    blocked_interventions: list[str] = Field(default_factory=list)
    allowed_interventions: list[str] = Field(default_factory=list)
    required_concurrent_actions: list[str] = Field(default_factory=list)
    unsafe_without_exclusion: list[str] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    severity: str
    issue_type: str
    detail: str


class GateEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gate_id: str
    score: float = 0.0
    threshold: float = 0.0
    passed: bool = False
    confidence_margin: float = 0.0
    remediable: bool = True
    rationale: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class UncertaintyAxes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    epistemic: float = 0.0
    aleatoric: float = 0.0
    distribution_shift: float = 0.0
    overall: float = 0.0


class ContradictionCluster(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cluster_id: str = ""
    hypothesis_labels: list[str] = Field(default_factory=list)
    finding_signals: list[str] = Field(default_factory=list)
    severity: str = "minor"
    resolution_cost: int = 1
    remediable: bool = True


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: DecisionStatus = DecisionStatus.PRELIMINARY
    reliability_score: float = 0.0
    epistemic_score: float = 0.0
    safety_score: float = 0.0
    evidence_coverage: float = 0.0
    contradiction_mass: float = 0.0
    issues: list[VerificationIssue] = Field(default_factory=list)
    recommended_disposition: str = ""
    diagnosis_gate: bool = False
    safety_gate: bool = False
    epistemic_gate: bool = False
    closure_gate: bool = False
    adaptive_profile: str = "default"
    decision_reason_category: str = "provisional"
    adaptive_thresholds: dict[str, float] = Field(default_factory=dict)
    gate_evaluations: list[GateEvaluation] = Field(default_factory=list)
    decision_path: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    uncertainty_axes: UncertaintyAxes = Field(default_factory=UncertaintyAxes)
    contradiction_clusters: list[ContradictionCluster] = Field(default_factory=list)
    safety_model_score: float = 0.0
    ie_judge_score: float = 0.0
    fusion_decision_basis: str = ""
    # W5 Module H — calibrated prediction set (conformal).
    conformal_prediction_set: list[str] = Field(default_factory=list)
    conformal_alpha: float = 0.0
    conformal_set_size: int = 0
    conformal_coverage_target: float = 0.0


class LearnedSafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    safety_model_score: float = 0.0
    must_not_miss_probability: float = 0.0
    treatment_harm_risk: float = 0.0
    contradiction_density: float = 0.0
    ood_risk: float = 0.0
    fused_decision: str = ""
    basis: list[str] = Field(default_factory=list)


class CandidateScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    proposer_score: float = 0.0
    llm_score: float = 0.0
    support_ml_score: float = 0.0
    retrieval_score: float = 0.0
    consistency_score: float = 0.0
    explanation_score: float = 0.0
    pairwise_score: float = 0.0
    state_fit: float = 0.0
    finding_fit: float = 0.0
    story_fit: float = 0.0
    simulation_score: float = 0.0
    contradiction_penalty: float = 0.0
    treatment_hazard_fit: float = 0.0
    explanation_debt: float = 0.0
    support_gate: float = 1.0
    safety_penalty: float = 0.0
    generic_penalty: float = 0.0
    final_score: float = 0.0
    learned_adjustment: float = 0.0
    epi_prior_penalty: float = 0.0


class ModelSupportSignals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_support: dict[str, float] = Field(default_factory=dict)
    calibrated_prob: dict[str, float] = Field(default_factory=dict)
    ood_score: float = 0.0
    low_risk_closure_support: float = 0.0
    support_reason_tags: list[str] = Field(default_factory=list)
    model_disagreement: float = 0.0
    helper_origin_share: float = 0.0
    effective_support_influence: float = 0.0
    support_proposer_correlation: float = 0.0
    score_breakdowns: dict[str, CandidateScoreBreakdown] = Field(default_factory=dict)
    feature_importance_stub: dict[str, float] = Field(default_factory=dict)


class OODAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ood_score: float = 0.0
    shift_features: list[str] = Field(default_factory=list)
    recommended_route: str = ""


class RetrievalRankingStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_encoder_used: str = ""
    cross_encoder_used: str = ""
    retrieved_count: int = 0
    reranked_count: int = 0
    specificity_gain: float = 0.0
    citation_alignment: float = 0.0
    novelty_gain: float = 0.0
    repeated_query_ratio: float = 0.0
    semantic_dedup_ratio: float = 0.0
    pairwise_discriminator_coverage: float = 0.0
    marginal_utility_score: float = 0.0
    marginal_utility_delta: float = 0.0
    utility_history: list[float] = Field(default_factory=list)
    query_hygiene_score: float = 0.0
    hazard_leakage_rate: float = 0.0
    hazard_leakage_detected: bool = False
    research_iterations: int = 0
    live_diagnostic_confidence: float = 0.0
    live_closure_confidence: float = 0.0
    simulation_alignment: float = 0.0
    simulation_scenarios: int = 0
    coverage_per_candidate: dict[str, float] = Field(default_factory=dict)
    evidence_starvation_flag: bool = False
    starved_candidates: list[str] = Field(default_factory=list)


class ReliabilitySignals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    semantic_confidence: float = 0.0
    frontier_entropy: float = 0.0
    evidence_coverage: float = 0.0
    contradiction_mass: float = 0.0
    intervention_risk_conflict: float = 0.0
    explanation_coverage: float = 0.0
    state_coherence: float = 0.0
    explanation_completeness: float = 0.0
    critical_unexplained_ratio: float = 0.0
    diagnostic_fit: float = 0.0
    action_safety: float = 0.0
    closure_readiness: float = 0.0
    evidence_sufficiency: float = 0.0
    distribution_stability: float = 0.0
    consensus_strength: float = 0.0
    stop_reason: str = ""
    stop_reasons: list[str] = Field(default_factory=list)
    overall: float = 0.0


class ComplexityAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    route: str = "local_primary"
    score: float = 0.0
    request_deep_review: bool = False
    reasons: list[str] = Field(default_factory=list)


class RetrospectiveRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str = ""
    failure_stage: str = ""
    root_cause: str = ""
    error_taxonomy: list[str] = Field(default_factory=list)
    lesson_candidate: str = ""
    promotion_eligibility: str = "shadow"
    status: str = "shadow"


class LearningPromotionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_type: str = ""
    shadow_metrics: dict[str, float] = Field(default_factory=dict)
    promotion_decision: str = "shadow"
    rollback_trigger: list[str] = Field(default_factory=list)
    prototype_updates: list[PrototypeUpdateSuggestion] = Field(default_factory=list)


class DecisionTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str = Field(default_factory=_utc_now)
    stage: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DecisionCandidateView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = ""
    family_label: str = ""
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    unsafe_interventions: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False


class SafetyStateView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: str = ""
    mode: str = ""
    summary: str = ""
    blocked_interventions: list[str] = Field(default_factory=list)
    blocked_orders: list[BlockedOrder] = Field(default_factory=list)
    mandatory_actions: list[MandatoryAction] = Field(default_factory=list)
    workflow_modifications: list[str] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)


class ConfidenceView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    diagnosis: float = 0.0
    closure: float = 0.0
    evidence: float = 0.0
    reliability: float = 0.0
    safety_model: float = 0.0
    ie_judge: float = 0.0
    unsupported_claims: int = 0
    zone: str = "critical"


class TimingView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_seconds: float = 0.0
    wall_seconds: float = 0.0
    llm_seconds: float = 0.0
    iterations: int = 1
    stages: dict[str, dict[str, float]] = Field(default_factory=dict)
    token_budget: dict[str, Any] = Field(default_factory=dict)


class EvidenceTraceView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    coverage: float = 0.0
    contradiction_mass: float = 0.0
    query_hygiene_score: float = 0.0
    hazard_leakage_detected: bool = False
    items: list[EvidenceItem] = Field(default_factory=list)
    retrieval_intents: list[RetrievalIntent] = Field(default_factory=list)


class CaseStateView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    status: str
    active_stage: str
    fact_graph: FactGraph = Field(default_factory=FactGraph)
    interventions: InterventionSet = Field(default_factory=InterventionSet)
    findings: StructuredFindings = Field(default_factory=StructuredFindings)
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)
    mechanism_frames: list[MechanismFrame] = Field(default_factory=list)
    family_hypotheses: list[FamilyHypothesis] = Field(default_factory=list)
    disease_hypotheses: list[DiseaseHypothesis] = Field(default_factory=list)
    generated_candidate_profiles: list[GeneratedCandidateProfile] = Field(default_factory=list)
    simulation_scorecards: list[SimulationScorecard] = Field(default_factory=list)
    hypothesis_slate: HypothesisSlate = Field(default_factory=HypothesisSlate)
    hypothesis_frontier: HypothesisFrontier = Field(default_factory=HypothesisFrontier)
    hypothesis_ledger: HypothesisLedger = Field(default_factory=HypothesisLedger)
    differential: DifferentialSet = Field(default_factory=DifferentialSet)
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    explanation_graph: ExplanationGraph = Field(default_factory=ExplanationGraph)
    intervention_safety: InterventionSafetyAssessment = Field(default_factory=InterventionSafetyAssessment)
    model_support: ModelSupportSignals = Field(default_factory=ModelSupportSignals)
    ood_assessment: OODAssessment = Field(default_factory=OODAssessment)
    retrieval_stats: RetrievalRankingStats = Field(default_factory=RetrievalRankingStats)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    reliability: ReliabilitySignals = Field(default_factory=ReliabilitySignals)
    complexity: ComplexityAssessment = Field(default_factory=ComplexityAssessment)
    loop_directive: LoopDirective = Field(default_factory=LoopDirective)
    r1_decomposition: R1Decomposition = Field(default_factory=R1Decomposition)
    r2_evidence_math: R2EvidenceMath = Field(default_factory=R2EvidenceMath)
    r3_causal_conflict: R3CausalConflict = Field(default_factory=R3CausalConflict)
    ie_override: IEOverride = Field(default_factory=IEOverride)
    retrospective: RetrospectiveRecord = Field(default_factory=RetrospectiveRecord)
    learning_promotion: LearningPromotionRecord = Field(default_factory=LearningPromotionRecord)
    missing_information: list[str] = Field(default_factory=list)
    contradictions_detected: list[str] = Field(default_factory=list)
    dangerous_exclusions: list[str] = Field(default_factory=list)
    abstention_reason: str = ""
    abstention_recommended: bool = False
    abstention_margin: float = 0.0
    abstention_grounding_risk: float = 0.0
    inline_grounding_pass_rate: float = 1.0
    escalation_reason: str = ""
    evidence_quality_score: float = 0.0
    specificity_score: float = 0.0
    coverage_score: float = 0.0
    session_isolation_id: str = ""
    engine_mode: str = "local_qwen"
    engine_model: str = ""
    mode_lock_violation: bool = False
    trace: list[DecisionTrace] = Field(default_factory=list)


class DecisionPacket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    status: DecisionStatus
    summary: str
    fact_graph: FactGraph = Field(default_factory=FactGraph)
    interventions: InterventionSet = Field(default_factory=InterventionSet)
    structured_findings: StructuredFindings
    risk_profile: RiskProfile
    mechanism_frames: list[MechanismFrame] = Field(default_factory=list)
    family_hypotheses: list[FamilyHypothesis] = Field(default_factory=list)
    disease_hypotheses: list[DiseaseHypothesis] = Field(default_factory=list)
    generated_candidate_profiles: list[GeneratedCandidateProfile] = Field(default_factory=list)
    simulation_scorecards: list[SimulationScorecard] = Field(default_factory=list)
    hypothesis_slate: HypothesisSlate = Field(default_factory=HypothesisSlate)
    hypothesis_frontier: HypothesisFrontier = Field(default_factory=HypothesisFrontier)
    hypothesis_ledger: HypothesisLedger = Field(default_factory=HypothesisLedger)
    frontier: HypothesisFrontier = Field(default_factory=HypothesisFrontier)
    differential: DifferentialSet
    evidence: EvidenceBundle
    explanation_graph: ExplanationGraph = Field(default_factory=ExplanationGraph)
    intervention_safety: InterventionSafetyAssessment = Field(default_factory=InterventionSafetyAssessment)
    model_support: ModelSupportSignals = Field(default_factory=ModelSupportSignals)
    ood_assessment: OODAssessment = Field(default_factory=OODAssessment)
    retrieval_stats: RetrievalRankingStats = Field(default_factory=RetrievalRankingStats)
    verification: VerificationReport
    reliability: ReliabilitySignals = Field(default_factory=ReliabilitySignals)
    complexity: ComplexityAssessment = Field(default_factory=ComplexityAssessment)
    loop_directive: LoopDirective = Field(default_factory=LoopDirective)
    r1_decomposition: R1Decomposition = Field(default_factory=R1Decomposition)
    r2_evidence_math: R2EvidenceMath = Field(default_factory=R2EvidenceMath)
    r3_causal_conflict: R3CausalConflict = Field(default_factory=R3CausalConflict)
    ie_override: IEOverride = Field(default_factory=IEOverride)
    anchor_hypothesis: str = ""
    must_not_miss: list[str] = Field(default_factory=list)
    blocked_interventions: list[str] = Field(default_factory=list)
    required_concurrent_actions: list[str] = Field(default_factory=list)
    loop_history: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    retrospective_stub: RetrospectiveRecord = Field(default_factory=RetrospectiveRecord)
    learning_promotion: LearningPromotionRecord = Field(default_factory=LearningPromotionRecord)
    recommended_actions: list[str] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    contradictions_detected: list[str] = Field(default_factory=list)
    dangerous_exclusions: list[str] = Field(default_factory=list)
    abstention_reason: str = ""
    abstention_recommended: bool = False
    abstention_margin: float = 0.0
    abstention_grounding_risk: float = 0.0
    inline_grounding_pass_rate: float = 1.0
    escalation_reason: str = ""
    evidence_quality_score: float = 0.0
    specificity_score: float = 0.0
    coverage_score: float = 0.0
    session_isolation_id: str = ""
    mode: str = "local"
    engine_mode: str = "local_qwen"
    engine_model: str = ""
    mode_lock_violation: bool = False
    diagnostic_confidence: float = 0.0
    closure_confidence: float = 0.0
    safety_model_score: float = 0.0
    ie_judge_score: float = 0.0
    fusion_decision_basis: str = ""
    memory_injection_policy: str = "validated_only"
    explanation: list[str] = Field(default_factory=list)
    trace: list[DecisionTrace] = Field(default_factory=list)


class DecisionPacketView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    packet_version: str = "decision_packet_view.v1"
    case_id: str
    summary: str
    decision: str
    top_differential: DecisionCandidateView = Field(default_factory=DecisionCandidateView)
    family: FamilyHypothesis = Field(default_factory=lambda: FamilyHypothesis(label=""))
    mechanism: MechanismFrame = Field(default_factory=MechanismFrame)
    disease_candidates: list[DecisionCandidateView] = Field(default_factory=list)
    required_data: list[str] = Field(default_factory=list)
    safety_state: SafetyStateView = Field(default_factory=SafetyStateView)
    confidence: ConfidenceView = Field(default_factory=ConfidenceView)
    timing: TimingView = Field(default_factory=TimingView)
    evidence_trace: EvidenceTraceView = Field(default_factory=EvidenceTraceView)
    reasoning_trace: list[str] = Field(default_factory=list)
    simulation_scorecards: list[SimulationScorecard] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
    abstention_recommended: bool = False
    abstention_reason: str = ""
    abstention_margin: float = 0.0
    abstention_grounding_risk: float = 0.0
    inline_grounding_pass_rate: float = 1.0
    evidence_starvation_flag: bool = False
    starved_candidates: list[str] = Field(default_factory=list)
    coverage_per_candidate: dict[str, float] = Field(default_factory=dict)


class SemanticParsePacket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    findings: StructuredFindings = Field(default_factory=StructuredFindings)
    fact_graph: FactGraph = Field(default_factory=FactGraph)
    interventions: InterventionSet = Field(default_factory=InterventionSet)


class HypothesisPacket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slate: HypothesisSlate = Field(default_factory=HypothesisSlate)
    frontier: HypothesisFrontier = Field(default_factory=HypothesisFrontier)
    differential: DifferentialSet = Field(default_factory=DifferentialSet)


class LearningRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    outcome: str = ""
    feedback: dict[str, Any] = Field(default_factory=dict)
    eligible_for_learning: bool = True


# ---------------------------------------------------------------------------
# Legacy fact-graph types (absorbed from cdss/memory/fact_graph.py)
# ---------------------------------------------------------------------------

class EvidenceNode(BaseModel):
    """A single fact node in the patient fact graph."""
    id: str = Field(..., description="Unique ID, e.g. 'fever_39'")
    type: str = Field("unknown", description="symptom|finding|diagnosis|treatment|demographic|evidence|vital|unknown")
    label: str = Field(..., description="Semantic label, e.g. 'High Fever'")
    confidence: float = Field(0.0, description="Confidence score 0-1")
    source: str = Field(..., description="Origin of this fact, e.g. 'patient_intake'")
    metadata: dict[str, str] = Field(default_factory=dict)


class SemanticEdge(BaseModel):
    """Directed relationship between two fact nodes."""
    source_id: str
    target_id: str
    relationship: str = Field("UNKNOWN", description="CAUSES|INDICATES|EXCLUDES|TREATS|COMPLICATES|UNKNOWN")
    weight: float = Field(1.0, description="Relationship strength")


class CdssPhase(str, Enum):
    INTAKE = "intake"
    R2 = "r2_planning"
    DIFFERENTIAL = "differential"
    COGNITIVE_LOOP = "cognitive_loop"
    BACKWARD_SIMULATION = "backward_simulation"
    OUTCOME_SIMULATION = "outcome_simulation"
    VERIFICATION = "verification"
    ACTION_PLAN = "action_plan"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Coarse-to-fine Bayesian diagnostic belief (W1 Module A)
# ---------------------------------------------------------------------------

class EvidenceDelta(BaseModel):
    """Audit record of one belief update."""

    model_config = ConfigDict(extra="ignore")

    source: str = ""  # swarm|causal_do|bma|mcts|user_input|prototype|...
    finding: str = ""
    step: int = 0
    level: int = 2  # 0=family, 1=genus, 2=species
    # Per-hypothesis likelihoods P(evidence | hypothesis) from Module B.
    likelihoods: dict[str, float] = Field(default_factory=dict)
    # Per-hypothesis likelihood variance (from temperature ensemble).
    likelihood_var: dict[str, float] = Field(default_factory=dict)
    posterior_before: dict[str, float] = Field(default_factory=dict)
    posterior_after: dict[str, float] = Field(default_factory=dict)
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    timestamp: str = Field(default_factory=_utc_now)
    meta: dict[str, Any] = Field(default_factory=dict)


class DiagnosticBelief(BaseModel):
    """Joint belief over 3-level disease hierarchy (family/genus/species).

    Posteriors are Dirichlet means; alphas carry conjugate counts (variance).
    Dempster-Shafer masses (Module E) are sparse — frozenset keys serialize as
    sorted tuples via dict[str, float] with '|'-joined hypothesis ids.
    """

    model_config = ConfigDict(extra="ignore")

    # Bayesian posteriors (normalized; sum ≈ 1.0 per level).
    family_posterior: dict[str, float] = Field(default_factory=dict)
    family_alpha: dict[str, float] = Field(default_factory=dict)
    genus_posterior: dict[str, float] = Field(default_factory=dict)
    genus_alpha: dict[str, float] = Field(default_factory=dict)
    species_posterior: dict[str, float] = Field(default_factory=dict)
    species_alpha: dict[str, float] = Field(default_factory=dict)
    # Dempster-Shafer mass function (sparse). Key = '|'-sorted hypothesis ids,
    # or the literal 'OMEGA' for total ignorance.
    ds_mass: dict[str, float] = Field(default_factory=dict)
    # Controller state.
    active_level: int = 0
    evidence_log: list[EvidenceDelta] = Field(default_factory=list)
    entropy_history: list[float] = Field(default_factory=list)  # H(b_S) per step
    miss_risk_history: list[float] = Field(default_factory=list)
    step: int = 0
    # Flag tripped by Module I meta-controller when termination conditions met.
    terminated: bool = False
    termination_reason: str = ""


class AgentMessage(BaseModel):
    agent_id: str
    content: str
    timestamp: str


class PatientFactGraph(BaseModel):
    """Patient-level fact graph (legacy state-machine format)."""
    patient_id: str
    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[SemanticEdge] = Field(default_factory=list)

    def get_missing_links(self) -> list[str]:
        """Return labels of finding nodes with no incoming edges (unexplained findings)."""
        incoming: set[str] = {edge.target_id for edge in self.edges}
        return [
            f"Unexplained finding: {node.label}"
            for node in self.nodes
            if node.type == "finding" and node.id not in incoming
        ]


# ---------------------------------------------------------------------------
# Reasoning artifact contracts (merged from contracts/reasoning_artifact.py)
# ---------------------------------------------------------------------------

class UncertaintyBand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    center: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    epistemic: float = 0.0
    aleatoric: float = 0.0
    model_disagreement: float = 0.0
    ood_score: float = 0.0


class DecisionGateAudit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    passed: bool = False
    score: float = 0.0
    reason: str = ""


class CandidateReasoning(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = ""
    score: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False
    support_count: int = 0
    refute_count: int = 0
    uncertainty: UncertaintyBand = Field(default_factory=UncertaintyBand)


class RequiredDataRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    category: str
    priority: str = "medium"
    rationale: str = ""
    target_hypothesis: str = ""


class ReasoningArtifactV2(BaseModel):
    model_config = ConfigDict(extra="ignore")

    packet_version: str = "reasoning_artifact.v2"
    case_id: str
    decision: str
    summary: str
    primary_candidate: CandidateReasoning = Field(default_factory=CandidateReasoning)
    differential: list[CandidateReasoning] = Field(default_factory=list)
    uncertainty: UncertaintyBand = Field(default_factory=UncertaintyBand)
    decision_gates: dict[str, DecisionGateAudit] = Field(default_factory=dict)
    required_data: list[RequiredDataRequest] = Field(default_factory=list)
    blocked_interventions: list[str] = Field(default_factory=list)
    required_concurrent_actions: list[str] = Field(default_factory=list)
    evidence_coverage: float = 0.0
    contradiction_mass: float = 0.0
    reasoning_trace: list[str] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Central state token passed between sub-agents in the State Machine."""

    patient_id: str
    raw_context: str = Field(default="", description="Raw medical intake text.")
    phase: CdssPhase = CdssPhase.INTAKE
    fact_graph: PatientFactGraph = Field(default_factory=lambda: PatientFactGraph(patient_id=""))
    working_hypotheses: list[str] = Field(default_factory=list)
    verification_queries: list[str] = Field(default_factory=list)
    final_plan: str | None = None
    agent_log: list[AgentMessage] = Field(default_factory=list)

    def transition(self, next_phase: CdssPhase) -> None:
        if not isinstance(next_phase, CdssPhase):
            raise ValueError(f"Invalid phase transition target: {next_phase!r}. Must be a CdssPhase enum member.")
        self.phase = next_phase
        self.agent_log.append(AgentMessage(agent_id="SYSTEM", content=f"Transitioned to {next_phase.value}", timestamp=""))
