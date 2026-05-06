"""Typed case-state container and patch application helpers."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.cdss.contracts.models import (
    R1Decomposition,
    R2EvidenceMath,
    R3CausalConflict,
    IEOverride,
    ComplexityAssessment,
    DecisionTrace,
    DifferentialSet,
    DiseaseHypothesis,
    EvidenceBundle,
    ExplanationGraph,
    FactGraph,
    FamilyHypothesis,
    GeneratedCandidateProfile,
    HypothesisSlate,
    HypothesisFrontier,
    HypothesisLedger,
    InterventionSafetyAssessment,
    InterventionSet,
    LearnedSafetyAssessment,
    LearningPromotionRecord,
    LoopDirective,
    MechanismFrame,
    ModelSupportSignals,
    OODAssessment,
    PatientInput,
    ReliabilitySignals,
    RetrospectiveRecord,
    RetrievalRankingStats,
    RiskProfile,
    SimulationScorecard,
    StructuredFindings,
    VerificationReport,
)


class CaseState(BaseModel):
    """Internal case state shared across vNext agents."""

    model_config = ConfigDict(extra="ignore")

    case_id: str
    patient_input: PatientInput
    status: str = "new"
    active_stage: str = "intake"
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
    safety_assessment: LearnedSafetyAssessment = Field(default_factory=LearnedSafetyAssessment)
    missing_information: list[str] = Field(default_factory=list)
    contradictions_detected: list[str] = Field(default_factory=list)
    dangerous_exclusions: list[str] = Field(default_factory=list)
    abstention_reason: str = ""
    escalation_reason: str = ""
    evidence_quality_score: float = 0.0
    specificity_score: float = 0.0
    coverage_score: float = 0.0
    session_isolation_id: str = Field(default_factory=lambda: str(uuid4()))
    engine_mode: str = "local_qwen"
    engine_model: str = ""
    mode_lock_violation: bool = False
    trace: list[DecisionTrace] = Field(default_factory=list)


class StatePatch(BaseModel):
    """Partial state update emitted by agents and policies."""

    model_config = ConfigDict(extra="ignore")

    status: str | None = None
    active_stage: str | None = None
    fact_graph: FactGraph | None = None
    interventions: InterventionSet | None = None
    findings: StructuredFindings | None = None
    risk_profile: RiskProfile | None = None
    mechanism_frames: list[MechanismFrame] | None = None
    family_hypotheses: list[FamilyHypothesis] | None = None
    disease_hypotheses: list[DiseaseHypothesis] | None = None
    generated_candidate_profiles: list[GeneratedCandidateProfile] | None = None
    simulation_scorecards: list[SimulationScorecard] | None = None
    hypothesis_slate: HypothesisSlate | None = None
    hypothesis_frontier: HypothesisFrontier | None = None
    hypothesis_ledger: HypothesisLedger | None = None
    differential: DifferentialSet | None = None
    evidence: EvidenceBundle | None = None
    explanation_graph: ExplanationGraph | None = None
    intervention_safety: InterventionSafetyAssessment | None = None
    model_support: ModelSupportSignals | None = None
    ood_assessment: OODAssessment | None = None
    retrieval_stats: RetrievalRankingStats | None = None
    verification: VerificationReport | None = None
    reliability: ReliabilitySignals | None = None
    complexity: ComplexityAssessment | None = None
    loop_directive: LoopDirective | None = None
    r1_decomposition: R1Decomposition | None = None
    r2_evidence_math: R2EvidenceMath | None = None
    r3_causal_conflict: R3CausalConflict | None = None
    ie_override: IEOverride | None = None
    retrospective: RetrospectiveRecord | None = None
    learning_promotion: LearningPromotionRecord | None = None
    safety_assessment: LearnedSafetyAssessment | None = None
    missing_information: list[str] | None = None
    contradictions_detected: list[str] | None = None
    dangerous_exclusions: list[str] | None = None
    abstention_reason: str | None = None
    escalation_reason: str | None = None
    evidence_quality_score: float | None = None
    specificity_score: float | None = None
    coverage_score: float | None = None
    session_isolation_id: str | None = None
    engine_mode: str | None = None
    engine_model: str | None = None
    mode_lock_violation: bool | None = None
    trace: list[DecisionTrace] = Field(default_factory=list)


def begin_case_state(patient_input: PatientInput) -> CaseState:
    """Create a new internal case state from user input."""
    generated_case_id = patient_input.case_id or f"case-{uuid4().hex[:10]}"
    requested_mode = str(patient_input.execution_mode or "").strip().lower()
    if requested_mode not in {"local_qwen", "cloud_gemini"}:
        requested_mode = "local_qwen"
    return CaseState(case_id=generated_case_id, patient_input=patient_input, engine_mode=requested_mode)


def apply_state_patch(state: CaseState, patch: StatePatch) -> CaseState:
    """Apply a patch without mutating the previous state object."""
    updates: dict[str, object] = {}
    for field_name in (
        "status",
        "active_stage",
        "fact_graph",
        "interventions",
        "findings",
        "risk_profile",
        "mechanism_frames",
        "family_hypotheses",
        "disease_hypotheses",
        "generated_candidate_profiles",
        "simulation_scorecards",
        "hypothesis_slate",
        "hypothesis_frontier",
        "hypothesis_ledger",
        "differential",
        "evidence",
        "explanation_graph",
        "intervention_safety",
        "model_support",
        "ood_assessment",
        "retrieval_stats",
        "verification",
        "reliability",
        "complexity",
        "loop_directive",
        "r1_decomposition",
        "r2_evidence_math",
        "r3_causal_conflict",
        "ie_override",
        "retrospective",
        "learning_promotion",
        "safety_assessment",
        "missing_information",
        "contradictions_detected",
        "dangerous_exclusions",
        "abstention_reason",
        "escalation_reason",
        "evidence_quality_score",
        "specificity_score",
        "coverage_score",
        "session_isolation_id",
        "engine_mode",
        "engine_model",
        "mode_lock_violation",
    ):
        value = getattr(patch, field_name)
        if value is not None:
            updates[field_name] = value
    next_state = state.model_copy(update=updates)
    if patch.trace:
        next_state.trace = [*state.trace, *patch.trace]
    return next_state
