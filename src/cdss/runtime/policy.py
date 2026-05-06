"""Runtime policy surface for the rebuilt local-first CDSS."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
import tomllib


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw if raw in allowed else default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    s = str(raw).strip()
    return s if s else default


def _settings_reasoning_mode() -> str:
    path = _PROJECT_ROOT / "data" / "settings.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(data.get("reasoning_mode", "") or "").strip().lower()


def _runtime_speed_profile() -> str:
    raw = str(os.getenv("CDSS_RUNTIME_SPEED_PROFILE", "") or "").strip().lower()
    if raw in {"fast", "balanced", "research"}:
        return raw
    mode = _settings_reasoning_mode()
    if mode == "deep":
        return "research"
    if mode == "thinking":
        return "balanced"
    return "fast"


def _fast_overrides_enabled(profile: str) -> bool:
    if profile != "fast":
        return False
    raw = str(os.getenv("CDSS_FAST_RESPECT_EXPENSIVE_ENV", "") or "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CdssRuntimePolicy:
    """Small, explicit runtime policy for the new CDSS surface."""

    external_evidence_enabled: bool
    pubmed_max_queries: int
    pubmed_max_results: int
    tavily_max_queries: int
    reactive_search_max_iterations: int
    icd11_max_queries: int
    learning_auto_record: bool
    learning_dir: Path
    local_llm_extraction_enabled: bool
    local_llm_max_tokens: int
    local_llm_temperature: float
    local_llm_ranker_enabled: bool
    llm_primary_reasoning_enabled: bool
    llm_guardrail_only_enabled: bool
    local_llm_verifier_enabled: bool
    local_llm_challenger_enabled: bool
    support_ml_enabled: bool
    retrieval_reranker_enabled: bool
    evidence_memory_enabled: bool
    support_influence_cap: float
    ood_revise_threshold: float
    max_reasoning_loops: int
    runtime_mode: str
    runtime_speed_profile: str
    gemini_enabled_in_cloud_mode: bool
    learned_safety_enabled: bool
    learned_label_quality_enabled: bool
    llm_label_validation_enabled: bool
    learned_label_quality_threshold: float
    memory_inference_policy: str
    open_world_expansion_enabled: bool
    registry_prior_enabled: bool
    static_pattern_map_enabled: bool
    evidence_confidence_requires_verified_edges: bool
    llm_research_planner_enabled: bool
    deterministic_mechanism_fallback_enabled: bool
    state_mismatch_rules_enabled: bool
    trigger_compatibility_rerank_enabled: bool
    research_min_evidence_coverage: float
    research_min_verified_edges: int
    reasoning_artifact_v2_enabled: bool
    llm_empty_slate_hard_fail_enabled: bool
    intervention_safety_pre_evidence_enabled: bool
    single_specificity_refine_pass_enabled: bool
    complexity_router_after_challenger_enabled: bool
    verification_adaptive_gates_enabled: bool
    verification_strict_issue_grounding_enabled: bool
    retrieval_pair_discriminator_enabled: bool
    retrieval_query_novelty_metrics_enabled: bool
    governor_retrieval_stall_early_stop_enabled: bool
    governor_utility_rebound_threshold: float
    governor_utility_score_floor: float
    llm_critical_stage_fail_fast_enabled: bool
    llm_sync_call_timeout_s: float
    epi_prior_enabled: bool
    epi_prior_pubmed_enabled: bool
    epi_prior_max_penalty: float
    specificity_calibrator_enabled: bool
    learning_score_feedback_enabled: bool
    # Coarse-to-fine Bayesian diagnostic agent (v2 extension). All OFF by default
    # until each wave passes replay on benchmark_log_new.txt.
    hierarchical_belief_enabled: bool  # W1.A: DiagnosticBelief + family prior LLM call
    bayesian_posterior_enabled: bool   # W1.B: Beta likelihoods + log-space Bayes update
    hierarchical_swarm_enabled: bool   # W2.C: coarse-to-fine L0→L1→L2 swarm
    dempster_shafer_fusion_enabled: bool  # W2.E
    cognitive_controller_enabled: bool    # W3.D: EVI + MCTS loop
    meta_controller_enabled: bool         # W3.I: termination/escalation/self-refine
    causal_do_verifier_enabled: bool      # W4.F: do-calculus mechanism verifier
    bma_outcome_enabled: bool             # W4.G: Bayesian-model-averaged utility
    conformal_output_enabled: bool        # W5.H: calibrated prediction set
    # Likelihood-ensemble tuning knobs.
    likelihood_ensemble_temps: str        # csv e.g. "0.0,0.2,0.4"
    likelihood_ensemble_max_tokens: int
    # MCTS controller budgets.
    mcts_max_ticks: int
    mcts_rollout_budget: int
    # Utility coefficients U(b) = -H - λ·miss_risk + μ·tightness - ν·budget.
    utility_miss_risk_weight: float
    utility_tightness_weight: float
    utility_budget_weight: float
    # Conformal miscoverage level (α). 0.1 → 90% coverage set.
    conformal_alpha: float
    # W6.1 Module J — swarm rank-fusion + label validation.
    swarm_rank_fusion_enabled: bool       # J.1: replace MAX with RRF + TWB + likelihood
    rank_fusion_alpha: float              # weight of RRF in convex mixture
    rank_fusion_beta: float               # weight of trust-weighted Borda
    rank_fusion_gamma: float              # weight of likelihood mean
    rank_fusion_rrf_k: int                # RRF k constant (Cormack default 60)
    swarm_label_validation_strict: bool   # J.6: drop labels that fail registry resolution at parse
    # Cross-case Plan Item 4 — multiplicative miss-cost prior on the fused
    # posterior. score' = score · (1 + δ · severity). δ=0 disables.
    severity_rank_weight: float
    severity_rank_enabled: bool
    # Cross-case Plan Item 7 — anchoring detector knobs.
    anchor_cosine_threshold: float
    anchor_penalty: float
    anchor_detector_enabled: bool
    # Cross-case Plan Item 5 — IE / mortality-sim budget knobs.
    mortality_sim_per_case_budget_s: float
    mortality_sim_min_depth: int
    mortality_sim_max_depth: int
    mortality_sim_parallel_enabled: bool
    # Cross-case Plan Item 6 — retrieval axis fallback knob.
    min_retrieval_intents: int
    retrieval_axis_fallback_enabled: bool
    # W6.2 Module J.2 — temperature-ensemble likelihood.
    swarm_temperature_ensemble_enabled: bool  # J.2: each worker fired at n temps
    swarm_ensemble_temps: str                 # comma-separated temperatures
    # W6.2 Module J.3 — cross-case persisted worker trust EMA.
    worker_trust_enabled: bool            # J.3: read+write worker_reliability.json post-verify
    # W6.3 Module J.4 — two-phase independent broadcast (Bayes independence repair).
    swarm_two_phase_broadcast_enabled: bool   # J.4: Round 1 no-prefix, Round 2 dissent-only refinement
    two_phase_round2_entropy_threshold: float # skip round 2 below this entropy
    # W6.3 Module J.5 — SID-style early worker dropout.
    swarm_early_dropout_enabled: bool         # J.5: cancel pending workers when consensus is sharp
    swarm_early_stop_entropy: float           # H threshold below which dropout fires
    swarm_early_stop_top1_mass: float         # top-1 mass threshold above which dropout fires
    swarm_early_stop_min_workers: int         # never drop below this many completed workers
    # W7.1 Module K.1 — always-on draft-tier challenger with full-tier escalation.
    always_on_challenger_enabled: bool        # K.1: draft-tier first, escalate to full only on signal
    challenger_draft_max_tokens_ratio: float  # draft tier token budget fraction of full
    challenger_escalate_alt_threshold: float  # ALT score above which we escalate
    challenger_escalate_severity: str         # comma list of severities that escalate
    # W7.1 Module K.6 — schema-validated parse-with-retry on challenger output.
    challenger_schema_retry_enabled: bool     # K.6: retry once with schema-printed prompt on parse failure
    # W7.2 Module K.2 — steel-man top-2 protocol.
    steelman_enabled: bool                    # K.2: run steelman defense for #2 candidate
    steelman_argument_threshold: float        # score above which we flag steelman swap candidate
    # W7.2 Module K.3 — counterfactual finding-ablation probe.
    counterfactual_ablation_enabled: bool     # K.3: ablate top-N findings, surface pivots
    counterfactual_ablation_top_n: int        # findings considered per probe
    # W7.3 Module K.5 — Bradley-Terry pairwise tournament.
    bradley_terry_tournament_enabled: bool    # K.5: pairwise MLE on contention cases
    bt_tournament_entropy_trigger: float      # species posterior entropy above which BT fires
    bt_tournament_top_k: int                  # candidates entered into pairwise tournament
    bt_tournament_judges_per_pair: int        # judge LLM calls per (i,j) pair
    # W7.3 Module K.4 — premise-conflict resolver agent.
    premise_conflict_resolver_enabled: bool   # K.4: LLM call when DS conflict K is high
    premise_conflict_k_threshold: float       # DS conflict above which K.4 fires
    premise_conflict_contradiction_threshold: float  # evidence.contradiction_mass alt trigger


def _validate_runtime_policy(policy: CdssRuntimePolicy) -> CdssRuntimePolicy:
    updates: dict[str, object] = {}

    # Adaptive verification needs at least two cycles to gather corrective evidence.
    if policy.verification_adaptive_gates_enabled and policy.max_reasoning_loops < 2:
        updates["max_reasoning_loops"] = 2

    effective_max_reasoning_loops = int(updates.get("max_reasoning_loops", policy.max_reasoning_loops))

    # LLM empty-slate hard-fail requires LLM-first reasoning to be meaningful.
    if policy.llm_empty_slate_hard_fail_enabled and not policy.llm_primary_reasoning_enabled:
        updates["llm_empty_slate_hard_fail_enabled"] = False

    # Pairwise discriminator routing is only useful when retrieval loops are permitted.
    if policy.retrieval_pair_discriminator_enabled and effective_max_reasoning_loops < 2:
        updates["retrieval_pair_discriminator_enabled"] = False

    # v2 coarse-to-fine dependencies: each higher wave requires the ones below.
    if policy.bayesian_posterior_enabled and not policy.hierarchical_belief_enabled:
        updates["hierarchical_belief_enabled"] = True
    # Effective flags so dependent rules below see the same values we're about to apply.
    eff_belief = bool(updates.get("hierarchical_belief_enabled", policy.hierarchical_belief_enabled))
    eff_bayes = bool(updates.get("bayesian_posterior_enabled", policy.bayesian_posterior_enabled))
    if policy.hierarchical_swarm_enabled and not (eff_belief and eff_bayes):
        updates["hierarchical_swarm_enabled"] = False
    eff_swarm = bool(updates.get("hierarchical_swarm_enabled", policy.hierarchical_swarm_enabled))
    if policy.dempster_shafer_fusion_enabled and not eff_bayes:
        updates["dempster_shafer_fusion_enabled"] = False
    if policy.cognitive_controller_enabled and not (eff_bayes and eff_swarm):
        updates["cognitive_controller_enabled"] = False
    eff_cognitive = bool(updates.get("cognitive_controller_enabled", policy.cognitive_controller_enabled))
    if policy.meta_controller_enabled and not eff_cognitive:
        updates["meta_controller_enabled"] = False
    # W4: causal-do verifier needs Bayesian posterior; BMA outcome does too.
    if policy.causal_do_verifier_enabled and not eff_bayes:
        updates["causal_do_verifier_enabled"] = False
    if policy.bma_outcome_enabled and not eff_bayes:
        updates["bma_outcome_enabled"] = False

    # W7.2 K.3 needs the Bayesian belief state to ablate against.
    if getattr(policy, "counterfactual_ablation_enabled", False) and not eff_bayes:
        updates["counterfactual_ablation_enabled"] = False

    if not updates:
        return policy
    return replace(policy, **updates)


def load_runtime_policy() -> CdssRuntimePolicy:
    """Load runtime policy from environment with safe local defaults."""
    learning_dir_raw = os.getenv("CDSS_LEARNING_DIR", "").strip()
    learning_dir = Path(learning_dir_raw) if learning_dir_raw else (_PROJECT_ROOT / "data" / "cdss" / "learning")
    speed_profile = _runtime_speed_profile()
    fast_overrides = _fast_overrides_enabled(speed_profile)
    policy = CdssRuntimePolicy(
        external_evidence_enabled=_env_flag("CDSS_EXTERNAL_EVIDENCE_ENABLED", True),
        pubmed_max_queries=_env_int("CDSS_PUBMED_MAX_QUERIES", 1 if fast_overrides else 2, 0, 8),
        pubmed_max_results=_env_int("CDSS_PUBMED_MAX_RESULTS", 2 if fast_overrides else 3, 1, 12),
        tavily_max_queries=_env_int("CDSS_TAVILY_MAX_QUERIES", 1 if fast_overrides else 2, 0, 8),
        reactive_search_max_iterations=_env_int("CDSS_REACTIVE_SEARCH_MAX_ITERATIONS", 0 if fast_overrides else 2, 0, 6),
        icd11_max_queries=_env_int("CDSS_ICD11_MAX_QUERIES", 1 if fast_overrides else 2, 0, 8),
        learning_auto_record=_env_flag("CDSS_LEARNING_AUTO_RECORD", True),
        learning_dir=learning_dir,
        local_llm_extraction_enabled=_env_flag("CDSS_LOCAL_LLM_EXTRACTION_ENABLED", True),
        local_llm_max_tokens=_env_int("CDSS_LOCAL_LLM_MAX_TOKENS", 192 if fast_overrides else 256, 128, 1024),
        local_llm_temperature=_env_float("CDSS_LOCAL_LLM_TEMPERATURE", 0.15, 0.0, 0.5),
        local_llm_ranker_enabled=_env_flag("CDSS_LOCAL_LLM_RANKER_ENABLED", True),
        llm_primary_reasoning_enabled=_env_flag("CDSS_LLM_PRIMARY_REASONING_ENABLED", True),
        llm_guardrail_only_enabled=_env_flag("CDSS_LLM_GUARDRAIL_ONLY_ENABLED", True),
        local_llm_verifier_enabled=_env_flag("CDSS_LOCAL_LLM_VERIFIER_ENABLED", True),
        local_llm_challenger_enabled=_env_flag("CDSS_LOCAL_LLM_CHALLENGER_ENABLED", True),
        support_ml_enabled=_env_flag("CDSS_SUPPORT_ML_ENABLED", True),
        retrieval_reranker_enabled=_env_flag("CDSS_RETRIEVAL_RERANKER_ENABLED", True),
        evidence_memory_enabled=_env_flag("CDSS_EVIDENCE_MEMORY_ENABLED", True),
        support_influence_cap=_env_float("CDSS_SUPPORT_INFLUENCE_CAP", 0.22, 0.05, 0.4),
        ood_revise_threshold=_env_float("CDSS_OOD_REVISE_THRESHOLD", 0.58, 0.2, 0.95),
        max_reasoning_loops=_env_int("CDSS_MAX_REASONING_LOOPS", 1, 1, 6),
        runtime_mode=_env_choice("CDSS_RUNTIME_MODE", "local", {"local", "cloud"}),
        runtime_speed_profile=speed_profile,
        gemini_enabled_in_cloud_mode=_env_flag("CDSS_GEMINI_ENABLED_IN_CLOUD_MODE", True),
        learned_safety_enabled=_env_flag("CDSS_LEARNED_SAFETY_ENABLED", True),
        learned_label_quality_enabled=_env_flag("CDSS_LEARNED_LABEL_QUALITY_ENABLED", True),
        llm_label_validation_enabled=False if fast_overrides else _env_flag("CDSS_LLM_LABEL_VALIDATION_ENABLED", False),
        learned_label_quality_threshold=_env_float("CDSS_LEARNED_LABEL_QUALITY_THRESHOLD", 0.52, 0.0, 1.0),
        memory_inference_policy=_env_choice(
            "CDSS_MEMORY_INFERENCE_POLICY",
            "validated_only",
            {"disabled", "validated_only", "all"},
        ),
        open_world_expansion_enabled=_env_flag("CDSS_OPEN_WORLD_EXPANSION_ENABLED", True),
        registry_prior_enabled=_env_flag("CDSS_REGISTRY_PRIOR_ENABLED", True),
        static_pattern_map_enabled=_env_flag("CDSS_STATIC_PATTERN_MAP_ENABLED", False),
        evidence_confidence_requires_verified_edges=_env_flag("CDSS_EVIDENCE_CONFIDENCE_REQUIRES_VERIFIED_EDGES", True),
        llm_research_planner_enabled=_env_flag("CDSS_LLM_RESEARCH_PLANNER_ENABLED", True),
        deterministic_mechanism_fallback_enabled=_env_flag("CDSS_DETERMINISTIC_MECHANISM_FALLBACK_ENABLED", False),
        state_mismatch_rules_enabled=_env_flag("CDSS_STATE_MISMATCH_RULES_ENABLED", True),
        trigger_compatibility_rerank_enabled=_env_flag("CDSS_TRIGGER_COMPATIBILITY_RERANK_ENABLED", True),
        research_min_evidence_coverage=_env_float("CDSS_RESEARCH_MIN_EVIDENCE_COVERAGE", 0.68, 0.2, 0.98),
        research_min_verified_edges=_env_int("CDSS_RESEARCH_MIN_VERIFIED_EDGES", 2, 0, 12),
        reasoning_artifact_v2_enabled=_env_flag("CDSS_REASONING_ARTIFACT_V2_ENABLED", False),
        llm_empty_slate_hard_fail_enabled=_env_flag("CDSS_LLM_EMPTY_SLATE_HARD_FAIL_ENABLED", True),
        intervention_safety_pre_evidence_enabled=_env_flag("CDSS_INTERVENTION_SAFETY_PRE_EVIDENCE_ENABLED", True),
        single_specificity_refine_pass_enabled=_env_flag("CDSS_SINGLE_SPECIFICITY_REFINE_PASS_ENABLED", True),
        complexity_router_after_challenger_enabled=_env_flag("CDSS_COMPLEXITY_ROUTER_AFTER_CHALLENGER_ENABLED", True),
        verification_adaptive_gates_enabled=_env_flag("CDSS_VERIFICATION_ADAPTIVE_GATES_ENABLED", False),
        verification_strict_issue_grounding_enabled=_env_flag("CDSS_VERIFICATION_STRICT_ISSUE_GROUNDING_ENABLED", True),
        retrieval_pair_discriminator_enabled=_env_flag("CDSS_RETRIEVAL_PAIR_DISCRIMINATOR_ENABLED", True),
        retrieval_query_novelty_metrics_enabled=_env_flag("CDSS_RETRIEVAL_QUERY_NOVELTY_METRICS_ENABLED", True),
        governor_retrieval_stall_early_stop_enabled=_env_flag("CDSS_GOVERNOR_RETRIEVAL_STALL_EARLY_STOP_ENABLED", True),
        governor_utility_rebound_threshold=_env_float("CDSS_GOVERNOR_UTILITY_REBOUND_THRESHOLD", 0.05, 0.0, 0.4),
        governor_utility_score_floor=_env_float("CDSS_GOVERNOR_UTILITY_SCORE_FLOOR", 0.34, 0.0, 1.0),
        llm_critical_stage_fail_fast_enabled=_env_flag("CDSS_LLM_CRITICAL_STAGE_FAIL_FAST_ENABLED", True),
        llm_sync_call_timeout_s=_env_float("CDSS_LLM_SYNC_CALL_TIMEOUT_S", 30.0 if fast_overrides else 45.0, 5.0, 300.0),
        epi_prior_enabled=_env_flag("CDSS_EPI_PRIOR_ENABLED", True),
        epi_prior_pubmed_enabled=_env_flag("CDSS_EPI_PRIOR_PUBMED_ENABLED", False),
        epi_prior_max_penalty=_env_float("CDSS_EPI_PRIOR_MAX_PENALTY", 0.24, 0.0, 0.40),
        specificity_calibrator_enabled=_env_flag("CDSS_SPECIFICITY_CALIBRATOR_ENABLED", True),
        learning_score_feedback_enabled=_env_flag("CDSS_LEARNING_SCORE_FEEDBACK_ENABLED", False),
        # Coarse-to-fine Bayesian agent (v2). Default OFF; flipped per-wave.
        hierarchical_belief_enabled=False if fast_overrides else _env_flag("CDSS_HIERARCHICAL_BELIEF_ENABLED", True),
        bayesian_posterior_enabled=False if fast_overrides else _env_flag("CDSS_BAYESIAN_POSTERIOR_ENABLED", True),
        # Default OFF as of 2026-05 optimization sprint: flat panel + rank fusion
        # delivers comparable diagnostic quality on gold10/fresh50 in ~30-50s
        # less per case. Set CDSS_HIERARCHICAL_SWARM_ENABLED=true to restore
        # coarse-to-fine waves for atypical / OOD cases.
        hierarchical_swarm_enabled=False if fast_overrides else _env_flag("CDSS_HIERARCHICAL_SWARM_ENABLED", False),
        dempster_shafer_fusion_enabled=False if fast_overrides else _env_flag("CDSS_DEMPSTER_SHAFER_FUSION_ENABLED", False),
        cognitive_controller_enabled=False if fast_overrides else _env_flag("CDSS_COGNITIVE_CONTROLLER_ENABLED", False),
        meta_controller_enabled=False if fast_overrides else _env_flag("CDSS_META_CONTROLLER_ENABLED", False),
        causal_do_verifier_enabled=False if fast_overrides else _env_flag("CDSS_CAUSAL_DO_VERIFIER_ENABLED", False),
        bma_outcome_enabled=False if fast_overrides else _env_flag("CDSS_BMA_OUTCOME_ENABLED", False),
        conformal_output_enabled=False if fast_overrides else _env_flag("CDSS_CONFORMAL_OUTPUT_ENABLED", False),
        likelihood_ensemble_temps=_env_str("CDSS_LIKELIHOOD_ENSEMBLE_TEMPS", "0.0,0.2,0.4"),
        likelihood_ensemble_max_tokens=_env_int("CDSS_LIKELIHOOD_ENSEMBLE_MAX_TOKENS", 24, 8, 128),
        mcts_max_ticks=_env_int("CDSS_MCTS_MAX_TICKS", 6, 1, 20),
        mcts_rollout_budget=_env_int("CDSS_MCTS_ROLLOUT_BUDGET", 20, 4, 100),
        utility_miss_risk_weight=_env_float("CDSS_UTILITY_MISS_RISK_WEIGHT", 0.8, 0.0, 5.0),
        utility_tightness_weight=_env_float("CDSS_UTILITY_TIGHTNESS_WEIGHT", 0.2, 0.0, 5.0),
        utility_budget_weight=_env_float("CDSS_UTILITY_BUDGET_WEIGHT", 0.05, 0.0, 1.0),
        conformal_alpha=_env_float("CDSS_CONFORMAL_ALPHA", 0.1, 0.01, 0.5),
        # W6.1 Module J — swarm rank fusion + strict label validation.
        swarm_rank_fusion_enabled=_env_flag("CDSS_SWARM_RANK_FUSION_ENABLED", False),
        rank_fusion_alpha=_env_float("CDSS_RANK_FUSION_ALPHA", 0.4, 0.0, 1.0),
        rank_fusion_beta=_env_float("CDSS_RANK_FUSION_BETA", 0.4, 0.0, 1.0),
        rank_fusion_gamma=_env_float("CDSS_RANK_FUSION_GAMMA", 0.2, 0.0, 1.0),
        rank_fusion_rrf_k=_env_int("CDSS_RANK_FUSION_RRF_K", 60, 1, 1000),
        swarm_label_validation_strict=_env_flag("CDSS_SWARM_LABEL_VALIDATION_STRICT", False),
        # Cross-case Plan Item 4 — severity-weighted ranking.
        severity_rank_weight=_env_float("CDSS_SEVERITY_RANK_WEIGHT", 0.25, 0.0, 5.0),
        severity_rank_enabled=_env_flag("CDSS_SEVERITY_RANK_ENABLED", True),
        # Cross-case Plan Item 7 — anchoring detector knobs.
        anchor_cosine_threshold=_env_float("CDSS_ANCHOR_COSINE_THRESHOLD", 0.78, 0.0, 1.0),
        anchor_penalty=_env_float("CDSS_ANCHOR_PENALTY", 0.7, 0.0, 1.0),
        anchor_detector_enabled=_env_flag("CDSS_ANCHOR_DETECTOR_ENABLED", True),
        # Cross-case Plan Item 5 — IE / mortality-sim budget knobs.
        mortality_sim_per_case_budget_s=_env_float("CDSS_MORTALITY_SIM_PER_CASE_BUDGET_S", 60.0, 5.0, 600.0),
        mortality_sim_min_depth=_env_int("CDSS_MORTALITY_SIM_MIN_DEPTH", 1, 1, 16),
        mortality_sim_max_depth=_env_int("CDSS_MORTALITY_SIM_MAX_DEPTH", 4, 1, 32),
        mortality_sim_parallel_enabled=_env_flag("CDSS_MORTALITY_SIM_PARALLEL_ENABLED", True),
        # Cross-case Plan Item 6 — retrieval axis fallback.
        min_retrieval_intents=_env_int("CDSS_MIN_RETRIEVAL_INTENTS", 2, 0, 16),
        retrieval_axis_fallback_enabled=_env_flag("CDSS_RETRIEVAL_AXIS_FALLBACK_ENABLED", True),
        # W6.2 Module J.2 — temperature-ensemble likelihood.
        swarm_temperature_ensemble_enabled=_env_flag("CDSS_SWARM_TEMPERATURE_ENSEMBLE_ENABLED", False),
        swarm_ensemble_temps=os.getenv("CDSS_SWARM_ENSEMBLE_TEMPS", "0.0,0.25,0.5"),
        # W6.2 Module J.3 — persisted worker trust.
        worker_trust_enabled=_env_flag("CDSS_WORKER_TRUST_ENABLED", False),
        # W6.3 Module J.4 — two-phase broadcast.
        swarm_two_phase_broadcast_enabled=_env_flag("CDSS_SWARM_TWO_PHASE_BROADCAST_ENABLED", False),
        two_phase_round2_entropy_threshold=_env_float("CDSS_SWARM_ROUND2_ENTROPY", 0.5, 0.0, 5.0),
        # W6.3 Module J.5 — SID early dropout.
        swarm_early_dropout_enabled=_env_flag("CDSS_SWARM_EARLY_DROPOUT_ENABLED", False),
        swarm_early_stop_entropy=_env_float("CDSS_SWARM_EARLY_STOP_ENTROPY", 0.4, 0.0, 5.0),
        swarm_early_stop_top1_mass=_env_float("CDSS_SWARM_EARLY_STOP_TOP1_MASS", 0.7, 0.0, 1.0),
        swarm_early_stop_min_workers=_env_int("CDSS_SWARM_EARLY_STOP_MIN_WORKERS", 3, 1, 20),
        # W7.1 Module K.1 — always-on draft challenger with full-tier escalation.
        always_on_challenger_enabled=_env_flag("CDSS_ALWAYS_ON_CHALLENGER_ENABLED", False),
        challenger_draft_max_tokens_ratio=_env_float("CDSS_CHALLENGER_DRAFT_TOKEN_RATIO", 0.5, 0.1, 1.0),
        challenger_escalate_alt_threshold=_env_float("CDSS_CHALLENGER_ESCALATE_ALT", 0.4, 0.0, 1.0),
        challenger_escalate_severity=os.getenv("CDSS_CHALLENGER_ESCALATE_SEVERITY", "high,critical"),
        # W7.1 Module K.6 — schema-validated parse retry.
        challenger_schema_retry_enabled=_env_flag("CDSS_CHALLENGER_SCHEMA_RETRY_ENABLED", False),
        # W7.2 Module K.2 — steel-man top-2 protocol.
        steelman_enabled=_env_flag("CDSS_STEELMAN_ENABLED", False),
        steelman_argument_threshold=_env_float("CDSS_STEELMAN_ARG_THRESHOLD", 0.7, 0.0, 1.0),
        # W7.2 Module K.3 — counterfactual finding-ablation probe.
        counterfactual_ablation_enabled=_env_flag("CDSS_COUNTERFACTUAL_ABLATION_ENABLED", False),
        counterfactual_ablation_top_n=_env_int("CDSS_COUNTERFACTUAL_ABLATION_TOP_N", 5, 1, 20),
        # W7.3 Module K.5 — Bradley-Terry pairwise tournament.
        bradley_terry_tournament_enabled=_env_flag("CDSS_BRADLEY_TERRY_ENABLED", False),
        bt_tournament_entropy_trigger=_env_float("CDSS_BT_ENTROPY_TRIGGER", 0.8, 0.0, 5.0),
        bt_tournament_top_k=_env_int("CDSS_BT_TOP_K", 3, 2, 6),
        bt_tournament_judges_per_pair=_env_int("CDSS_BT_JUDGES_PER_PAIR", 3, 1, 5),
        # W7.3 Module K.4 — premise-conflict resolver.
        premise_conflict_resolver_enabled=_env_flag("CDSS_PREMISE_CONFLICT_RESOLVER_ENABLED", False),
        premise_conflict_k_threshold=_env_float("CDSS_PREMISE_CONFLICT_K_THRESHOLD", 0.6, 0.0, 1.0),
        premise_conflict_contradiction_threshold=_env_float("CDSS_PREMISE_CONFLICT_CONTRADICTION", 0.24, 0.0, 1.0),
    )
    return _validate_runtime_policy(policy)
