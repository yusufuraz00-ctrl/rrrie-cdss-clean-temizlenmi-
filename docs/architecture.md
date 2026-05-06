# Architecture

## Pipeline

```
PatientInput
   │
   ▼
INTAKE ──► R2 (Research) ──► DIFFERENTIAL (swarm) ──► COGNITIVE_LOOP
                                       │                     │
                                       ▼                     ▼
                              BACKWARD_SIMULATION ◄── OUTCOME_SIMULATION
                                       │
                                       ▼
                              VERIFICATION ──► ACTION_PLAN ──► DecisionPacket
```

Each stage is a method on `CdssStateMachine` in `src/cdss/runtime/state_machine.py`.
Stages communicate via a typed `WorkflowState` (no untyped dicts in transit).

## Key modules

| Module | Responsibility |
|---|---|
| `src/cdss/runtime/state_machine.py` | Stage orchestration, swarm fan-out, dedup, fusion call sites |
| `src/cdss/reasoning/rank_fusion.py` | Reciprocal Rank Fusion + Trust-Weighted Borda + likelihood mean |
| `src/cdss/reasoning/bradley_terry.py` | Pairwise MLE for ensemble ranking |
| `src/cdss/reasoning/likelihood_ensemble.py` | Beta-posterior likelihood from temperature ensemble |
| `src/cdss/clinical/grounding.py` | Token-overlap grounding validator |
| `src/cdss/clinical/inline_grounding.py` | Inline candidate-level grounding gate (Batch 2) |
| `src/cdss/clinical/safety.py` | Vital-driven risk profile + urgency tiering |
| `src/cdss/clinical/confirmation_planner.py` | Objective-confirmation evidence-need synthesis |
| `src/cdss/clinical/verification.py` | Final-stage gates: closure coverage, contradiction, abstention |
| `src/cdss/retrieval/planner.py` | Gap-driven query generation, axis fallback, heuristic reranker |
| `src/cdss/runtime/policy_gates.py` | Adaptive depth + abstention decisions (Batch 4) |
| `src/cdss/core/thresholds.py` | Centralized config-driven clinical / scoring thresholds |
| `src/cdss/core/tracing.py` | In-process structured span emitter (JSONL + subscribers) |

## Fusion math

For each candidate `d`:

```
RRF(d)  = Σ_i 1 / (k + rank_i(d))                      [Cormack 2009, k = 60]
TWB(d)  = Σ_i τ_i · w_hier(profile_i) · Borda_i(d)     [TeamMedAgents 2025]
L̄(d)   = mean of Beta(α, β) posterior over k samples [temperature ensemble]

score(d) = α · RRF(d) + β · TWB(d) + γ · L̄(d)
         (defaults α = β = 0.4, γ = 0.2; tunable in clinical_thresholds.json)
```

## Grounding model

Two layers:

1. **Inline gate** (`inline_grounding.gate_candidate_grounding`) runs after
   the swarm dedup step. Demotes or drops candidates whose rationale cites
   findings that are not traceable to the patient narrative.
2. **Final validator** (`grounding.validate_narrative_grounding`) runs in
   verification. Operates on the full `StructuredFindings` and produces a
   `GroundingReport` with `hallucination_risk_score`.

Both layers reuse the same tokenizer and segment-supporting heuristic; the
inline gate adds candidate-aware demotion/drop thresholds from
`config/clinical_thresholds.json`.

## Retrieval flow

1. **Plan** — gap-driven `RetrievalIntent`s synthesized from the hypothesis
   frontier + active red flags.
2. **Fallback** — axis-template fallback when the planner produces fewer than
   the configured minimum intents.
3. **Fetch** — sequential calls into PubMed, Europe PMC, Wikipedia, web.
4. **Rerank** — heuristic token-overlap rerank with candidate-aligned bonus
   and novelty scoring.
5. **Surface** — `RetrievalRankingStats` reports specificity gain, citation
   alignment, novelty gain, evidence-starvation flag (Batch 2).

## Observability

Every `_run_*` stage method is wrapped in a `tracing.span` context. Spans are
written as JSONL to `output/traces/<run_id>.jsonl` and broadcast to any
subscriber (the WebSocket layer mirrors them as `trace` events). Subscribers
must not raise; they are isolated by a try/except.
