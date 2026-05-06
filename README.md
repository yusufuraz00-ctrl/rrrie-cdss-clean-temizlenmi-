# RRRIE-CDSS / Cortex Engine

Research-grade clinical decision-support system. Multi-stage diagnostic
reasoning with grounded retrieval, calibrated uncertainty, and explicit
abstention. Local-first (Qwen via `llama-server`), with optional Gemini
escalation.

> **Disclaimer:** Research and engineering use only. The system is not
> certified for clinical use. Outputs require clinician review.

---

## What it is

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

- **Reasoning ensemble:** Bradley-Terry MLE pairwise ranking, Reciprocal
  Rank Fusion (Cormack 2009), Trust-Weighted Borda (TeamMedAgents 2025),
  and a Beta-posterior likelihood ensemble — combined as a tunable
  convex mixture in `src/cdss/reasoning/rank_fusion.py`.
- **Grounding** runs in two layers: an inline gate that demotes or drops
  candidates whose rationale is not traceable to the patient narrative
  (`src/cdss/clinical/inline_grounding.py`), and a final validator that
  reports a hallucination-risk score on the packet
  (`src/cdss/clinical/grounding.py`).
- **Retrieval** plans gap-driven queries from the hypothesis frontier,
  reranks with a token-overlap heuristic, and surfaces an
  `evidence_starvation_flag` when no items match a candidate.
- **Safety** uses a typed `RiskProfile` derived from objective vitals plus
  generic time-sensitive hazard markers — no disease-specific shortcuts.
- **Abstention** is first-class: when the fused margin is below a
  configurable threshold and grounding risk is high, the packet status
  flips to `ABSTAIN` and the UI shows a banner.

See `docs/architecture.md` for the long version.

## What it is *not*

- Not a chatbot.
- Not a disease-name lookup table — see `docs/anti_hardcoding.md`.
- Not certified. See `docs/limitations.md`.

---

## Quick start

```powershell
powershell -ExecutionPolicy Bypass -File .\setup\install.ps1
powershell -ExecutionPolicy Bypass -File .\setup\start.ps1
# open http://127.0.0.1:7860
```

Prerequisites: Python 3.11+, NVIDIA GPU (4 GB+ VRAM),
`llama-server.exe` in `llama-server\`. See `INSTALL.md` for details.

---

## DecisionPacket schema (key fields)

```python
class DecisionPacket(BaseModel):
    case_id: str
    status: DecisionStatus           # PRELIMINARY | REVISE | ABSTAIN | URGENT_ESCALATION | ...
    differential: DifferentialSet    # ranked candidates with grounding verdicts
    evidence: EvidenceBundle         # ranked items + retrieval intents
    risk_profile: RiskProfile        # urgency tier + vital alerts
    verification: VerificationReport # gates, issues, conformal set
    retrieval_stats: RetrievalRankingStats  # coverage_per_candidate, evidence_starvation_flag
    diagnostic_confidence: float
    closure_confidence: float
    abstention_recommended: bool
    abstention_reason: str
    inline_grounding_pass_rate: float
    reasoning_trace: list[str]
```

Full schema in `src/cdss/contracts/models.py`.

---

## Eval

```powershell
.venv\Scripts\python scripts\run_eval.py --suite all --baseline
```

Metrics: family hit @ k, calibration ECE, hallucination rate (via inline
grounding pass rate), safety correctness on must-not-miss cases,
abstention precision, margin distribution. See `docs/evaluation.md`.

CI regression test in `tests/eval/test_regression.py` blocks merges that
regress hallucination rate or family@3 by more than 2 percentage points.

---

## Tunable thresholds

All clinical / scoring constants live in
`config/clinical_thresholds.json`. Each entry includes a numeric `value`,
a `source` citation, and a `last_reviewed` date. The system falls back
to in-code defaults if the file is missing or malformed.

Removing or breaking the file does not crash the pipeline.

---

## Anti-hardcoding policy

- No symptom → diagnosis lookup tables.
- No disease-name-keyed `if`/`elif` branches.
- Eval suites are read-only references — never used to seed prompts,
  retrieval queries, or rules.
- A CI guard (`tests/test_no_hardcoded_diseases.py`) statically scans the
  source tree and fails the build if disease tokens appear in control
  flow.

See `docs/anti_hardcoding.md`.

---

## Observability

Structured spans for every pipeline stage are emitted as JSONL to
`output/traces/<run_id>.jsonl`. The WebSocket layer mirrors them as
`trace` events. See `src/cdss/core/tracing.py`. Disable with
`CDSS_TRACE_DISABLE=1`.

---

## Hardening checks

```powershell
py run.py --secret-scan
py run.py --complexity-check
py run.py --audit-matrix
py run.py --hardening-checks
```

Artifacts go to `artifacts\verification\`. CI workflow at
`.github/workflows/hardening-audit.yml`.

---

## Documentation

| File | Purpose |
|---|---|
| `docs/architecture.md` | Pipeline stages, fusion math, grounding, retrieval flow |
| `docs/evaluation.md` | What we measure, how to reproduce |
| `docs/safety.md` | Escalation tiers, abstention policy, what the system will not do |
| `docs/limitations.md` | Explicit research-grade limits |
| `docs/anti_hardcoding.md` | Forbidden patterns and the CI guard |
| `docs/upgrade_report_2026-05.md` | System-level upgrade summary (this round) |

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/vnext/analyze` | POST | Main diagnostic analysis |
| `/api/vnext/health` | GET  | Health check |
| `/ws/chat`           | WS   | Streaming analysis with stage events |

---

## License

MIT — see `LICENSE`.
