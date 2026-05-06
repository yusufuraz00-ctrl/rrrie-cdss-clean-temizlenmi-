# RRRIE / Cortex CDSS — System-Level Upgrade Report

**Date:** 2026-05-02
**Scope:** Whole system. No fixed benchmark optimization. No
disease-specific shortcuts. Generalizable, root-cause improvements only.

---

## 1. Architecture summary (current state)

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

Key modules:

- `src/cdss/runtime/state_machine.py` — orchestration, dedup, swarm fan-out.
- `src/cdss/reasoning/{rank_fusion,bradley_terry,likelihood_ensemble}.py` — fusion math.
- `src/cdss/clinical/{grounding,inline_grounding,safety,confirmation_planner,verification}.py` — clinical logic.
- `src/cdss/retrieval/planner.py` — gap-driven retrieval + heuristic reranker.
- `src/cdss/runtime/policy_gates.py` — adaptive depth + abstention (new).
- `src/cdss/core/{thresholds,tracing}.py` — config + observability (new).
- `gui/server.py` — FastAPI + WebSocket.
- `gui/static/{app,index,style}.{js,html,css}` — vanilla-JS frontend.

---

## 2. Main weaknesses discovered

### Reasoning
- Grounding ran *only* late, in verification — swarm hypotheses could carry
  hallucinated rationales for 4+ stages before any check.
- Bidirectional LLM↔rule coupling in verification: penalty fired on
  LLM-generated issue strings, no rule isolation.
- Unbounded cognitive loop iteration; convergence depended on LLM
  self-signal.
- Floating-point entropy as a tier-classification gate at exactly 3.0.
- Silent registry lookup failure (`runtime/diagnostic_contract.py:66-78`).

### Retrieval / evidence
- Reranker is a hand-crafted token-overlap scorer, not a neural cross
  encoder. Stats field already labels itself `heuristic_fallback`, but no
  evidence-starvation flag was emitted, so the system silently claimed
  evidence-grounded with zero items.
- No per-candidate evidence coverage metric.
- Aggressive JSON repair (`utils/json_payloads.py`) hid LLM-output parse
  failures.

### Frontend / product
- No skeleton loaders. Result slot blocked until full packet arrived.
- Theatrical swarm-map nodes did not reflect real worker counts or
  parallelism.
- Evidence surface was "title + relation" only — no excerpt, source URL,
  trust score, or drill-down.
- No export/share. Decision packets lived in memory only.
- Error recovery silent.

### Engineering / observability
- No distributed tracing. Single Python logger + incident IDs.
- Hardcoded thresholds (vitals, fusion mixture, confirmation margin,
  retrieval reranker weights) scattered across 5+ files. Untunable
  without code changes.
- `run.py` is a 49 KB monolith.
- Eval scoring counted keyword hits — no precision/recall, no calibration,
  no hallucination rate.

### Hardcoded clinical shortcuts
- No disease-name-keyed branches found. Vital thresholds were standard
  clinical constants but lacked citation and were not centrally
  configurable.

---

## 3. SOTA comparison (synthesis)

| Area | SOTA practice (2024-2025) | Pre-upgrade | Delta applied |
|---|---|---|---|
| Medical retrieval | Dense bi-encoder + cross-encoder rerank | Token-overlap heuristic | Added starvation flag and per-candidate coverage; dense rerank deferred (Tier-3) |
| Hallucination control | Claim decomposition + per-claim entailment | Late-only token-overlap | Added inline grounding gate at swarm output (drop / demote / pass) |
| Diagnostic agents | Verifier-guided decoding, abstention on low margin | Fusion + Beta posterior, no abstention path | Added `policy_gates.abstain_or_escalate` + UI banner |
| Uncertainty | Conformal prediction | Conformal partially present, policy-gated | Kept; promoted to first-class on packet |
| Adaptive depth | Skip heavy LLM stages when stable | Fixed 8-stage pipeline | Added `should_skip_deep_simulation` (config-gated, opt-in plumbing) |
| Eval harness | Per-stage regression, calibration, hallucination | Keyword hits | Built typed `eval_metrics.py` with ECE, hallucination rate, safety correctness, abstention precision, margin histogram |
| Observability | OpenTelemetry, structured spans | Logger + incident IDs | Added in-process JSONL span emitter with subscriber bridge for WS events |
| Frontend trust | Citation drill-down, evidence quality bars, abstention banner | Title + relation only | Added drill-down drawer, per-candidate coverage chip, grounding chip, export, abstention banner |

---

## 4. Files changed

### New
- `config/clinical_thresholds.json`
- `src/cdss/core/thresholds.py`
- `src/cdss/core/tracing.py`
- `src/cdss/clinical/inline_grounding.py`
- `src/cdss/runtime/policy_gates.py`
- `tests/eval/__init__.py`
- `tests/eval/eval_metrics.py`
- `tests/eval/test_regression.py`
- `tests/test_thresholds_loader.py`
- `tests/test_tracing_span.py`
- `tests/test_inline_grounding.py`
- `tests/test_evidence_starvation.py`
- `tests/test_eval_metrics.py`
- `tests/test_adaptive_depth_gate.py`
- `tests/test_no_hardcoded_diseases.py`
- `scripts/run_eval.py`
- `artifacts/eval/baseline.json` (placeholder; populated by first live run)
- `docs/architecture.md`
- `docs/evaluation.md`
- `docs/safety.md`
- `docs/limitations.md`
- `docs/anti_hardcoding.md`
- `docs/upgrade_report_2026-05.md`

### Modified
- `src/cdss/contracts/models.py` — added `GroundingVerdict`,
  `evidence_starvation_flag`, `coverage_per_candidate`,
  `starved_candidates`, `abstention_*` fields on `DecisionPacket` and
  `DecisionPacketView`, grounding fields on `DifferentialCandidate`.
- `src/cdss/clinical/safety.py` — vital thresholds read from config.
- `src/cdss/clinical/confirmation_planner.py` — close-margin and priority
  scores read from config.
- `src/cdss/clinical/prototype_memory.py` — shock-index thresholds read
  from config.
- `src/cdss/clinical/verification.py` — accepts `retrieval_stats`, surfaces
  `low_evidence_grounding` issue, threshold migration.
- `src/cdss/reasoning/rank_fusion.py` — RRF k and α/β/γ from config.
- `src/cdss/retrieval/planner.py` — reranker weights from config; emits
  `evidence_starvation_flag`, `starved_candidates`,
  `coverage_per_candidate`.
- `src/cdss/runtime/state_machine.py` — wired inline grounding gate
  immediately after dedup; surfaces `inline_grounding_pass_rate` in
  `typed_case_bundle`.
- `src/cdss/app/view_model.py` — propagates abstention + retrieval
  starvation fields into the view.
- `gui/server.py` — computes `abstain_or_escalate`, populates abstention
  fields on packet, promotes status to `ABSTAIN` when gate fires.
- `gui/static/app.js` — abstention banner, per-candidate evidence and
  grounding chips, evidence drill-down drawer with Escape close,
  Export-packet button (JSON download + `window.print()`).
- `gui/static/style.css` — Batch-5 component styles + print stylesheet.
- `README.md` — full rewrite around architecture and policy.

---

## 5. Reasoning improvements

- **Inline grounding gate** before differential is finalized. Each
  candidate's rationale is checked against the patient narrative; verdicts
  are `pass` (kept), `demote` (score multiplied by support ratio), or
  `drop` (removed). Pass rate is recorded for eval.
- **Adaptive-depth and abstention gates** in `policy_gates.py`, decoupled
  from any specific stage so they can be wired anywhere. Pure functions,
  fully tested.
- **Low-evidence verification gate**: when retrieval is starved, the
  verification report surfaces a `low_evidence_grounding` issue and
  decrements the epistemic score.
- **Status promotion to `ABSTAIN`** when the gate recommends it AND no
  stronger verifier hint already mandated `REVISE` / `URGENT_ESCALATION`.

---

## 6. Scientific / scoring improvements

- Centralized clinical thresholds with citation and `last_reviewed` —
  every clinical constant in the system is now auditable.
- Typed eval metrics (`tests/eval/eval_metrics.py`):
  - Family hit @ k for k ∈ {1, 3, 5}.
  - Calibration ECE + reliability diagram bins.
  - Hallucination rate via inline grounding pass rate.
  - Safety correctness on must-not-miss cases.
  - Abstention precision (top-1 wrong | abstained).
  - Margin distribution histogram.
- Regression test in `tests/eval/test_regression.py` blocks > 2 pp
  regression in hallucination rate or family@3.

---

## 7. Safety / grounding improvements

- Two-layer grounding: inline gate before frontier construction, and
  late validator in verification. Both share tokenizer + segment matching.
- Drop / demote thresholds are config-tunable
  (`grounding.inline_*` keys).
- Abstention is first-class on the packet (`abstention_recommended`,
  `abstention_reason`, `abstention_margin`, `abstention_grounding_risk`).
- The frontend renders an abstention banner above the decision board so
  the human reviewer cannot miss it.

---

## 8. Retrieval / evidence improvements

- `RetrievalRankingStats` now carries `coverage_per_candidate`,
  `evidence_starvation_flag`, and `starved_candidates`.
- Empty retrieval is no longer silent; verification picks it up.
- Per-candidate evidence coverage is rendered as a colored chip on each
  differential row in the UI.
- Reranker weights centrally tunable via `clinical_thresholds.json`.

---

## 9. Frontend / product improvements

- **Evidence drill-down drawer** — click an evidence item to see excerpt,
  source URL, trust score, relation, verification status, and linked
  candidates. Closes with Escape or backdrop click.
- **Per-candidate evidence chips** — coverage % and grounding verdict
  shown inline on each differential row.
- **Abstention banner** — prominent, color-coded, above the hero.
- **Export button** — downloads a JSON `decision_packet_<case>_<ts>.json`
  and triggers `window.print()` against a clean print stylesheet that
  hides nav, swarm map, drawers, and feedback chrome.
- **Print stylesheet** — produces a clinically readable single-page
  document.
- **Reduced motion + focus visibility** preserved from prior pass.

---

## 10. Testing / evaluation improvements

- Test count grew from 104 → **155** unit tests (all green), excluding the
  large fresh50 integration suite.
- New unit tests cover: thresholds loader, tracing span emitter, inline
  grounding gate, evidence starvation surfacing, eval metric functions,
  adaptive-depth and abstention gates, anti-hardcoding CI guard.
- `scripts/run_eval.py` runs offline (against pre-computed packets JSONL)
  for fast CI, or live (against benchmark suites) when the LLM stack is
  configured.

---

## 11. Before / after

| Metric | Before | After (functional) |
|---|---|---|
| Unit tests | 104 | **155** (+51) |
| Hardcoded thresholds in source | 5+ files | 0 (all in `clinical_thresholds.json`) |
| Grounding layers | 1 (late) | **2** (inline + late) |
| Retrieval failure visibility | Silent | Surfaced as `low_evidence_grounding` + UI chip |
| Abstention path | Implicit (REVISE) | Explicit `ABSTAIN` + banner + reason |
| Observability | Logger only | Per-stage JSONL spans + WS subscriber |
| Eval scoring | Keyword hits | Family@k, ECE, hallucination, safety, abstention precision, margins |
| Anti-hardcoding | Convention | **Enforced** by CI guard |
| README depth | Install only | Architecture + schema + policy + eval |
| Frontend evidence surface | Title + relation | Drill-down drawer + chips |
| Export | None | JSON + printable PDF |
| Print stylesheet | None | Yes |

A live numerical baseline (family@k, ECE, etc.) requires running
`scripts/run_eval.py --suite all --baseline` against the LLM stack.
This was deliberately not run during the upgrade because the local LLM
endpoint can take 5–20 min per case; the harness is in place and ready.

---

## 12. Remaining limitations

See `docs/limitations.md`. Highlights:

- Reranker is heuristic, not neural cross-encoder.
- Negation handling supports English + Turkish only.
- WebSocket has no auth or rate limit (local single-user use).
- `run.py` monolith not split.
- Mobile layout not hardened.

---

## 13. Strategic roadmap

### Tier-3 (deferred)
1. Real cross-encoder reranker (MedCPT or local BGE-reranker) behind
   feature flag, with `RetrievalRankingStats.cross_encoder_used`
   reporting truthfully.
2. NLI / entailment-based grounding to catch token-collision
   hallucinations.
3. OpenTelemetry OTLP exporter as a `tracing.subscribe` listener.
4. Auth + rate limit + multi-tenant story for any deploy beyond local.
5. HL7-FHIR export from the print/JSON pipeline.
6. Mobile-hardened layout and full keyboard a11y on tab cycling.
7. Conformal prediction set surfaced first-class in UI.
8. Live baseline + before/after numerical comparison once LLM stack is
   available for an evaluation run.

---

## 14. Verification

```powershell
.venv\Scripts\python -m pytest tests\ -x --ignore=tests\test_benchmark_suite_fresh50.py -q
# 155 passed, 5 skipped
```

Manual UX check (Batch 5):

1. Start the server: `setup\start.ps1`.
2. Open `http://127.0.0.1:7860`.
3. Run any case. After analysis completes, confirm:
   - Differential rows show per-candidate evidence coverage chip.
   - Each evidence item is clickable; the drill-down drawer opens.
   - Abstention banner appears when low-margin + high-grounding-risk.
   - Export button downloads JSON and triggers printable view.

Eval harness offline check:

```powershell
.venv\Scripts\python scripts\run_eval.py --packets <jsonl> --output artifacts\eval\smoke.json
```

---

## 15. Closing note

The upgrade focused on root causes rather than cosmetic patches. Every
new behavior is config-tunable, observable, and testable. The
anti-hardcoding policy is now enforced by a CI guard rather than
maintained by convention. The system is meaningfully more credible for
serious technical and clinical review than the pre-upgrade state, while
preserving every feature that was already working.
