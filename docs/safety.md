# Safety policy

The system is a research-grade decision-support engine. It is **not** a
substitute for clinician judgment, and it is not certified for clinical use.

## What the system will not do

- It will not provide a final diagnosis without surfacing the supporting
  evidence and unresolved uncertainty.
- It will not return a `closed` decision when a critical finding is
  unexplained.
- It will not silently swallow retrieval failure. When evidence is empty
  for any candidate, the verification layer adds a `low_evidence_grounding`
  issue and the frontend shows an explicit warning chip.
- It will not invent symptoms, exposures, or medications. The grounding
  validator (inline + final) flags any claim that is not traceable to the
  patient narrative.
- It will not recommend dangerous self-treatment. The unsafe-self-treatment
  classifier in `safety.py` blocks delay-or-home plans when red flags
  exist.

## Escalation tiers

| Tier | Triggers | Response |
|---|---|---|
| `EMERGENCY` | Vital instability, time-sensitive permanent-harm risk pattern, hazard context, device-reliability failure | Mandatory clinician review, repeat vitals, objective testing, blocked unsupervised discharge |
| `URGENT` | Red flags, escalation reasons present, pathway-fit context | Same-day clinician assessment, targeted objective workup |
| `ROUTINE` | Otherwise | Collect missing context if symptoms persist |

Vital thresholds live in `config/clinical_thresholds.json` with citation
and `last_reviewed` date. They are not embedded in source code.

## Abstention

When `top1 − top2` margin is below `adaptive_depth.abstain_margin_max`
**and** the inline grounding-risk score exceeds
`adaptive_depth.abstain_grounding_risk_min`, the system emits
`decision = "abstain_pending_objective_data"` rather than forcing a top-1
answer. The frontend renders an abstention banner above the result.

## Anti-hardcoding

See `docs/anti_hardcoding.md`. No symptom→diagnosis lookup tables, no
disease-name-keyed if/elif branches, no benchmark-tuned shortcuts.
