# Known limitations

This system is research-grade. The following limitations are explicit:

## Clinical
- Not certified for clinical decision-making. Outputs require clinician
  review before any patient-facing action.
- The benchmark suites cover ~15 diagnostic families across ~80 cases. The
  system has no validated performance outside these families.
- Objective confirmation planning is generic ("targeted exam, vitals, labs,
  imaging, or specialist assessment as clinically indicated"). It does not
  prescribe specific tests; that is by design to avoid disease-specific
  shortcuts.

## Reasoning
- The grounding validator is **token-overlap based**, not NLI / entailment.
  Hallucinations that re-use narrative tokens but invert their meaning may
  pass the inline gate. A real entailment-based grounding layer is on the
  Tier-3 roadmap.
- Negation handling supports English and Turkish negation markers only and
  does not handle scope or double negation.
- The LLM-driven cognitive loop has a configurable max-iter but no formal
  termination guarantee on degenerate self-critique.

## Retrieval
- The reranker is a heuristic token-overlap scorer. It is **not** a neural
  cross-encoder. The `RetrievalRankingStats.cross_encoder_used` field
  reports `"heuristic_fallback"` so this is not hidden.
- There is no persistent retrieval cache across sessions; each run hits
  external APIs (subject to per-tool TTL caches).

## Engineering
- The WebSocket layer has no auth / rate limit. The system is intended for
  local single-user use.
- The `run.py` bootstrap is a 49 KB monolith and will be split in a
  separate cleanup pass.
- Mobile layout is not hardened. Desktop only for now.

## Eval
- The eval harness measures family-level hits, calibration, hallucination
  rate, safety correctness, and abstention precision — but does not yet
  measure clinical outcome impact. Outcome evaluation requires a
  prospective study, which is out of scope.
