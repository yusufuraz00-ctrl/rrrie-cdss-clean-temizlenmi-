# Evaluation

The system is measured, not guessed. Every change must be accompanied by a
re-run of the eval harness; regressions in hallucination rate or family@k
are blocking.

## Suites

| Suite | Path | Purpose |
|---|---|---|
| Gold-10 | `tests/benchmark_suite_gold10.json` | Curated, well-resolved cases |
| Fresh-50 | `tests/benchmark_suite_fresh50.json` | 50 diverse cases (TR + EN), difficulty 2–9, 15+ families |
| Stress-26 | `tests/benchmark_suite_stress26.json` | Adversarial / ambiguous |

The benchmarks are **read-only references**. They are never used to seed
prompts, retrieval queries, or rule logic.

## Metrics

The eval harness (`tests/eval/eval_metrics.py`, `scripts/run_eval.py`)
computes per run:

- **Family hit @ k** for k ∈ {1, 3, 5}
- **Calibration** (ECE) across ten confidence bins
- **Hallucination rate** = unsupported_claims / total_claims, computed via
  the inline grounding verdict on swarm output (not just the final packet)
- **Safety correctness** for `must_not_miss` cases: did the system flag
  urgency ≥ URGENT and surface the must-not-miss candidate?
- **Abstention precision**: when the system abstained, was top-1 wrong?
- **Margin distribution**: histogram of `top1 - top2` fused scores

## Reproducing

```
python scripts/run_eval.py --suite all --output artifacts/eval/<timestamp>.json
```

Compare against the saved baseline at `artifacts/eval/baseline.json`. The
regression test in `tests/eval/test_regression.py` will fail the build if
hallucination rate increases by more than 2 pp or family@3 drops by more
than 2 pp versus baseline.

## What we do not claim

- We do not claim clinical-grade performance.
- We do not claim accuracy on cases outside the benchmark families.
- We do not claim that the retrieval reranker is an ML cross-encoder; the
  fallback is a token-overlap heuristic. A real cross-encoder is on the
  Tier-3 roadmap.
