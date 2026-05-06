# Anti-hardcoding policy

This system is built to avoid disease-specific shortcuts. The reasoning
engine must improve generally, not by memorizing.

## Forbidden patterns

The following are prohibited anywhere in the source tree:

- Symptom → diagnosis lookup tables.
- Disease-name-keyed `if` / `elif` branches.
- ICD codes embedded as control-flow keys.
- Benchmark-tuned constants, prompt hacks, or memorized diagnosis
  mappings.
- Shortcut rules that only resolve a small set of cases.

## Allowed patterns

- Standard clinical constants (vital thresholds, shock-index cutoffs,
  guideline-defined urgency triggers) live in
  `config/clinical_thresholds.json` with a `source` citation and a
  `last_reviewed` date. They are versioned and auditable.
- Generic hazard pattern detectors (e.g. "rapid + severe + functional
  loss") are allowed because they are diagnosis-agnostic.
- Stage-thresholds for fusion, grounding, abstention, and adaptive depth
  live in the same config file.

## Eval data is read-only

The benchmark suites in `tests/benchmark_suite_*.json` are reference
fixtures only. They are never used to:

- Seed prompts to LLM workers.
- Drive retrieval queries.
- Train, fit, or calibrate any in-pipeline weights.

## CI guard

`tests/test_no_hardcoded_diseases.py` (Batch 6) statically scans the
source tree for common disease tokens embedded in `if` / `elif` branches
and fails the build if any are found.
