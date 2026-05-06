"""Eval regression test.

This test does not run the live pipeline. It loads
`artifacts/eval/baseline.json` and the most recent candidate report from
`artifacts/eval/<latest>.json` (excluding `baseline.json` itself) and asserts
that hallucination rate has not increased by more than 2 percentage points and
family@3 has not dropped by more than 2 percentage points versus the baseline.

If no candidate report exists yet, the test is skipped (the harness has not
been run). If the baseline is the placeholder (n == 0), the test is also
skipped — there is nothing to regress against.

Run a fresh candidate report with:

    .venv/Scripts/python scripts/run_eval.py --suite all
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "eval"
_REGRESSION_TOLERANCE = 0.02  # 2 percentage points


def _latest_candidate_report() -> Path | None:
    if not _ARTIFACTS.exists():
        return None
    candidates = sorted(
        (p for p in _ARTIFACTS.glob("*.json") if p.name != "baseline.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_regression_in_hallucination_or_family_at_3():
    baseline_path = _ARTIFACTS / "baseline.json"
    if not baseline_path.exists():
        pytest.skip("no baseline.json")
    baseline = _load(baseline_path)
    if int(baseline.get("n", 0) or 0) == 0:
        pytest.skip("baseline is placeholder; run run_eval.py --baseline once to produce a real baseline")

    candidate_path = _latest_candidate_report()
    if candidate_path is None:
        pytest.skip("no candidate eval report; run scripts/run_eval.py first")
    candidate = _load(candidate_path)

    base_halluc = float(baseline.get("hallucination", {}).get("hallucination_rate", 0.0) or 0.0)
    cand_halluc = float(candidate.get("hallucination", {}).get("hallucination_rate", 0.0) or 0.0)
    assert cand_halluc <= base_halluc + _REGRESSION_TOLERANCE, (
        f"hallucination regressed: baseline={base_halluc:.4f} candidate={cand_halluc:.4f}"
    )

    base_f3 = float(baseline.get("family_hit_at_3", 0.0) or 0.0)
    cand_f3 = float(candidate.get("family_hit_at_3", 0.0) or 0.0)
    assert cand_f3 >= base_f3 - _REGRESSION_TOLERANCE, (
        f"family_hit_at_3 regressed: baseline={base_f3:.4f} candidate={cand_f3:.4f}"
    )
