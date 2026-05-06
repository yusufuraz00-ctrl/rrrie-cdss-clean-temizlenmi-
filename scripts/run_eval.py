"""Eval orchestration script.

Two modes:

  --packets <jsonl>   Read pre-computed decision packets (one JSON object per line)
                      with fields {case_id, expected_family, must_not_miss,
                      packet, abstained?, top1_family?, confidence?} and produce
                      a typed eval report. This mode is fast, fully offline, and
                      is what CI runs.

  --suite all|gold10|fresh50|stress26
                      Run the live pipeline over a benchmark suite and produce
                      packets *and* a typed eval report. This mode requires the
                      LLM stack (llama-cpp / Gemini) to be configured. The
                      orchestration here is a thin wrapper around the existing
                      CdssApplicationService — the metric functions are
                      identical to the offline path.

Both modes write `artifacts/eval/<run_id>.json` plus a markdown summary at
`artifacts/eval/<run_id>.md`. Pass `--baseline` to additionally write
`artifacts/eval/baseline.json` (overwrites any prior baseline).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.eval.eval_metrics import (  # noqa: E402
    AbstentionRecord,
    EvalRecord,
    cohort_report,
)


_SUITES: dict[str, str] = {
    "gold10": "tests/benchmark_suite_gold10.json",
    "fresh50": "tests/benchmark_suite_fresh50.json",
    "stress26": "tests/benchmark_suite_stress26.json",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _records_from_jsonl(path: Path) -> list[EvalRecord]:
    rows = _read_jsonl(path)
    out: list[EvalRecord] = []
    for row in rows:
        out.append(
            EvalRecord(
                case_id=str(row.get("case_id", "")),
                expected_family=str(row.get("expected_family", "")),
                must_not_miss=bool(row.get("must_not_miss", False)),
                packet=row.get("packet") or {},
                abstained=bool(row.get("abstained", False)),
                top1_family=str(row.get("top1_family", "")),
                confidence=float(row.get("confidence", 0.0) or 0.0),
            )
        )
    return out


def _load_suite_cases(suite_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    return list(payload.get("cases", payload) or [])


async def _run_live_suite(suite_name: str) -> list[EvalRecord]:
    """Run the live CDSS pipeline over a benchmark suite. Requires LLM stack."""
    from src.cdss.app.service import CdssApplicationService
    from src.cdss.contracts.models import PatientInput
    from src.llm.llama_cpp_client import LlamaCppClient

    suite_path = ROOT / _SUITES[suite_name]
    cases = _load_suite_cases(suite_path)

    client = LlamaCppClient()
    service = CdssApplicationService(llm_client=client)
    out: list[EvalRecord] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        text = str(case.get("patient_text", "") or "")
        expected_family = str(case.get("disease_family", "") or "")
        expectations = case.get("expectations", {}) or {}
        must_not_miss = bool(expectations.get("must_not_miss", False) or "must_not_miss" in (expectations.get("status_hits") or []))
        try:
            packet = await service.analyze(PatientInput(case_id=case_id, patient_text=text))
        except Exception as exc:  # noqa: BLE001
            packet = {"error": f"{type(exc).__name__}: {exc}"}
        packet_dict = packet.model_dump() if hasattr(packet, "model_dump") else dict(packet)
        diff = packet_dict.get("differential", {}) or {}
        cands = diff.get("candidates", []) or []
        top = cands[0] if cands else {}
        top1_family = str(top.get("parent_category") or top.get("family_label") or top.get("label") or "").lower()
        confidence = float(top.get("score") or 0.0)
        decision = str(packet_dict.get("decision", "") or packet_dict.get("status", "")).lower()
        abstained = "abstain" in decision
        out.append(
            EvalRecord(
                case_id=case_id,
                expected_family=expected_family,
                must_not_miss=must_not_miss,
                packet=packet_dict,
                abstained=abstained,
                top1_family=top1_family,
                confidence=confidence,
            )
        )
    return out


def _markdown_report(report: dict[str, Any], suite: str) -> str:
    lines = [
        f"# Eval report — {suite}",
        "",
        f"- Cases: **{report.get('n', 0)}**",
        f"- Family hit @1: **{report.get('family_hit_at_1', 0):.3f}**",
        f"- Family hit @3: **{report.get('family_hit_at_3', 0):.3f}**",
        f"- Family hit @5: **{report.get('family_hit_at_5', 0):.3f}**",
        f"- Calibration ECE: **{report.get('calibration_ece', 0):.4f}** (lower is better)",
        f"- Hallucination rate: **{report.get('hallucination', {}).get('hallucination_rate', 0):.3f}**",
        f"- Safety pass rate (must-not-miss cases): **{report.get('safety', {}).get('safety_pass_rate', 0):.3f}**",
        f"- Abstention precision: **{report.get('abstention', {}).get('abstention_precision', 0):.3f}**",
        "",
        "## Margin distribution",
        "",
        "| Bucket | Count |",
        "|---|---|",
    ]
    for bucket, count in (report.get("margin_histogram") or {}).items():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.append("## Reliability diagram")
    lines.append("")
    lines.append("| Bin | Count | Mean conf | Mean acc |")
    lines.append("|---|---|---|---|")
    for row in (report.get("reliability_diagram") or []):
        lines.append(
            f"| {row['bin_lo']:.2f}–{row['bin_hi']:.2f} | {row['count']} | "
            f"{row.get('mean_conf', 0):.3f} | {row.get('mean_acc', 0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="CDSS typed eval harness")
    parser.add_argument("--packets", help="JSONL of pre-computed packets to score (offline mode)")
    parser.add_argument("--suite", choices=list(_SUITES) + ["all"], help="Live-run mode: benchmark suite name")
    parser.add_argument("--output", help="Output JSON path (default: artifacts/eval/<timestamp>.json)")
    parser.add_argument("--baseline", action="store_true", help="Also write artifacts/eval/baseline.json")
    args = parser.parse_args()

    if not args.packets and not args.suite:
        parser.error("provide either --packets or --suite")

    artifacts_dir = ROOT / "artifacts" / "eval"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    suite_label = "offline" if args.packets else args.suite

    if args.packets:
        records = _records_from_jsonl(Path(args.packets))
    elif args.suite == "all":
        records = []
        for s in _SUITES:
            records.extend(asyncio.run(_run_live_suite(s)))
    else:
        records = asyncio.run(_run_live_suite(args.suite))

    report = cohort_report(records)
    report["suite"] = suite_label
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.output) if args.output else artifacts_dir / f"{timestamp}_{suite_label}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_markdown_report(report, suite_label), encoding="utf-8")

    if args.baseline:
        (artifacts_dir / "baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Eval report: {out_path}")
    print(f"Markdown:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
