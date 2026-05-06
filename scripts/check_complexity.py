from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_audit_utils import analyze_module, evaluate_budget, iter_python_files, load_budget_config, resolve_budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Check module-level complexity budgets.")
    parser.add_argument(
        "--config",
        default="config/quality/complexity_budgets.json",
        help="Budget config JSON path.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/verification/complexity_report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--report-md",
        default="artifacts/verification/complexity_report.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_budget_config(project_root / args.config)

    rows: list[dict[str, object]] = []
    failing: list[dict[str, object]] = []
    for path in iter_python_files(project_root):
        audit = analyze_module(path, project_root)
        budget = resolve_budget(audit.path, config)
        violations = evaluate_budget(audit, budget)
        row = {
            "path": audit.path,
            "loc": audit.loc,
            "function_count": audit.function_count,
            "class_count": audit.class_count,
            "max_function_cc": audit.max_function_cc,
            "avg_function_cc": audit.avg_function_cc,
            "budget_max_loc": budget.max_loc,
            "budget_max_function_cc": budget.max_function_cc,
            "violations": violations,
            "status": "fail" if violations else "pass",
        }
        rows.append(row)
        if violations:
            failing.append(row)

    report = {
        "summary": {
            "module_count": len(rows),
            "failing_count": len(failing),
            "status": "fail" if failing else "pass",
        },
        "modules": rows,
    }

    json_out = project_root / args.report_json
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Complexity Budget Report",
        "",
        f"- Module count: {len(rows)}",
        f"- Failing modules: {len(failing)}",
        f"- Status: {report['summary']['status']}",
        "",
        "| Module | LOC | Max CC | Budget LOC | Budget CC | Status | Violations |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        violation_text = ", ".join(row["violations"]) if row["violations"] else "-"
        md_lines.append(
            f"| {row['path']} | {row['loc']} | {row['max_function_cc']} | {row['budget_max_loc']} | {row['budget_max_function_cc']} | {row['status']} | {violation_text} |"
        )

    md_out = project_root / args.report_md
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if failing:
        print(f"Complexity budget check failed: {len(failing)} module(s) exceeded budget.")
        for row in failing[:20]:
            print(f"- {row['path']}: {', '.join(row['violations'])}")
        return 1

    print(f"Complexity budget check passed for {len(rows)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
