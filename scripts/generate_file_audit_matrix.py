from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_audit_utils import analyze_module, evaluate_budget, iter_python_files, load_budget_config, resolve_budget


def _tier(loc: int, max_cc: int) -> str:
    if loc <= 350 and max_cc <= 12:
        return "low"
    if loc <= 700 and max_cc <= 20:
        return "medium"
    return "high"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a file audit matrix artifact.")
    parser.add_argument(
        "--config",
        default="config/quality/complexity_budgets.json",
        help="Budget config JSON path.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/verification/file_audit_matrix.json",
        help="Matrix JSON output path.",
    )
    parser.add_argument(
        "--out-md",
        default="artifacts/verification/file_audit_matrix.md",
        help="Matrix markdown output path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_budget_config(project_root / args.config)

    modules: list[dict[str, object]] = []
    failing_count = 0
    for path in iter_python_files(project_root):
        audit = analyze_module(path, project_root)
        budget = resolve_budget(audit.path, config)
        violations = evaluate_budget(audit, budget)
        status = "fail" if violations else "pass"
        if status == "fail":
            failing_count += 1
        modules.append(
            {
                "path": audit.path,
                "loc": audit.loc,
                "function_count": audit.function_count,
                "class_count": audit.class_count,
                "max_function_cc": audit.max_function_cc,
                "avg_function_cc": audit.avg_function_cc,
                "complexity_tier": _tier(audit.loc, audit.max_function_cc),
                "budget": {
                    "max_loc": budget.max_loc,
                    "max_function_cc": budget.max_function_cc,
                },
                "status": status,
                "violations": violations,
            }
        )

    modules.sort(key=lambda row: (row["status"] != "fail", -int(row["loc"]), str(row["path"])))
    payload = {
        "summary": {
            "module_count": len(modules),
            "failing_count": failing_count,
            "passing_count": len(modules) - failing_count,
        },
        "modules": modules,
    }

    out_json = project_root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# File Audit Matrix",
        "",
        f"- Module count: {payload['summary']['module_count']}",
        f"- Passing: {payload['summary']['passing_count']}",
        f"- Failing: {payload['summary']['failing_count']}",
        "",
        "| File | LOC | Fn | Max CC | Tier | Budget LOC | Budget CC | Status |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in modules:
        md_lines.append(
            f"| {row['path']} | {row['loc']} | {row['function_count']} | {row['max_function_cc']} | {row['complexity_tier']} | {row['budget']['max_loc']} | {row['budget']['max_function_cc']} | {row['status']} |"
        )

    out_md = project_root / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote file audit matrix to {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
