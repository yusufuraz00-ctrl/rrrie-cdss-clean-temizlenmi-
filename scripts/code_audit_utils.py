from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ModuleAudit:
    path: str
    loc: int
    function_count: int
    class_count: int
    max_function_cc: int
    avg_function_cc: float


@dataclass(frozen=True)
class ComplexityBudget:
    max_loc: int
    max_function_cc: int


_DEFAULT_CONFIG = {
    "defaults": {
        "max_loc": 900,
        "max_function_cc": 24,
    },
    "overrides": [
        {"pattern": "run.py", "max_loc": 1300, "max_function_cc": 36},
        {"pattern": "src/cdss/app/service.py", "max_loc": 1800, "max_function_cc": 48},
    ],
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _function_cyclomatic_complexity(node: ast.AST) -> int:
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.IfExp, ast.Assert)):
            score += 1
        elif isinstance(child, ast.Try):
            score += 1 + len(child.handlers)
        elif isinstance(child, ast.BoolOp):
            score += max(1, len(child.values) - 1)
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            score += max(1, len(child.generators))
        elif isinstance(child, ast.Match):
            score += max(1, len(child.cases))
    return score


def _analyze_tree(tree: ast.AST) -> tuple[int, int, int, float]:
    function_scores: list[int] = []
    class_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_scores.append(_function_cyclomatic_complexity(node))
    if not function_scores:
        return 0, class_count, 0, 0.0
    return len(function_scores), class_count, max(function_scores), round(sum(function_scores) / len(function_scores), 2)


def _should_skip(path: Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    skip_markers = (
        "/.venv/",
        "/env/",
        "/_removed_legacy_disabled/",
        "/output/",
        "/models/",
        "/data/",
        "/artifacts/",
        "/.git/",
        "__pycache__",
    )
    return any(marker in lowered for marker in skip_markers)


def iter_python_files(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    include_dirs = [project_root / "src", project_root / "gui", project_root / "scripts"]
    for directory in include_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            if _should_skip(path):
                continue
            candidates.append(path)
    run_py = project_root / "run.py"
    if run_py.exists() and not _should_skip(run_py):
        candidates.append(run_py)
    return sorted(set(candidates))


def analyze_module(path: Path, project_root: Path) -> ModuleAudit:
    source = _read_text(path)
    loc = _line_count(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.parse("")
    function_count, class_count, max_function_cc, avg_function_cc = _analyze_tree(tree)
    rel_path = path.relative_to(project_root).as_posix()
    return ModuleAudit(
        path=rel_path,
        loc=loc,
        function_count=function_count,
        class_count=class_count,
        max_function_cc=max_function_cc,
        avg_function_cc=avg_function_cc,
    )


def load_budget_config(config_path: Path) -> dict:
    if not config_path.exists():
        return _DEFAULT_CONFIG
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return _DEFAULT_CONFIG
    merged = {
        "defaults": dict(_DEFAULT_CONFIG["defaults"]),
        "overrides": list(_DEFAULT_CONFIG["overrides"]),
    }
    defaults = parsed.get("defaults", {})
    if isinstance(defaults, dict):
        merged["defaults"].update({k: int(v) for k, v in defaults.items() if k in {"max_loc", "max_function_cc"}})
    overrides = parsed.get("overrides", [])
    if isinstance(overrides, list):
        merged["overrides"] = []
        for row in overrides:
            if not isinstance(row, dict):
                continue
            pattern = str(row.get("pattern", "")).strip()
            if not pattern:
                continue
            merged["overrides"].append(
                {
                    "pattern": pattern,
                    "max_loc": int(row.get("max_loc", merged["defaults"]["max_loc"])),
                    "max_function_cc": int(row.get("max_function_cc", merged["defaults"]["max_function_cc"])),
                }
            )
    return merged


def resolve_budget(path: str, config: dict) -> ComplexityBudget:
    defaults = config.get("defaults", {})
    max_loc = int(defaults.get("max_loc", _DEFAULT_CONFIG["defaults"]["max_loc"]))
    max_cc = int(defaults.get("max_function_cc", _DEFAULT_CONFIG["defaults"]["max_function_cc"]))
    for row in config.get("overrides", []):
        pattern = str(row.get("pattern", "")).strip()
        if pattern and fnmatch.fnmatch(path, pattern):
            max_loc = int(row.get("max_loc", max_loc))
            max_cc = int(row.get("max_function_cc", max_cc))
            break
    return ComplexityBudget(max_loc=max_loc, max_function_cc=max_cc)


def evaluate_budget(audit: ModuleAudit, budget: ComplexityBudget) -> list[str]:
    violations: list[str] = []
    if audit.loc > budget.max_loc:
        violations.append(f"loc {audit.loc}>{budget.max_loc}")
    if audit.max_function_cc > budget.max_function_cc:
        violations.append(f"max_cc {audit.max_function_cc}>{budget.max_function_cc}")
    return violations
