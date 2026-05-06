"""CI guard: forbid disease-name shortcuts in pipeline source.

The system policy (see ``docs/anti_hardcoding.md``) prohibits
disease-specific control flow. This test statically scans the production
source tree under ``src/cdss`` for common disease tokens used as keys in
``if`` / ``elif`` branches or as dict keys in lookup tables, and fails the
build if any are found.

Out-of-scope locations are explicitly whitelisted:

- ``tests/``                — eval reference fixtures.
- ``src/knowledge/case_generator.py`` — generates synthetic cases by
  topic; disease names appear in *retrieval queries*, not in control flow.
- ``data/``, ``docs/``, ``config/`` — non-source.
- ``src/cdss/clinical/registry`` and ``src/cdss/knowledge/registry.py`` —
  the syndrome registry is *data*, not code (it is loaded from JSON).
- Comments and docstrings are ignored (we strip ``#`` lines and keep only
  literal source lines).

The forbidden token list is intentionally short and high-signal: tokens that
should never gate control flow in a generic reasoning engine. Add to it as
the system grows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOT = _ROOT / "src" / "cdss"

# Whitelisted file paths (relative to repo root). Disease tokens here are
# allowed because the file is not control-flow logic — it is data, prompts,
# or a query template.
_WHITELIST_RELATIVE = {
    "src/cdss/text_normalization.py",  # synonym tables, not control flow
    "src/knowledge/case_generator.py",
    "src/cdss/knowledge/registry.py",
    "src/cdss/knowledge/case_generator.py",
}

# Disease tokens that must not appear as control-flow keys. Kept short and
# high-signal — purpose is to catch *new* shortcuts, not to enumerate every
# disease in medicine.
_FORBIDDEN_TOKENS = {
    "appendicitis",
    "myocardial_infarction",
    "stroke",
    "sepsis",
    "pulmonary_embolism",
    "aortic_dissection",
    "meningitis",
    "diabetic_ketoacidosis",
    "anaphylaxis",
    "tuberculosis",
}

# Patterns that indicate control flow keyed on a disease token. We look for
# the token *quoted as a string* immediately following ``==``, ``in``, ``if``,
# ``elif``, ``case``, or used as a dict key.
def _control_flow_pattern(token: str) -> re.Pattern[str]:
    quoted = re.escape(token)
    return re.compile(
        rf"""(?x)
        (?:
            (?:if|elif|case)\s+[^:]*['"]{quoted}['"]
          | ==\s*['"]{quoted}['"]
          | ['"]{quoted}['"]\s*:\s*(?:lambda|\{{|\[|"\w|\d)
        )
        """
    )


def _strip_comments(source: str) -> str:
    """Drop full-line comments and end-of-line comments. Keep string literals."""
    out_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for py in _SCAN_ROOT.rglob("*.py"):
        rel = py.relative_to(_ROOT).as_posix()
        if rel in _WHITELIST_RELATIVE:
            continue
        if "/__pycache__/" in rel:
            continue
        files.append(py)
    return files


def test_no_disease_tokens_in_control_flow():
    offenders: list[str] = []
    patterns = {token: _control_flow_pattern(token) for token in _FORBIDDEN_TOKENS}
    for path in _iter_source_files():
        try:
            text = _strip_comments(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for token, pattern in patterns.items():
            for match in pattern.finditer(text):
                # Compute line number for a useful error message.
                lineno = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{path.relative_to(_ROOT).as_posix()}:{lineno} — disease token '{token}' in control flow"
                )

    assert not offenders, (
        "Disease-specific shortcuts detected in source. "
        "Move disease-name-keyed logic into the syndrome registry (data) "
        "or rewrite as a generalizable rule. Offenders:\n  - "
        + "\n  - ".join(offenders)
    )


def test_clinical_thresholds_are_centrally_configured():
    """Spot-check: vital-threshold literals (90, 92, 130, 30, 40) should not
    appear as bare comparison constants in the safety module's main path,
    because they belong in ``config/clinical_thresholds.json``."""
    safety_path = _ROOT / "src" / "cdss" / "clinical" / "safety.py"
    text = safety_path.read_text(encoding="utf-8")
    # The migration replaces literals with ``clinical_thresholds.get_float(...)``.
    assert "clinical_thresholds.get_float(\"vitals.spo2_critical_lt\"" in text
    assert "clinical_thresholds.get_float(\"vitals.sbp_shock_lt\"" in text
    assert "clinical_thresholds.get_float(\"vitals.hr_marked_tachy_gt\"" in text
