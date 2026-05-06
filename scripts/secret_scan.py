from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
]

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".ps1",
    ".bat",
    ".sh",
    ".env",
}


def _should_skip(path: Path) -> bool:
    lowered = path.as_posix().lower()
    if path.name in {".env", ".env.local"}:
        return True
    skip_markers = (
        "/.git/",
        "/.venv/",
        "/env/",
        "/models/",
        "/output/",
        "/artifacts/",
        "/_removed_legacy_disabled/",
        "__pycache__",
    )
    if any(marker in lowered for marker in skip_markers):
        return True
    if path.suffix.lower() and path.suffix.lower() not in TEXT_EXTENSIONS:
        return True
    return False


def _iter_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repo for leaked high-confidence secrets.")
    parser.add_argument(
        "--out-json",
        default="artifacts/verification/secret_scan_report.json",
        help="JSON report output path.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    findings: list[dict[str, object]] = []

    for file_path in _iter_files(project_root):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if "secret-scan: allow" in line.lower():
                continue
            for kind, pattern in SECRET_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                findings.append(
                    {
                        "file": file_path.relative_to(project_root).as_posix(),
                        "line": idx,
                        "kind": kind,
                        "match": match.group(0)[:12] + "...",
                    }
                )

    report = {
        "summary": {
            "files_scanned": len(_iter_files(project_root)),
            "finding_count": len(findings),
            "status": "fail" if findings else "pass",
        },
        "findings": findings,
    }

    out_path = project_root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if findings:
        print(f"Secret scan failed with {len(findings)} finding(s).")
        for row in findings[:20]:
            print(f"- {row['file']}:{row['line']} ({row['kind']})")
        return 1

    print(f"Secret scan passed. Files scanned: {report['summary']['files_scanned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
