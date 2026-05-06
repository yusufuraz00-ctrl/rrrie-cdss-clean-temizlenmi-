"""Build the generated syndrome registry artifact from its source manifest."""

from __future__ import annotations

from pathlib import Path

from src.cdss.knowledge.registry import build_syndrome_registry_artifact


def main() -> None:
    artifact_path = build_syndrome_registry_artifact()
    print(f"Built syndrome registry artifact: {artifact_path}")


if __name__ == "__main__":
    main()
