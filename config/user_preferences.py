"""Persisted user preferences for local runtime defaults."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_OPERATION_MODES = {
    "strict_offline",
    "local_core_with_online_r2",
    "cloud_reference",
}

VALID_RUNTIME_PROFILES = {
    "auto",
    "standard_6gb",
    "compact_4gb",
    "legacy_4gb",
    "cpu_only",
}

VALID_DLLM_MODES = {
    "auto",
    "full",
    "lite",
    "deterministic",
    "off",
}

VALID_REASONING_MODES = {"fast", "thinking", "deep"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _quote(value: str) -> str:
    escaped = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _normalize_choice(value: str | None, valid: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in valid else default


@dataclass(frozen=True)
class UserPreferences:
    operation_mode_default: str = "local_core_with_online_r2"
    runtime_profile_default: str = "auto"
    runtime_profile_dllm_mode: str = "auto"
    reasoning_mode: str = "fast"  # "fast" | "thinking" | "deep"
    updated_at: str = ""
    source: str = "defaults"

    @property
    def atom_mode(self) -> bool:
        return self.reasoning_mode == "deep"

    def to_dict(self) -> dict:
        return {
            "operation_mode_default": self.operation_mode_default,
            "runtime_profile_default": self.runtime_profile_default,
            "runtime_profile_dllm_mode": self.runtime_profile_dllm_mode,
            "reasoning_mode": self.reasoning_mode,
            "atom_mode": self.atom_mode,
            "updated_at": self.updated_at,
            "source": self.source,
        }


def get_user_preferences_path(project_root: Path | None = None) -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    return root / "data" / "settings.toml"


def load_user_preferences(project_root: Path | None = None) -> UserPreferences:
    path = get_user_preferences_path(project_root)
    if not path.exists():
        return UserPreferences(source="defaults")

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return UserPreferences(source="defaults_invalid")

    return UserPreferences(
        operation_mode_default=_normalize_choice(
            payload.get("operation_mode_default"),
            VALID_OPERATION_MODES,
            "local_core_with_online_r2",
        ),
        runtime_profile_default=_normalize_choice(
            payload.get("runtime_profile_default"),
            VALID_RUNTIME_PROFILES,
            "auto",
        ),
        runtime_profile_dllm_mode=_normalize_choice(
            payload.get("runtime_profile_dllm_mode"),
            VALID_DLLM_MODES,
            "auto",
        ),
        reasoning_mode=_normalize_choice(
            payload.get("reasoning_mode"),
            VALID_REASONING_MODES,
            "fast",
        ),
        updated_at=str(payload.get("updated_at", "") or ""),
        source="user",
    )


def save_user_preferences(
    project_root: Path | None = None,
    *,
    operation_mode_default: str | None = None,
    runtime_profile_default: str | None = None,
    runtime_profile_dllm_mode: str | None = None,
    reasoning_mode: str | None = None,
) -> UserPreferences:
    current = load_user_preferences(project_root)
    updated = UserPreferences(
        operation_mode_default=_normalize_choice(
            operation_mode_default if operation_mode_default is not None else current.operation_mode_default,
            VALID_OPERATION_MODES,
            "local_core_with_online_r2",
        ),
        runtime_profile_default=_normalize_choice(
            runtime_profile_default if runtime_profile_default is not None else current.runtime_profile_default,
            VALID_RUNTIME_PROFILES,
            "auto",
        ),
        runtime_profile_dllm_mode=_normalize_choice(
            runtime_profile_dllm_mode if runtime_profile_dllm_mode is not None else current.runtime_profile_dllm_mode,
            VALID_DLLM_MODES,
            "auto",
        ),
        reasoning_mode=_normalize_choice(
            reasoning_mode if reasoning_mode is not None else current.reasoning_mode,
            VALID_REASONING_MODES,
            "fast",
        ),
        updated_at=_utc_now(),
        source="user",
    )

    path = get_user_preferences_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# RRRIE-CDSS persisted user preferences",
                "# This file is safe to edit manually.",
                f"operation_mode_default = {_quote(updated.operation_mode_default)}",
                f"runtime_profile_default = {_quote(updated.runtime_profile_default)}",
                f"runtime_profile_dllm_mode = {_quote(updated.runtime_profile_dllm_mode)}",
                f"reasoning_mode = {_quote(updated.reasoning_mode)}",
                f"updated_at = {_quote(updated.updated_at)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return updated


def merged_runtime_defaults(settings: object, project_root: Path | None = None) -> UserPreferences:
    prefs = load_user_preferences(project_root)
    return UserPreferences(
        operation_mode_default=_normalize_choice(
            prefs.operation_mode_default or getattr(settings, "operation_mode_default", "local_core_with_online_r2"),
            VALID_OPERATION_MODES,
            "local_core_with_online_r2",
        ),
        runtime_profile_default=_normalize_choice(
            prefs.runtime_profile_default or getattr(settings, "runtime_profile_default", "auto"),
            VALID_RUNTIME_PROFILES,
            "auto",
        ),
        runtime_profile_dllm_mode=_normalize_choice(
            prefs.runtime_profile_dllm_mode or getattr(settings, "runtime_profile_dllm_mode", "auto"),
            VALID_DLLM_MODES,
            "auto",
        ),
        reasoning_mode=_normalize_choice(
            prefs.reasoning_mode or getattr(settings, "reasoning_mode", "fast"),
            VALID_REASONING_MODES,
            "fast",
        ),
        updated_at=prefs.updated_at,
        source=prefs.source,
    )
