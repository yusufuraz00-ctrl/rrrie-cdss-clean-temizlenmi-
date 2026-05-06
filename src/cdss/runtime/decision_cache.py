"""Small persistent runtime decision cache for CDSS bootstrap."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.cdss.runtime.hardware import GPUHardwareInfo


CACHE_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_runtime_decision_cache_path(project_root: Path | None = None) -> Path:
    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    return root / "data" / "runtime_decision_cache.json"


def _runtime_signature(
    hardware: GPUHardwareInfo,
    main_model: Path | None,
    dllm_model: Path | None,
    server_exe: Path | None,
) -> dict[str, Any]:
    return {
        "gpu_name": hardware.name or "",
        "compute_cap": hardware.compute_cap,
        "driver_version": hardware.driver_version or "",
        "total_vram_gb": hardware.total_vram_gb,
        "main_model": main_model.name if main_model else "",
        "dllm_model": dllm_model.name if dllm_model else "",
        "server_exe": server_exe.name if server_exe else "",
    }


def build_runtime_decision_key(
    hardware: GPUHardwareInfo,
    main_model: Path | None,
    dllm_model: Path | None,
    server_exe: Path | None,
) -> str:
    signature = _runtime_signature(hardware, main_model, dllm_model, server_exe)
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:20]


def load_runtime_decision_cache(project_root: Path | None = None) -> dict[str, Any]:
    path = get_runtime_decision_cache_path(project_root)
    if not path.exists():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    payload["version"] = CACHE_VERSION
    return payload


def save_runtime_decision_cache(payload: dict[str, Any], project_root: Path | None = None) -> None:
    path = get_runtime_decision_cache_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def get_cached_runtime_decision(
    project_root: Path | None,
    *,
    hardware: GPUHardwareInfo,
    main_model: Path | None,
    dllm_model: Path | None,
    server_exe: Path | None,
) -> dict[str, Any] | None:
    cache = load_runtime_decision_cache(project_root)
    key = build_runtime_decision_key(hardware, main_model, dllm_model, server_exe)
    entry = cache.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None
    return entry


def record_runtime_decision(
    project_root: Path | None,
    *,
    hardware: GPUHardwareInfo,
    main_model: Path | None,
    dllm_model: Path | None,
    server_exe: Path | None,
    runtime_profile: str,
    server_topology: str,
    dllm_mode: str,
    binary_compatibility: str,
    post_load_free_vram_gb: float | None,
    stable: bool,
    degrade_reason: str = "",
) -> dict[str, Any]:
    cache = load_runtime_decision_cache(project_root)
    entries = cache.setdefault("entries", {})
    key = build_runtime_decision_key(hardware, main_model, dllm_model, server_exe)
    current = dict(entries.get(key, {}))

    stable_runs = int(current.get("stable_runs", 0) or 0)
    degraded_runs = int(current.get("degraded_runs", 0) or 0)
    if stable:
        stable_runs += 1
    else:
        degraded_runs += 1

    entry = {
        **current,
        "key": key,
        "signature": _runtime_signature(hardware, main_model, dllm_model, server_exe),
        "preferred_profile": runtime_profile,
        "preferred_topology": server_topology,
        "preferred_dllm_mode": dllm_mode,
        "binary_compatibility": binary_compatibility,
        "post_load_free_vram_gb": post_load_free_vram_gb,
        "degrade_reason": degrade_reason,
        "stable_runs": stable_runs,
        "degraded_runs": degraded_runs,
        "last_updated": _utc_now(),
    }
    entries[key] = entry
    save_runtime_decision_cache(cache, project_root)
    return entry
