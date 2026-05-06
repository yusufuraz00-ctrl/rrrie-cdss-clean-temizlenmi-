"""Native runtime asset resolution for launcher and GUI health snapshots."""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.user_preferences import load_user_preferences
from src.cdss.runtime.decision_cache import get_cached_runtime_decision
from src.cdss.runtime.hardware import (
    FLASH_ATTN_MIN_COMPUTE_CAP,
    GPUHardwareInfo,
    STANDARD_6GB_TOTAL_THRESHOLD,
    VALID_SERVER_TOPOLOGIES,
    detect_gpu_hardware,
    probe_llama_server_binary_compatibility,
    resolve_dllm_mode,
    resolve_flash_attn_mode,
    resolve_runtime_profile,
    resolve_server_topology,
)
from src.utils.llama_manager import has_valid_gguf_header


PROFILE_CTX_CAPS = {
    "standard_6gb": 4096,
    "compact_4gb": 2048,
    "legacy_4gb": 1536,
    "cpu_only": 1536,
}

DEFAULT_LLAMA_HOST = "127.0.0.1"
DEFAULT_MAIN_PORT = 8080
DEFAULT_DLLM_PORT = 8081


@dataclass
class RuntimeAssetResolution:
    project_root: Path
    env_file: Path
    requested_profile: str
    requested_dllm_mode: str
    hardware: GPUHardwareInfo
    primary_server_exe: Path | None
    legacy_server_exe: Path | None
    selected_server_exe: Path | None
    fit_params_exe: Path | None
    binary_compatibility: str
    runtime_profile: str
    dllm_mode: str
    server_topology: str
    main_model: Path | None
    dllm_model: Path | None
    effective_ctx: int
    allow_4b_escalation: bool
    degrade_reason: str
    flash_attn_mode: str
    llama_server_url: str
    dllm_server_url: str
    decision_source: str = "heuristic"
    cached_decision: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fatal_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.runtime_profile

    @property
    def detected_vram_gb(self) -> float | None:
        return self.hardware.detected_vram_gb

    @property
    def total_vram_gb(self) -> float | None:
        return self.hardware.total_vram_gb

    @property
    def free_vram_gb(self) -> float | None:
        return self.hardware.free_vram_gb

    @free_vram_gb.setter
    def free_vram_gb(self, value: float | None) -> None:
        self.hardware = GPUHardwareInfo(
            name=self.hardware.name,
            total_vram_gb=self.hardware.total_vram_gb,
            free_vram_gb=value,
            compute_cap=self.hardware.compute_cap,
            driver_version=self.hardware.driver_version,
        )

    @property
    def compute_cap(self) -> float | None:
        return self.hardware.compute_cap

    @property
    def driver_version(self) -> str:
        return self.hardware.driver_version

    def to_doctor_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "project_root",
            "env_file",
            "primary_server_exe",
            "legacy_server_exe",
            "selected_server_exe",
            "fit_params_exe",
            "main_model",
            "dllm_model",
        ):
            value = payload.get(key)
            payload[key] = str(value) if value else ""
        payload["detected_vram_gb"] = self.detected_vram_gb
        payload["total_vram_gb"] = self.total_vram_gb
        payload["free_vram_gb"] = self.free_vram_gb
        payload["compute_cap"] = self.compute_cap
        payload["is_ready"] = not self.fatal_issues
        return payload

    def to_doctor_json(self) -> str:
        return json.dumps(self.to_doctor_dict(), indent=2, ensure_ascii=False)


def _read_env_value(env_file: Path, key: str, default: str = "") -> str:
    if key in os.environ:
        return str(os.environ.get(key, "")).strip()
    if not env_file.exists():
        return default
    try:
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_key, env_value = stripped.split("=", 1)
            if env_key.strip() == key:
                return env_value.strip().strip('"').strip("'")
    except Exception:
        return default
    return default


def _read_env_float(env_file: Path, key: str, default: float = 0.0) -> float:
    raw = _read_env_value(env_file, key, str(default))
    try:
        return float(raw or default)
    except (TypeError, ValueError):
        return default


def _read_env_bool(env_file: Path, key: str, default: bool = False) -> bool:
    raw = _read_env_value(env_file, key, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ctx_cap_for_profile(env_file: Path, profile: str) -> int:
    compact_ctx = int(_read_env_float(env_file, "RUNTIME_PROFILE_COMPACT_CTX", PROFILE_CTX_CAPS["compact_4gb"]))
    if profile == "compact_4gb":
        return max(1024, min(compact_ctx, PROFILE_CTX_CAPS["compact_4gb"]))
    return PROFILE_CTX_CAPS.get(profile, PROFILE_CTX_CAPS["legacy_4gb"])


def _valid_gguf_candidate(path: Path, min_size: int = 0) -> bool:
    return path.is_file() and path.stat().st_size >= min_size and has_valid_gguf_header(path)


def _resolve_binary_path(env_file: Path, env_key: str, default_candidates: list[Path], path_candidates: list[str]) -> Path | None:
    env_path = _read_env_value(env_file, env_key, "")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(default_candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for binary_name in path_candidates:
        resolved = shutil.which(binary_name)
        if resolved:
            return Path(resolved)
    return None


def resolve_llama_server_exe(project_root: Path, env_file: Path) -> Path | None:
    return _resolve_binary_path(
        env_file,
        "LLAMA_SERVER_EXE",
        [project_root / "llama-server" / "llama-server.exe"],
        ["llama-server.exe", "llama-server"],
    )


def resolve_legacy_llama_server_exe(project_root: Path, env_file: Path) -> Path | None:
    return _resolve_binary_path(
        env_file,
        "LLAMA_SERVER_EXE_LEGACY",
        [
            project_root / "llama-server" / "llama-server-legacy.exe",
            project_root / "llama-server" / "llama-server-sm75.exe",
            project_root / "llama-server" / "llama-server-cc75.exe",
        ],
        [],
    )


def resolve_llama_fit_params_exe(project_root: Path, env_file: Path) -> Path | None:
    return _resolve_binary_path(
        env_file,
        "LLAMA_FIT_PARAMS_EXE",
        [project_root / "llama-server" / "llama-fit-params.exe"],
        [],
    )


def _search_model_candidates(project_root: Path, env_file: Path) -> tuple[Path | None, Path | None]:
    main_model: Path | None = None
    dllm_model: Path | None = None

    main_env = _read_env_value(env_file, "MAIN_GGUF_MODEL_PATH", _read_env_value(env_file, "GGUF_MODEL_PATH", ""))
    dllm_env = _read_env_value(env_file, "DLLM_GGUF_MODEL_PATH", _read_env_value(env_file, "DLLM_MODEL_PATH", ""))

    if main_env:
        candidate = Path(main_env)
        if _valid_gguf_candidate(candidate, 100_000_000):
            main_model = candidate

    if dllm_env:
        candidate = Path(dllm_env)
        if _valid_gguf_candidate(candidate, 100_000_000):
            dllm_model = candidate

    if not main_model:
        patterns = [
            str(project_root / "models" / "Qwen3.5-4B*.gguf"),
            str(project_root / "models" / "*4B*.gguf"),
            str(project_root / "models" / "*4b*.gguf"),
        ]
        for pattern in patterns:
            for match in glob.glob(pattern):
                candidate = Path(match)
                if _valid_gguf_candidate(candidate, 100_000_000):
                    main_model = candidate
                    break
            if main_model:
                break

    if not dllm_model:
        patterns = [
            str(project_root / "models" / "Qwen3.5-0.8B*.gguf"),
            str(project_root / "models" / "*0.8B*.gguf"),
            str(project_root / "models" / "*0.5B*.gguf"),
        ]
        for pattern in patterns:
            for match in glob.glob(pattern):
                candidate = Path(match)
                if _valid_gguf_candidate(candidate, 100_000_000):
                    dllm_model = candidate
                    break
            if dllm_model:
                break

    return main_model, dllm_model


def _normalize_url(url: str, default: str) -> str:
    normalized = (url or default).strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized or default


def _derive_degrade_reason(
    runtime_profile: str,
    dllm_mode: str,
    server_topology: str,
    binary_compatibility: str,
    *,
    dllm_model: Path | None,
) -> str:
    if binary_compatibility in {"legacy_required_missing", "incompatible", "cpu_only"}:
        return binary_compatibility
    if server_topology == "shared_main_server" and runtime_profile == "standard_6gb" and dllm_model is None:
        return "shared_main_server_missing_dllm_model"
    if server_topology == "shared_main_server" and runtime_profile in {"compact_4gb", "legacy_4gb"}:
        return "shared_main_server_low_free_vram"
    if dllm_mode in {"deterministic", "off"}:
        return f"dllm_{dllm_mode}"
    if runtime_profile == "compact_4gb":
        return "compact_runtime"
    if runtime_profile == "legacy_4gb":
        return "legacy_runtime"
    return ""


def _build_runtime_recommendations(
    *,
    hardware: GPUHardwareInfo,
    runtime_profile: str,
    binary_compatibility: str,
    main_model: Path | None,
    dllm_model: Path | None,
    fit_params_exe: Path | None,
    selected_server_exe: Path | None,
) -> list[str]:
    recommendations: list[str] = []
    gpu_missing = not hardware.name and hardware.total_vram_gb is None and hardware.free_vram_gb is None and hardware.compute_cap is None
    if gpu_missing:
        recommendations.append("NVIDIA GPU algilanamadi. Surucuyu ve nvidia-smi erisimini dogrulayin.")
    if binary_compatibility == "legacy_required_missing":
        recommendations.append("Bu cihaz icin legacy uyumlu llama-server binary gerekli ama bulunamadi.")
    elif binary_compatibility in {"incompatible", "cpu_only"}:
        recommendations.append("GPU ile uyumlu llama-server binary kullanin.")
    elif selected_server_exe is None:
        recommendations.append("llama-server.exe bulunamadi. setup ile kurun veya LLAMA_SERVER_EXE belirtin.")
    if main_model is None:
        recommendations.append("Ana 4B GGUF modeli bulunamadi.")
    if runtime_profile == "standard_6gb" and dllm_model is None:
        recommendations.append("0.8B DLLM modeli yok; sistem shared_main_server ile calisacak.")
    if fit_params_exe is None:
        recommendations.append("llama-fit-params.exe yok; statik launch parametreleri kullanilacak.")
    if hardware.compute_cap is not None and hardware.compute_cap < FLASH_ATTN_MIN_COMPUTE_CAP:
        recommendations.append("Pre-Ampere kartta Flash Attention kapali kalmali.")
    return recommendations


def resolve_runtime_assets(project_root: Path | None = None, *, env_file: Path | None = None) -> RuntimeAssetResolution:
    project_root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    env_file = (env_file or project_root / ".env").resolve()
    user_preferences = load_user_preferences(project_root)

    requested_profile = _read_env_value(env_file, "RUNTIME_PROFILE_DEFAULT", user_preferences.runtime_profile_default).lower()
    requested_dllm_mode = _read_env_value(env_file, "RUNTIME_PROFILE_DLLM_MODE", user_preferences.runtime_profile_dllm_mode).lower()
    requested_server_topology = _read_env_value(env_file, "RUNTIME_SERVER_TOPOLOGY", "").lower()
    hardware = detect_gpu_hardware(_read_env_float(env_file, "RUNTIME_PROFILE_VRAM_GB_OVERRIDE", 0.0))

    primary_server_exe = resolve_llama_server_exe(project_root, env_file)
    legacy_server_exe = resolve_legacy_llama_server_exe(project_root, env_file)
    fit_params_exe = resolve_llama_fit_params_exe(project_root, env_file)

    binary_compatibility = probe_llama_server_binary_compatibility(primary_server_exe)
    selected_server_exe = primary_server_exe
    if binary_compatibility == "legacy_required":
        if legacy_server_exe:
            selected_server_exe = legacy_server_exe
            binary_compatibility = "legacy_selected"
        else:
            selected_server_exe = None
            binary_compatibility = "legacy_required_missing"
    elif binary_compatibility == "missing":
        selected_server_exe = None

    runtime_profile = resolve_runtime_profile(
        requested_profile,
        detected_vram_gb=hardware.total_vram_gb or hardware.detected_vram_gb,
        free_vram_gb=hardware.free_vram_gb,
        binary_compatibility=binary_compatibility,
    )
    main_model, dllm_model = _search_model_candidates(project_root, env_file)
    cached_decision = get_cached_runtime_decision(
        project_root,
        hardware=hardware,
        main_model=main_model,
        dllm_model=dllm_model,
        server_exe=selected_server_exe,
    )
    decision_source = "heuristic"
    if requested_profile == "auto" and cached_decision:
        cached_profile = str(cached_decision.get("preferred_profile", "") or "").strip().lower()
        stable_runs = int(cached_decision.get("stable_runs", 0) or 0)
        total_vram = hardware.total_vram_gb or hardware.detected_vram_gb or 0.0
        if stable_runs > 0 and runtime_profile == "compact_4gb" and cached_profile == "standard_6gb" and total_vram >= STANDARD_6GB_TOTAL_THRESHOLD:
            runtime_profile = "standard_6gb"
            decision_source = "cache_promoted"

    dllm_mode = resolve_dllm_mode(requested_dllm_mode, runtime_profile, binary_compatibility=binary_compatibility)
    if requested_dllm_mode == "auto" and cached_decision:
        cached_dllm_mode = str(cached_decision.get("preferred_dllm_mode", "") or "").strip().lower()
        if cached_dllm_mode in {"full", "lite", "deterministic", "off"}:
            dllm_mode = cached_dllm_mode
            if decision_source == "heuristic":
                decision_source = "cache_hint"

    server_topology = resolve_server_topology(
        requested_server_topology,
        runtime_profile,
        dllm_mode,
        dllm_model_available=dllm_model is not None,
        free_vram_gb=hardware.free_vram_gb,
    )
    if requested_server_topology not in VALID_SERVER_TOPOLOGIES and cached_decision:
        cached_profile = str(cached_decision.get("preferred_profile", "") or "").strip().lower()
        cached_topology = str(cached_decision.get("preferred_topology", "") or "").strip().lower()
        if runtime_profile == cached_profile and cached_topology in VALID_SERVER_TOPOLOGIES:
            server_topology = cached_topology
            if decision_source == "heuristic":
                decision_source = "cache_hint"

    allow_4b_escalation = (
        runtime_profile == "standard_6gb"
        and dllm_mode == "full"
        and server_topology == "dual_server"
        and _read_env_bool(env_file, "RUNTIME_ALLOW_4B_ESCALATION", default=True)
    )
    degrade_reason = _derive_degrade_reason(runtime_profile, dllm_mode, server_topology, binary_compatibility, dllm_model=dllm_model)

    main_url = _normalize_url(_read_env_value(env_file, "LLAMA_SERVER_URL", f"http://{DEFAULT_LLAMA_HOST}:{DEFAULT_MAIN_PORT}"), f"http://{DEFAULT_LLAMA_HOST}:{DEFAULT_MAIN_PORT}")
    dllm_default_url = main_url if server_topology == "shared_main_server" else f"http://{DEFAULT_LLAMA_HOST}:{DEFAULT_DLLM_PORT}"
    dllm_url = _normalize_url(_read_env_value(env_file, "DLLM_SERVER_URL", dllm_default_url), dllm_default_url)
    if server_topology == "shared_main_server":
        dllm_url = main_url

    warnings: list[str] = []
    fatal_issues: list[str] = []
    gpu_missing = not hardware.name and hardware.total_vram_gb is None and hardware.free_vram_gb is None and hardware.compute_cap is None

    if not env_file.exists():
        warnings.append(".env missing; defaults will be used until you create it.")
    if gpu_missing:
        fatal_issues.append("No NVIDIA GPU/runtime was detected. Local execution is only guaranteed on Windows + NVIDIA.")
    if selected_server_exe is None:
        if binary_compatibility == "legacy_required_missing":
            fatal_issues.append("This GPU requires a legacy-compatible llama-server binary, but no legacy binary was found.")
        elif primary_server_exe is None:
            fatal_issues.append("No compatible llama-server executable was found.")
        else:
            fatal_issues.append(f"Selected llama-server binary is incompatible with this GPU ({binary_compatibility}).")
    if main_model is None:
        fatal_issues.append("Main 4B GGUF model not found.")
    if runtime_profile == "cpu_only":
        fatal_issues.append("Resolved runtime profile is cpu_only.")

    if runtime_profile == "standard_6gb" and dllm_model is None:
        warnings.append("DLLM 0.8B model not found. Shared main server topology will be used.")
    if decision_source == "cache_promoted":
        warnings.append("Runtime decision cache promoted this device from compact_4gb to standard_6gb.")
    elif decision_source == "cache_hint":
        warnings.append("Runtime decision cache applied a previously stable topology/profile hint for this device.")
    if runtime_profile in {"compact_4gb", "legacy_4gb"}:
        warnings.append(f"{runtime_profile} resolved; DLLM requests will route through shared_main_server.")
    if fit_params_exe is None:
        warnings.append("llama-fit-params.exe not found. Static launch parameters will be used.")

    flash_attn_mode = resolve_flash_attn_mode(hardware.compute_cap)
    if hardware.compute_cap is not None and hardware.compute_cap < FLASH_ATTN_MIN_COMPUTE_CAP:
        warnings.append(
            f"GPU compute capability {hardware.compute_cap:.1f} is below {FLASH_ATTN_MIN_COMPUTE_CAP:.1f}. Flash Attention is disabled."
        )

    recommendations = _build_runtime_recommendations(
        hardware=hardware,
        runtime_profile=runtime_profile,
        binary_compatibility=binary_compatibility,
        main_model=main_model,
        dllm_model=dllm_model,
        fit_params_exe=fit_params_exe,
        selected_server_exe=selected_server_exe,
    )

    return RuntimeAssetResolution(
        project_root=project_root,
        env_file=env_file,
        requested_profile=requested_profile,
        requested_dllm_mode=requested_dllm_mode,
        hardware=hardware,
        primary_server_exe=primary_server_exe,
        legacy_server_exe=legacy_server_exe,
        selected_server_exe=selected_server_exe,
        fit_params_exe=fit_params_exe,
        binary_compatibility=binary_compatibility,
        runtime_profile=runtime_profile,
        dllm_mode=dllm_mode,
        server_topology=server_topology,
        main_model=main_model,
        dllm_model=dllm_model,
        effective_ctx=_ctx_cap_for_profile(env_file, runtime_profile),
        allow_4b_escalation=allow_4b_escalation,
        degrade_reason=degrade_reason,
        flash_attn_mode=flash_attn_mode,
        llama_server_url=main_url,
        dllm_server_url=dllm_url,
        decision_source=decision_source,
        cached_decision=cached_decision or {},
        warnings=warnings,
        fatal_issues=fatal_issues,
        recommendations=recommendations,
    )
