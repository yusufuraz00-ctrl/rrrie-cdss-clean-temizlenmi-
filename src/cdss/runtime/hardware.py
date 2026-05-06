"""Native hardware and runtime policy helpers for CDSS bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path


STANDARD_6GB_TOTAL_THRESHOLD = 5.5
FLASH_ATTN_MIN_COMPUTE_CAP = 8.0
VALID_SERVER_TOPOLOGIES = {"dual_server", "shared_main_server"}


@dataclass(frozen=True)
class GPUHardwareInfo:
    name: str = ""
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    compute_cap: float | None = None
    driver_version: str = ""

    @property
    def detected_vram_gb(self) -> float | None:
        return self.free_vram_gb if self.free_vram_gb is not None else self.total_vram_gb


def _coerce_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _run_nvidia_smi(query_fields: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(query_fields)}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    return result.stdout


def detect_gpu_hardware(override_gb: float = 0.0) -> GPUHardwareInfo:
    if override_gb and override_gb > 0:
        rounded = round(float(override_gb), 2)
        return GPUHardwareInfo(total_vram_gb=rounded, free_vram_gb=rounded)

    stdout = ""
    fields = ["name", "memory.total", "memory.free", "compute_cap", "driver_version"]
    try:
        stdout = _run_nvidia_smi(fields)
    except Exception:
        try:
            stdout = _run_nvidia_smi(["name", "memory.total", "memory.free", "driver_version"])
        except Exception:
            return GPUHardwareInfo()

    first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
    if not first_line:
        return GPUHardwareInfo()

    parts = [part.strip() for part in first_line.split(",")]
    name = parts[0] if parts else ""
    total_gb = round(float(parts[1]) / 1024.0, 2) if len(parts) > 1 and _coerce_float(parts[1]) is not None else None
    free_gb = round(float(parts[2]) / 1024.0, 2) if len(parts) > 2 and _coerce_float(parts[2]) is not None else None

    compute_cap = None
    driver_version = ""
    if len(parts) >= 5:
        compute_cap = _coerce_float(parts[3])
        driver_version = parts[4]
    elif len(parts) >= 4:
        driver_version = parts[3]

    return GPUHardwareInfo(
        name=name,
        total_vram_gb=total_gb,
        free_vram_gb=free_gb,
        compute_cap=compute_cap,
        driver_version=driver_version,
    )


CUDA_ARCH_INCOMPATIBILITY_PATTERNS = (
    "not compiled with any cuda arch",
    "no kernel image is available for execution on the device",
    "cuda error 209",
    "cuda error: no kernel image is available",
)


def probe_llama_server_binary_compatibility(server_exe: str | Path | None) -> str:
    if not server_exe:
        return "missing"

    server_path = Path(server_exe)
    if not server_path.exists():
        return "missing"

    probe_text = ""
    try:
        probe = subprocess.run(
            [str(server_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(server_path.parent),
        )
        probe_text = f"{probe.stdout}\n{probe.stderr}".lower()
        if any(pattern in probe_text for pattern in CUDA_ARCH_INCOMPATIBILITY_PATTERNS):
            return "legacy_required"
        if probe.returncode == 0:
            return "compatible"
        if "cuda" in probe_text or "gpu" in probe_text:
            return "incompatible"
    except Exception:
        return "unknown"

    return "unknown"


def resolve_flash_attn_mode(compute_cap: float | None) -> str:
    if compute_cap is None:
        return "off"
    if compute_cap >= FLASH_ATTN_MIN_COMPUTE_CAP:
        return "auto"
    return "off"


def resolve_runtime_profile(
    requested_profile: str | None,
    *,
    detected_vram_gb: float | None,
    free_vram_gb: float | None = None,
    binary_compatibility: str = "unknown",
) -> str:
    if binary_compatibility in {"legacy_required_missing", "incompatible", "cpu_only"}:
        return "cpu_only"

    normalized = str(requested_profile or "auto").strip().lower()
    if normalized in {"standard_6gb", "compact_4gb", "legacy_4gb", "cpu_only"}:
        return normalized

    observed_vram = detected_vram_gb if detected_vram_gb is not None else free_vram_gb
    if observed_vram is None:
        return "compact_4gb"
    if observed_vram >= STANDARD_6GB_TOTAL_THRESHOLD:
        return "standard_6gb"
    if observed_vram >= 3.5:
        return "compact_4gb"
    return "legacy_4gb"


def resolve_dllm_mode(requested_mode: str | None, runtime_profile: str, *, binary_compatibility: str = "unknown") -> str:
    normalized = str(requested_mode or "auto").strip().lower()
    if normalized in {"full", "lite", "deterministic", "off"}:
        return normalized
    if runtime_profile in {"standard_6gb", "compact_4gb", "legacy_4gb"} and binary_compatibility not in {"legacy_required_missing", "incompatible", "cpu_only"}:
        return "full"
    return "deterministic"


def resolve_server_topology(
    requested_topology: str | None,
    runtime_profile: str,
    dllm_mode: str,
    *,
    dllm_model_available: bool = True,
    free_vram_gb: float | None = None,
) -> str:
    normalized = str(requested_topology or "").strip().lower()
    if normalized in VALID_SERVER_TOPOLOGIES:
        return normalized
    if runtime_profile == "standard_6gb" and dllm_mode in {"full", "lite"} and dllm_model_available and (free_vram_gb is None or free_vram_gb >= 4.25):
        return "dual_server"
    return "shared_main_server"
