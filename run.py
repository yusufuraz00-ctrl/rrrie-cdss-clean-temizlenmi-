"""RRRIE-CDSS unified launcher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
from src.utils.llama_manager import is_server_running, is_server_healthy, start_llama_server, stop_llama_server, close_process_log, _tail_log, ServerLaunchSpec
import time
from dataclasses import dataclass
from pathlib import Path

from src.cdss.runtime.resolver import RuntimeAssetResolution, resolve_runtime_assets
from src.cdss.runtime.hardware import detect_gpu_hardware, resolve_flash_attn_mode
from src.cdss.runtime.decision_cache import record_runtime_decision

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_LOG_DIR = PROJECT_ROOT / "output" / "logs"

LLAMA_HOST = "127.0.0.1"
MAIN_LLAMA_PORT = 8080
DLLM_LLAMA_PORT = 8081
GUI_PORT = 7860

G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[1m"
RST = "\033[0m"
_SHORT_PATH_CACHE: dict[str, Path] = {}
_SHORT_PATH_LOGGED: set[str] = set()


class LauncherError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "launcher_error").strip() or "launcher_error"


def _raise_launcher_error(code: str, message: str) -> None:
    raise LauncherError(code, message)


def _launcher_error_code(exc: Exception) -> str:
    if isinstance(exc, LauncherError):
        return exc.code
    if isinstance(exc, FileNotFoundError):
        return "launcher_file_not_found"
    if isinstance(exc, TimeoutError):
        return "launcher_timeout"
    if isinstance(exc, ValueError):
        return "launcher_invalid_input"
    return "launcher_unhandled_exception"


def _print(icon: str, msg: str, color: str = "") -> None:
    print(f"  {color}{icon}{RST} {msg}")


def _contains_non_ascii(value: str) -> bool:
    return any(ord(ch) > 127 for ch in value)


def _windows_short_path(path: Path) -> Path:
    """Return 8.3 short path on Windows when non-ASCII chars may break native binaries."""
    if sys.platform != "win32":
        return path

    path_str = str(path)
    cached = _SHORT_PATH_CACHE.get(path_str)
    if cached is not None:
        return cached

    resolved = path
    if _contains_non_ascii(path_str):
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(32768)
            written = ctypes.windll.kernel32.GetShortPathNameW(path_str, buffer, len(buffer))
            if written > 0 and buffer.value:
                resolved = Path(buffer.value)
        except Exception:
            resolved = path

    _SHORT_PATH_CACHE[path_str] = resolved
    return resolved


def _to_subprocess_path(path: Path, *, label: str) -> str:
    resolved = _windows_short_path(path)
    if resolved != path and label not in _SHORT_PATH_LOGGED:
        _print("i", f"{label} path normalized for Windows: {resolved}", C)
        _SHORT_PATH_LOGGED.add(label)
    return str(resolved)


def _read_env_value(key: str, default: str = "") -> str:
    if key in os.environ:
        return str(os.environ.get(key, "")).strip()
    if not ENV_FILE.exists():
        return default
    try:
        for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_key, env_value = stripped.split("=", 1)
            if env_key.strip() == key:
                return env_value.strip().strip('"').strip("'")
    except Exception:
        return default
    return default


def _read_env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_read_env_value(key, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _detect_launcher_hardware():
    return detect_gpu_hardware(_read_env_float("RUNTIME_PROFILE_VRAM_GB_OVERRIDE", 0.0))


def resolve_launcher_runtime_state() -> RuntimeAssetResolution:
    return resolve_runtime_assets(PROJECT_ROOT, env_file=ENV_FILE)


def _set_runtime_environment(state: RuntimeAssetResolution) -> None:
    os.environ["RUNTIME_PROFILE_DEFAULT"] = state.name
    os.environ["RUNTIME_SELECTED_PROFILE"] = state.name
    os.environ["RUNTIME_BINARY_COMPATIBILITY"] = state.binary_compatibility
    os.environ["RUNTIME_PROFILE_DLLM_MODE"] = state.dllm_mode
    os.environ["RUNTIME_SERVER_TOPOLOGY"] = state.server_topology
    os.environ["RUNTIME_DEGRADE_REASON"] = state.degrade_reason
    os.environ["RUNTIME_DECISION_SOURCE"] = state.decision_source
    os.environ["RUNTIME_ALLOW_4B_ESCALATION"] = "true" if state.allow_4b_escalation else "false"
    os.environ["RUNTIME_DLLM_MODEL_AVAILABLE"] = "true" if state.dllm_model else "false"
    os.environ["RUNTIME_EFFECTIVE_CTX"] = str(state.effective_ctx)
    if state.cached_decision:
        os.environ["RUNTIME_CACHE_STABLE_RUNS"] = str(int(state.cached_decision.get("stable_runs", 0) or 0))
    else:
        os.environ["RUNTIME_CACHE_STABLE_RUNS"] = "0"
    if state.total_vram_gb is not None:
        os.environ["RUNTIME_TOTAL_VRAM_GB"] = f"{state.total_vram_gb:.2f}"
    if state.free_vram_gb is not None:
        os.environ["RUNTIME_FREE_VRAM_GB"] = f"{state.free_vram_gb:.2f}"
    if state.compute_cap is not None:
        os.environ["RUNTIME_COMPUTE_CAP"] = f"{state.compute_cap:.1f}"
    if state.driver_version:
        os.environ["RUNTIME_DRIVER_VERSION"] = state.driver_version
    if state.selected_server_exe:
        os.environ["LLAMA_SERVER_EXE"] = str(state.selected_server_exe)
    if state.legacy_server_exe:
        os.environ["LLAMA_SERVER_EXE_LEGACY"] = str(state.legacy_server_exe)
    os.environ["LLAMA_SERVER_URL"] = state.llama_server_url
    os.environ["DLLM_SERVER_URL"] = state.dllm_server_url
    os.environ["RUNTIME_FLASH_ATTN_MODE"] = state.flash_attn_mode


def _update_runtime_dllm_state(
    state: RuntimeAssetResolution,
    *,
    dllm_mode: str | None = None,
    server_topology: str | None = None,
    degrade_reason: str | None = None,
    allow_4b_escalation: bool | None = None,
) -> None:
    if dllm_mode is not None:
        state.dllm_mode = dllm_mode
    if server_topology is not None:
        state.server_topology = server_topology
        state.dllm_server_url = state.llama_server_url if server_topology == "shared_main_server" else f"http://{LLAMA_HOST}:{DLLM_LLAMA_PORT}"
    if degrade_reason is not None:
        state.degrade_reason = degrade_reason
    if allow_4b_escalation is not None:
        state.allow_4b_escalation = allow_4b_escalation
    _set_runtime_environment(state)

def find_gguf_model() -> tuple[Path | None, Path | None]:
    state = resolve_launcher_runtime_state()
    return state.main_model, state.dllm_model


def is_server_running(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def is_server_healthy(host: str, port: int) -> bool:
    try:
        import requests

        response = requests.get(f"http://{host}:{port}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def _tail_log(log_path: Path, line_count: int = 12) -> list[str]:
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def close_process_log(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    log_handle = getattr(proc, "log_handle", None)
    if log_handle:
        log_handle.close()


def print_doctor_report(*, json_mode: bool = False) -> int:
    state = resolve_launcher_runtime_state()
    _set_runtime_environment(state)

    if json_mode:
        print(state.to_doctor_json())
        return 0 if not state.fatal_issues else 1

    print(f"\n{B}RRRIE-CDSS Doctor{RST}")
    print("-" * 48)
    _print(
        "i",
        (
            f"profile={state.name} topology={state.server_topology} "
            f"dllm={state.dllm_mode} ctx={state.effective_ctx} source={state.decision_source}"
        ),
        C,
    )
    _print(
        "i",
        (
            f"gpu={state.hardware.name or 'unknown'} total_vram={state.total_vram_gb or 'unknown'}GB "
            f"free_vram={state.free_vram_gb or 'unknown'}GB cc={state.compute_cap or 'unknown'} "
            f"flash_attn={state.flash_attn_mode or 'off'}"
        ),
        C,
    )
    _print("i", f"llama_url={state.llama_server_url}", C)
    _print("i", f"dllm_url={state.dllm_server_url}", C)
    if state.selected_server_exe:
        _print("ok", f"selected llama-server: {state.selected_server_exe}", G)
    else:
        _print("x", "selected llama-server: none", R)
    if state.legacy_server_exe:
        _print("i", f"legacy binary: {state.legacy_server_exe}", C)
    if state.main_model:
        _print("ok", f"main model: {state.main_model}", G)
    else:
        _print("x", "main model missing", R)
    if state.dllm_model:
        _print("ok", f"dllm model: {state.dllm_model}", G)
    else:
        _print("!", "dllm model missing", Y)
    _print("i", f"binary_compatibility={state.binary_compatibility}", C)
    if state.degrade_reason:
        _print("!", f"degrade_reason={state.degrade_reason}", Y)
    for warning in state.warnings:
        _print("!", warning, Y)
    for issue in state.fatal_issues:
        _print("x", issue, R)
    for recommendation in state.recommendations:
        _print(">", recommendation, C)
    print()
    return 0 if not state.fatal_issues else 1


def _build_launch_spec(
    profile: str,
    model_path: Path,
    port: int,
    *,
    is_dllm: bool = False,
) -> ServerLaunchSpec:
    """Build launch spec with GPU-adaptive flash attention mode."""
    label = "dllm" if is_dllm else "main"
    scale = _read_env_float("RUNTIME_REASONING_BUDGET_SCALE", 1.0)
    if scale <= 0:
        scale = 1.0

    def _scaled_reasoning_budget(base: int) -> int:
        return max(64, min(1536, int(base * scale)))

    # Resolve flash_attn from environment (set by _set_runtime_environment)
    flash_attn = os.environ.get("RUNTIME_FLASH_ATTN_MODE", "off")
    compute_cap_raw = os.environ.get("RUNTIME_COMPUTE_CAP", "")
    if not flash_attn or flash_attn not in {"auto", "off"}:
        try:
            cc = float(compute_cap_raw) if compute_cap_raw else None
        except (TypeError, ValueError):
            cc = None
        flash_attn = resolve_flash_attn_mode(cc)

    if profile == "standard_6gb" and not is_dllm:
        return ServerLaunchSpec(label, model_path, port, 4096, 256, 128, 768, flash_attn, None, None, reasoning="on", reasoning_budget=_scaled_reasoning_budget(512))
    if profile == "standard_6gb" and is_dllm:
        return ServerLaunchSpec(label, model_path, port, 2048, 128, 64, 512, flash_attn, "q4_0", "q4_0", reasoning="on", reasoning_budget=_scaled_reasoning_budget(384))
    if profile == "compact_4gb" and not is_dllm:
        return ServerLaunchSpec(label, model_path, port, 2048, 128, 64, 512, flash_attn, "q4_0", "q4_0", reasoning="on", reasoning_budget=_scaled_reasoning_budget(384))
    if profile == "compact_4gb" and is_dllm:
        return ServerLaunchSpec(label, model_path, port, 1536, 64, 32, 384, flash_attn, "q4_0", "q4_0", reasoning="on", reasoning_budget=_scaled_reasoning_budget(256))
    return ServerLaunchSpec(label, model_path, port, 1536, 64, 32, 384, "off", "q4_0", "q4_0", reasoning="on", reasoning_budget=_scaled_reasoning_budget(256))


def _parse_fit_output(stdout: str, spec: ServerLaunchSpec) -> tuple[int, str | None]:
    arg_line = next((line.strip() for line in stdout.splitlines() if line.strip().startswith("-")), "")
    if not arg_line:
        return spec.ctx_size, None

    ctx_match = re.search(r"(?:^|\s)-c\s+(-?\d+)", arg_line)
    ngl_match = re.search(r"(?:^|\s)-ngl\s+([^\s]+)", arg_line)
    fitted_ctx = spec.ctx_size
    if ctx_match:
        try:
            parsed_ctx = int(ctx_match.group(1))
            if parsed_ctx > 0:
                fitted_ctx = parsed_ctx
        except ValueError:
            pass
    fitted_ngl = ngl_match.group(1) if ngl_match else None
    return fitted_ctx, fitted_ngl


def _run_fit_params(spec: ServerLaunchSpec, server_exe: Path, fit_exe: Path | None) -> tuple[int, str | None]:
    if not fit_exe:
        raise RuntimeError("llama-fit-params missing")

    cmd = [
        _to_subprocess_path(fit_exe, label="llama-fit-params"),
        "-m",
        _to_subprocess_path(spec.model_path, label=f"{spec.label} model"),
        "-c",
        str(spec.ctx_size),
        "-fitc",
        str(spec.ctx_size),
        "-fitt",
        str(spec.fit_target_mib),
        "-b",
        str(spec.batch_size),
        "-ub",
        str(spec.ubatch_size),
        "-np",
        str(spec.parallel),
        "-fa",
        spec.flash_attn,
    ]
    if spec.cache_type_k:
        cmd.extend(["-ctk", spec.cache_type_k])
    if spec.cache_type_v:
        cmd.extend(["-ctv", spec.cache_type_v])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=20,
        cwd=_to_subprocess_path(server_exe.parent, label="llama-server working directory"),
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(stderr or "llama-fit-params failed")
    return _parse_fit_output(result.stdout, spec)


def _build_server_command(
    server_exe: Path,
    spec: ServerLaunchSpec,
    *,
    fitted_ctx: int | None = None,
    fitted_ngl: str | None = None,
) -> list[str]:
    cmd = [
        _to_subprocess_path(server_exe, label="llama-server"),
        "-m",
        _to_subprocess_path(spec.model_path, label=f"{spec.label} model"),
        "--host",
        LLAMA_HOST,
        "--port",
        str(spec.port),
        "-c",
        str(fitted_ctx or spec.ctx_size),
        "-b",
        str(spec.batch_size),
        "-ub",
        str(spec.ubatch_size),
        "-np",
        str(spec.parallel),
        "--flash-attn",
        spec.flash_attn,
        "--no-cache-prompt" if not spec.cache_prompt else "--cache-prompt",
    ]
    if spec.cache_type_k:
        cmd.extend(["-ctk", spec.cache_type_k])
    if spec.cache_type_v:
        cmd.extend(["-ctv", spec.cache_type_v])
    if fitted_ngl:
        cmd.extend(["-ngl", fitted_ngl])
    if spec.reasoning:
        cmd.extend(["--reasoning", spec.reasoning])
    if spec.reasoning_budget is not None:
        cmd.extend(["--reasoning-budget", str(spec.reasoning_budget)])
    return cmd


def start_llama_server(server_exe: Path, spec: ServerLaunchSpec, *, fit_exe: Path | None = None) -> subprocess.Popen:
    if not server_exe.exists():
        _print("x", f"llama-server not found: {server_exe}", R)
        _raise_launcher_error("launcher_server_binary_missing", f"llama-server not found: {server_exe}")

    fitted_ctx = spec.ctx_size
    fitted_ngl: str | None = None
    try:
        fitted_ctx, fitted_ngl = _run_fit_params(spec, server_exe, fit_exe)
        fit_summary = f"ctx={fitted_ctx}" + (f" ngl={fitted_ngl}" if fitted_ngl is not None else "")
        _print("i", f"{spec.label} fit params resolved: {fit_summary}", C)
    except Exception as exc:
        _print("!", f"{spec.label} fit params failed, using static fallback: {exc}", Y)

    cmd = _build_server_command(
        server_exe,
        spec,
        fitted_ctx=fitted_ctx,
        fitted_ngl=fitted_ngl,
    )

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_LOG_DIR / f"llama-server-{spec.label}.log"
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")

    _print("...", f"Starting {spec.label} llama-server on port {spec.port}", Y)
    _print(" ", f"  model: {spec.model_path.name}", C)
    _print(
        " ",
        (
            f"  ctx={fitted_ctx} b={spec.batch_size} ub={spec.ubatch_size} "
            f"np={spec.parallel} fa={spec.flash_attn} "
            f"cache={'on' if spec.cache_prompt else 'off'}"
        ),
        C,
    )

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=_to_subprocess_path(server_exe.parent, label="llama-server working directory"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    proc.log_handle = log_handle  # type: ignore[attr-defined]
    proc.log_path = log_path  # type: ignore[attr-defined]
    proc.fitted_ctx = fitted_ctx  # type: ignore[attr-defined]
    proc.fitted_ngl = fitted_ngl  # type: ignore[attr-defined]

    for second in range(120):
        if proc.poll() is not None:
            _print("x", f"{spec.label} llama-server exited early with code {proc.returncode}", R)
            for line in _tail_log(log_path):
                _print(" ", line)
            close_process_log(proc)
            _raise_launcher_error(
                "launcher_server_start_failed",
                f"{spec.label} llama-server exited early with code {proc.returncode}",
            )

        if is_server_healthy(LLAMA_HOST, spec.port):
            _print("ok", f"{spec.label} llama-server ready after {second + 1}s", G)
            _print(" ", f"  log: {log_path}", C)
            return proc

        time.sleep(1)

    _print("x", f"{spec.label} llama-server did not become healthy within 120s", R)
    proc.terminate()
    close_process_log(proc)
    _raise_launcher_error(
        "launcher_server_health_timeout",
        f"{spec.label} llama-server did not become healthy within 120s",
    )


def stop_llama_server() -> None:
    if sys.platform == "win32":
        os.system("taskkill /f /im llama-server.exe >nul 2>&1")
    else:
        os.system("pkill -f llama-server")
    _print("ok", "llama-server stopped", G)


def _string_payload_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item for item in value if isinstance(item, str)]
        return "\n".join(parts)
    return ""


def _extract_warmup_message(data: dict[str, object]) -> str:
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    if not isinstance(message, dict):
        message = {}

    candidates = [
        _string_payload_value(message.get("content", "")),
        _string_payload_value(message.get("reasoning", "")),
        _string_payload_value(message.get("reasoning_content", "")),
        _string_payload_value(first_choice.get("text", "")) if isinstance(first_choice, dict) else "",
        _string_payload_value(data.get("content", "")),
    ]
    for item in candidates:
        if str(item or "").strip():
            return str(item)
    return ""


def _warmup_server(host: str, port: int, label: str = "main") -> bool:
    """Send a trivial completion to the server and validate non-empty output.

    This catches silent GPU failures where the server appears healthy
    (HTTP responds) but CUDA kernels silently crash and produce no output.
    """
    import json as _json
    _print("...", f"Running {label} server warmup test...", Y)
    try:
        import requests
        resp = requests.post(
            f"http://{host}:{port}/v1/chat/completions",
            json={
                "model": "warmup",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            _print("x", f"{label} warmup failed: HTTP {resp.status_code}", R)
            _print(" ", f"  Response: {resp.text[:300]}")
            return False
        data = resp.json()
        content = _extract_warmup_message(data)
        if not content.strip():
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            completion_tokens = int((usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0) or 0)
            if completion_tokens > 0:
                _print("!", f"{label} warmup returned empty assistant content but generated tokens; accepting warmup", Y)
                return True
            _print("x", f"{label} warmup returned EMPTY content — model may have silent GPU failure", R)
            _print(" ", "  This typically occurs when Flash Attention is incompatible with this GPU.")
            _print(" ", "  Try setting RUNTIME_FLASH_ATTN_MODE=off in .env")
            return False
        _print("ok", f"{label} warmup passed: '{content.strip()[:60]}'", G)
        return True
    except Exception as exc:
        _print("x", f"{label} warmup failed: {exc}", R)
        return False


def _list_listening_pids(port: int) -> list[int]:
    if sys.platform != "win32":
        return []
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    pids: set[int] = set()
    target_suffix = f":{port}"
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) < 5 or columns[0].upper() != "TCP":
            continue
        local_address = columns[1]
        state = columns[3].upper()
        pid_raw = columns[4]
        if state != "LISTENING" or not local_address.endswith(target_suffix):
            continue
        if pid_raw.isdigit():
            pids.add(int(pid_raw))
    return sorted(pids)


def _process_image_name(pid: int) -> str:
    if sys.platform != "win32":
        return ""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""

    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    if not line or line.startswith("INFO:"):
        return ""
    try:
        row = next(csv.reader([line]))
    except Exception:
        return ""
    return row[0].strip() if row else ""


def _stop_llama_server_on_port(port: int) -> None:
    for pid in _list_listening_pids(port):
        image_name = _process_image_name(pid).lower()
        if "llama-server" not in image_name:
            if image_name:
                _print(
                    "!",
                    f"port {port} is occupied by {image_name}; leaving it untouched",
                    Y,
                )
            continue
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                _print("i", f"stopped llama-server PID {pid} on port {port}", C)
            else:
                error_text = (result.stderr or result.stdout or "").strip()
                _print("!", f"failed to stop llama-server PID {pid} on port {port}: {error_text}", Y)
        except Exception as exc:
            _print("!", f"failed to stop llama-server PID {pid} on port {port}: {exc}", Y)


def _refresh_runtime_after_main_start(state: RuntimeAssetResolution) -> None:
    refreshed = _detect_launcher_hardware()
    state.free_vram_gb = refreshed.free_vram_gb
    state.hardware = refreshed
    _set_runtime_environment(state)


def print_status() -> None:
    print(f"\n{B}RRRIE-CDSS Status{RST}")
    print("-" * 48)

    state = resolve_launcher_runtime_state()
    _set_runtime_environment(state)
    total_label = f"{state.total_vram_gb:.2f} GB" if state.total_vram_gb is not None else "unknown"
    free_label = f"{state.free_vram_gb:.2f} GB" if state.free_vram_gb is not None else "unknown"
    cc_label = f"{state.compute_cap:.1f}" if state.compute_cap is not None else "unknown"
    _print(
        "i",
        (
            f"profile={state.name} total_vram={total_label} free_vram={free_label} "
            f"cc={cc_label} ctx={state.effective_ctx} dllm={state.dllm_mode} "
            f"topology={state.server_topology} source={state.decision_source}"
        ),
        C,
    )
    if state.driver_version:
        _print("i", f"driver={state.driver_version}", C)
    if state.selected_server_exe:
        _print("ok", f"llama-server: {state.selected_server_exe}", G)
    else:
        _print("x", "llama-server: missing or incompatible", R)
    if state.legacy_server_exe:
        _print("i", f"legacy binary: {state.legacy_server_exe}", C)
    _print("i", f"binary_compatibility={state.binary_compatibility}", C)
    if state.degrade_reason:
        _print("!", f"degrade_reason={state.degrade_reason}", Y)

    if is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT):
        _print("ok", f"main server healthy at http://{LLAMA_HOST}:{MAIN_LLAMA_PORT}", G)
    elif is_server_running(LLAMA_HOST, MAIN_LLAMA_PORT):
        _print("!", "main server port is open but unhealthy", Y)
    else:
        _print("x", "main server offline", R)

    if state.server_topology == "shared_main_server":
        _print("i", f"dllm routes through main server at {state.dllm_server_url}", C)
        if is_server_running(LLAMA_HOST, DLLM_LLAMA_PORT):
            _print("!", f"port {DLLM_LLAMA_PORT} is still in use even though shared_main_server is active", Y)
    elif is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT):
        _print("ok", f"dllm server healthy at http://{LLAMA_HOST}:{DLLM_LLAMA_PORT}", G)
    elif is_server_running(LLAMA_HOST, DLLM_LLAMA_PORT):
        _print("!", "dllm server port is open but unhealthy", Y)
    else:
        _print("x", "dllm server offline", R)

    if is_server_running(LLAMA_HOST, GUI_PORT):
        _print("ok", f"gui listening at http://{LLAMA_HOST}:{GUI_PORT}", G)
    else:
        _print("x", f"gui offline on port {GUI_PORT}", R)

    main_model, dllm_model = find_gguf_model()
    if main_model:
        _print("ok", f"main model: {main_model.name}", G)
    else:
        _print("x", "main model missing", R)
    if dllm_model:
        _print("ok", f"dllm model: {dllm_model.name}", G)
    else:
        _print("!", "dllm model missing", Y)

    if ENV_FILE.exists():
        _print("ok", ".env present", G)
    else:
        _print("!", ".env missing, defaults will be used", Y)
    for warning in state.warnings:
        _print("!", warning, Y)

    print()


def launch_gui() -> None:
    _print("ok", f"Launching GUI on http://localhost:{GUI_PORT}", C)
    print()
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from gui.server import main as gui_main

    gui_main()


def run_tests(extra_args: list[str]) -> int:
    os.chdir(str(PROJECT_ROOT))
    test_script = PROJECT_ROOT / "tests" / "test_e2e_medical.py"
    if test_script.exists():
        cmd = [sys.executable, str(test_script)] + extra_args
        _print(">", "Running legacy pipeline tests", C)
        if extra_args:
            _print(" ", f"  args: {' '.join(extra_args)}")
        print()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return result.returncode

    run_all = "--all" in extra_args
    case_filter = ""
    if "--case" in extra_args:
        try:
            case_filter = str(extra_args[extra_args.index("--case") + 1] or "").strip()
        except (IndexError, ValueError):
            case_filter = ""

    preset = ""
    passthrough_args: list[str] = []
    iterator = iter(extra_args)
    for token in iterator:
        if token == "--test-preset":
            preset = str(next(iterator, "") or "").strip().lower()
            continue
        passthrough_args.append(token)

    cmd = [sys.executable, "-m", "pytest"]
    if run_all:
        cmd.append("tests")
    else:
        preset_targets = {
            "smoke": ["tests/test_cdss_smoke.py"],
            "safety": ["tests/test_cdss_mode_and_evidence.py", "-k", "safety or diagnosis_first_policy or intervention"],
            "retrieval": ["tests/test_cdss_mode_and_evidence.py", "-k", "retrieval or query_quality or governor"],
            "multilingual": ["tests/test_cdss_regression.py", "-k", "multilingual or colloquial or thyrotoxic"],
        }
        cmd.extend(preset_targets.get(preset, ["tests/test_cdss_smoke.py"]))
    if case_filter:
        cmd.extend(["-k", case_filter])
    if passthrough_args:
        cmd.extend(passthrough_args)

    _print("!", "tests/test_e2e_medical.py not found; using pytest launcher fallback", Y)
    _print(">", "Running pipeline tests", C)
    if extra_args:
        _print(" ", f"  args: {' '.join(extra_args)}")
    if preset:
        _print(" ", f"  preset: {preset}")
    print()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_preflight_checks(*, json_mode: bool = False) -> int:
    state = resolve_launcher_runtime_state()
    _set_runtime_environment(state)
    if json_mode:
        payload = {
            "doctor": json.loads(state.to_doctor_json()),
            "preflight": {
                "env_file_exists": ENV_FILE.exists(),
                "main_model_present": bool(state.main_model),
                "dllm_model_present": bool(state.dllm_model),
                "fatal_issues": list(state.fatal_issues),
                "warnings": list(state.warnings),
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if not state.fatal_issues else 1

    _print(">", "Running startup preflight diagnostics", C)
    rc = print_doctor_report(json_mode=False)
    if rc != 0:
        _raise_launcher_error("launcher_preflight_failed", "runtime doctor reported fatal issues during preflight")
    if not ENV_FILE.exists():
        _raise_launcher_error("launcher_env_missing", ".env file is missing; preflight failed for strict startup mode")
    if not state.main_model:
        _raise_launcher_error("launcher_main_model_missing", "main model is missing; preflight failed")
    _print("ok", "preflight diagnostics passed", G)
    return 0


def _suite_case_ids(suite: str) -> list[str]:
    normalized = str(suite or "gold10").strip().lower()
    suite_file = PROJECT_ROOT / "tests" / f"benchmark_suite_{normalized}.json"
    if normalized == "fresh10":
        suite_file = PROJECT_ROOT / "tests" / "benchmark_suite_fresh50.json"
    if not suite_file.exists():
        return []
    try:
        payload = json.loads(suite_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    case_ids = [str(item.get("case_id", "")).strip() for item in list(payload.get("cases", []) or []) if str(item.get("case_id", "")).strip()]
    if normalized == "fresh10":
        return case_ids[:10]
    return case_ids


def run_benchmark(*, suite: str, case_id: str | None = None, offline: bool = False) -> int:
    os.chdir(str(PROJECT_ROOT))
    benchmark_script = PROJECT_ROOT / "tests" / "benchmark_cdss_cases.py"
    cmd = [sys.executable, str(benchmark_script), "--suite", suite]
    if case_id:
        cmd.extend(["--case", case_id])
    if offline:
        cmd.append("--offline")
    _print(">", f"Running benchmark suite `{suite}`", C)
    if case_id:
        _print(" ", f"  case: {case_id}")
    if offline:
        _print(" ", "  mode: offline")
    print()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def _run_project_script(script_relative_path: str, *, label: str) -> int:
    script_path = PROJECT_ROOT / script_relative_path
    if not script_path.exists():
        _print("x", f"{label} script missing: {script_path}", R)
        return 1
    _print(">", f"Running {label}", C)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
    return int(result.returncode)


def run_hardening_checks() -> int:
    checks = [
        ("scripts/secret_scan.py", "secret scan"),
        ("scripts/check_complexity.py", "complexity budget check"),
        ("scripts/generate_file_audit_matrix.py", "file audit matrix generation"),
    ]
    for script_path, label in checks:
        rc = _run_project_script(script_path, label=label)
        if rc != 0:
            _print("!", f"{label} failed with exit code {rc}", Y)
            return rc
    return 0


def ensure_servers() -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
    main_proc = None
    dllm_proc = None

    _print("i", f"platform={sys.platform} python={sys.version.split()[0]}")
    state = resolve_launcher_runtime_state()
    _set_runtime_environment(state)

    if state.fatal_issues:
        _print("x", "runtime doctor found fatal issues", R)
        for issue in state.fatal_issues:
            _print(" ", issue)
        _raise_launcher_error("launcher_runtime_doctor_failed", "runtime doctor found fatal issues")

    free_vram_label = f"{state.free_vram_gb:.2f}" if state.free_vram_gb is not None else "unknown"
    compute_cap_label = f"{state.compute_cap:.1f}" if state.compute_cap is not None else "unknown"
    _print(
        "i",
        (
            f"runtime profile={state.name} free_vram={free_vram_label} "
            f"compute_cap={compute_cap_label} dllm_mode={state.dllm_mode} "
            f"topology={state.server_topology}"
        ),
        C,
    )
    _print("i", f"binary_compatibility={state.binary_compatibility}", C)
    for warning in state.warnings:
        _print("!", warning, Y)

    if not is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT):
        main_spec = _build_launch_spec(state.name, state.main_model, MAIN_LLAMA_PORT, is_dllm=False)
        main_proc = start_llama_server(state.selected_server_exe, main_spec, fit_exe=state.fit_params_exe)
        fitted_ctx = getattr(main_proc, "fitted_ctx", None)
        if isinstance(fitted_ctx, int) and fitted_ctx > 0:
            state.effective_ctx = fitted_ctx
            _set_runtime_environment(state)
        _refresh_runtime_after_main_start(state)
    else:
        _print("ok", "main llama-server already healthy", G)
        _refresh_runtime_after_main_start(state)

    warmup_ok = _warmup_server(LLAMA_HOST, MAIN_LLAMA_PORT, "main")
    if not warmup_ok and str(os.environ.get("RUNTIME_FLASH_ATTN_MODE", state.flash_attn_mode or "") or "").strip().lower() != "off":
        _print("!", "main warmup failed; retrying once with Flash Attention disabled", Y)
        os.environ["RUNTIME_FLASH_ATTN_MODE"] = "off"
        if hasattr(state, "flash_attn_mode"):
            state.flash_attn_mode = "off"
        _set_runtime_environment(state)
        _stop_llama_server_on_port(MAIN_LLAMA_PORT)
        if main_proc:
            try:
                main_proc.terminate()
            except Exception:
                pass
            close_process_log(main_proc)
        main_spec = _build_launch_spec(state.name, state.main_model, MAIN_LLAMA_PORT, is_dllm=False)
        main_proc = start_llama_server(state.selected_server_exe, main_spec, fit_exe=state.fit_params_exe)
        fitted_ctx = getattr(main_proc, "fitted_ctx", None)
        if isinstance(fitted_ctx, int) and fitted_ctx > 0:
            state.effective_ctx = fitted_ctx
        _refresh_runtime_after_main_start(state)
        warmup_ok = _warmup_server(LLAMA_HOST, MAIN_LLAMA_PORT, "main")
        if warmup_ok:
            _update_runtime_dllm_state(
                state,
                degrade_reason=state.degrade_reason or "flash_attn_auto_disabled_after_warmup_failure",
            )

    if not warmup_ok:
        _print("x", "Main server warmup failed. The model is not generating output.", R)
        _print(" ", "  Possible causes:")
        _print(" ", "    - GPU compute capability incompatible with Flash Attention")
        _print(" ", "    - CUDA driver mismatch")
        _print(" ", "    - Insufficient VRAM for model")
        if main_proc:
            main_proc.terminate()
            close_process_log(main_proc)
        _raise_launcher_error(
            "launcher_warmup_failed",
            "Main server warmup failed. The model is not generating output.",
        )

    remaining_free = state.free_vram_gb
    min_free_vram = _read_env_float("RUNTIME_PROFILE_MIN_FREE_VRAM_GB", 0.9)
    if (
        state.server_topology == "dual_server"
        and remaining_free is not None
        and remaining_free < min_free_vram
    ):
        _print(
            "!",
            f"remaining free VRAM {remaining_free:.2f} GB is below {min_free_vram:.2f} GB; switching to shared_main_server",
            Y,
        )
        _update_runtime_dllm_state(
            state,
            server_topology="shared_main_server",
            degrade_reason="shared_main_server_low_free_vram",
            allow_4b_escalation=False,
        )

    if state.server_topology == "dual_server" and state.dllm_mode in {"full", "lite"}:
        if not is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT):
            if state.dllm_model:
                dllm_spec = _build_launch_spec(state.name, state.dllm_model, DLLM_LLAMA_PORT, is_dllm=True)
                dllm_proc = start_llama_server(state.selected_server_exe, dllm_spec, fit_exe=state.fit_params_exe)
            else:
                _print("!", "dllm model missing; switching to shared_main_server topology", Y)
                _update_runtime_dllm_state(
                    state,
                    server_topology="shared_main_server",
                    degrade_reason="shared_main_server_missing_dllm_model",
                    allow_4b_escalation=False,
                )
        else:
            _print("ok", "dllm llama-server already healthy", G)
    elif state.server_topology == "shared_main_server":
        _stop_llama_server_on_port(DLLM_LLAMA_PORT)
        _print(
            "i",
            "using shared_main_server topology; DLLM, entropy probes, and fallback reasoning will target the main 4B server",
            C,
        )
    else:
        _print("i", f"skipping DLLM server (mode={state.dllm_mode})", C)

    refreshed = _detect_launcher_hardware()
    state.free_vram_gb = refreshed.free_vram_gb
    allow_4b = (
        state.server_topology == "dual_server"
        and state.name == "standard_6gb"
        and state.dllm_mode == "full"
        and is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT)
        and is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT)
        and refreshed.free_vram_gb is not None
        and refreshed.free_vram_gb >= 1.0
    )
    degrade_reason = state.degrade_reason
    if state.server_topology == "dual_server" and state.name == "standard_6gb" and state.dllm_mode == "full" and not allow_4b:
        degrade_reason = degrade_reason or "4b_escalation_disabled_low_free_vram"
    _update_runtime_dllm_state(
        state,
        degrade_reason=degrade_reason,
        allow_4b_escalation=allow_4b,
    )
    stable_runtime = is_server_healthy(LLAMA_HOST, MAIN_LLAMA_PORT) and (
        state.server_topology == "shared_main_server"
        or is_server_healthy(LLAMA_HOST, DLLM_LLAMA_PORT)
    )
    state.cached_decision = record_runtime_decision(
        PROJECT_ROOT,
        hardware=state.hardware,
        main_model=state.main_model,
        dllm_model=state.dllm_model,
        server_exe=state.selected_server_exe,
        runtime_profile=state.name,
        server_topology=state.server_topology,
        dllm_mode=state.dllm_mode,
        binary_compatibility=state.binary_compatibility,
        post_load_free_vram_gb=state.free_vram_gb,
        stable=stable_runtime,
        degrade_reason=state.degrade_reason,
    )
    _set_runtime_environment(state)

    return main_proc, dllm_proc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RRRIE-CDSS unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py run.py
  py run.py --doctor
  py run.py --doctor --json
    py run.py --preflight
  py run.py --gui
  py run.py --status
  py run.py --stop
  py run.py --test --all
  py run.py --benchmark
  py run.py --benchmark --benchmark-suite gold10
    py run.py --secret-scan
    py run.py --complexity-check
    py run.py --audit-matrix
    py run.py --hardening-checks
""",
    )
    parser.add_argument("--doctor", action="store_true", help="Resolve runtime assets and print a readiness report")
    parser.add_argument("--preflight", action="store_true", help="Run strict startup preflight diagnostics")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for --doctor")
    parser.add_argument("--gui", action="store_true", help="Start only the GUI process")
    parser.add_argument("--test", action="store_true", help="Run the pipeline tests")
    parser.add_argument(
        "--test-preset",
        type=str,
        choices=["smoke", "safety", "retrieval", "multilingual"],
        help="Use curated test subsets for targeted validation",
    )
    parser.add_argument("--status", action="store_true", help="Show current system status")
    parser.add_argument("--stop", action="store_true", help="Stop running llama-server processes")
    parser.add_argument("--benchmark", action="store_true", help="Run the native CDSS benchmark harness")
    parser.add_argument("--benchmark-suite", type=str, default="gold10", help="Benchmark suite: gold10, fresh10, fresh50, legacy, stress26")
    parser.add_argument("--benchmark-case", type=str, help="Run a single benchmark case id")
    parser.add_argument("--offline-benchmark", action="store_true", help="Disable external evidence during benchmark runs")
    parser.add_argument("--secret-scan", action="store_true", help="Run secret scanning against repository text files")
    parser.add_argument("--complexity-check", action="store_true", help="Run module-level complexity budget checks")
    parser.add_argument("--audit-matrix", action="store_true", help="Generate file audit matrix artifacts")
    parser.add_argument("--hardening-checks", action="store_true", help="Run secret scan, complexity check, and audit matrix generation")
    parser.add_argument("--all", action="store_true", help="Run all real cases")
    parser.add_argument("--case", type=str, help="Run a single named case")
    parser.add_argument("--narrative", action="store_true", help="Disabled compatibility flag")
    parser.add_argument("--count", type=int, default=3, help="Disabled compatibility flag")
    parser.add_argument("--hard", action="store_true", help="Disabled compatibility flag")
    parser.add_argument("--mix", action="store_true", help="Disabled compatibility flag")
    parser.add_argument("--who", action="store_true", help="Run WHO cases only")
    parser.add_argument("--list", action="store_true", help="List available real cases")
    parser.add_argument(
        "--atom",
        action="store_true",
        help="Enable ATOM deep-reasoning mode: thinking ON for R1/R3/IE stages (slower, more thorough)",
    )

    args = parser.parse_args()

    # ATOM mode: set env var before any settings are loaded so get_settings() picks it up.
    # Also clear lru_cache in case settings was already instantiated during import.
    if args.atom:
        os.environ["CDSS_ATOM_MODE"] = "1"
        try:
            from config.settings import get_settings as _gs
            _gs.cache_clear()
        except Exception:
            pass

    if (args.test or args.list or args.who) and (args.narrative or args.hard or args.mix):
        _print("x", "synthetic and narrative modes are disabled", Y)
        return 1

    if not (args.doctor and args.json):
        atom_label = f"  {Y}[ATOM MODE — deep reasoning ON]{RST}" if args.atom else ""
        print(
            f"""
{B}+----------------------------------------------------------+
| RRRIE-CDSS Unified Launcher                               |
| Clinical Decision Support System                          |
+----------------------------------------------------------+{RST}{atom_label}
"""
        )

    if args.status:
        print_status()
        return 0

    if args.hardening_checks:
        return run_hardening_checks()

    if args.secret_scan:
        return _run_project_script("scripts/secret_scan.py", label="secret scan")

    if args.complexity_check:
        return _run_project_script("scripts/check_complexity.py", label="complexity budget check")

    if args.audit_matrix:
        return _run_project_script("scripts/generate_file_audit_matrix.py", label="file audit matrix generation")

    if args.doctor:
        return print_doctor_report(json_mode=args.json)

    if args.preflight:
        return run_preflight_checks(json_mode=args.json)

    if args.stop:
        stop_llama_server()
        return 0

    if args.list:
        case_ids: list[str] = []
        legacy_case_module = PROJECT_ROOT / "tests" / "test_e2e_medical.py"
        if legacy_case_module.exists():
            os.chdir(str(PROJECT_ROOT))
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from tests.test_e2e_medical import ALL_CASES

            case_ids = list(ALL_CASES)
        else:
            case_ids = _suite_case_ids(args.benchmark_suite)
        if not case_ids:
            _print("!", "No cases found for --list. Use --benchmark-suite to select a valid benchmark inventory.", Y)
            return 1

        print(f"{B}Cases ({len(case_ids)}):{RST}")
        for item in case_ids:
            print(f"  - {item}")
        print()
        return 0

    main_proc = None
    dllm_proc = None

    try:
        if not args.gui:
            main_proc, dllm_proc = ensure_servers()
        else:
            _set_runtime_environment(resolve_launcher_runtime_state())

        if args.test or args.who or args.benchmark:
            test_args: list[str] = []
            if args.benchmark:
                rc = run_benchmark(
                    suite=args.benchmark_suite,
                    case_id=args.benchmark_case,
                    offline=args.offline_benchmark,
                )
            elif args.who:
                test_args.extend(["--case", "who_"])
                if args.all:
                    test_args.append("--all")
                rc = run_tests(test_args)
            else:
                if args.all:
                    test_args.append("--all")
                if args.case:
                    test_args.extend(["--case", args.case])
                if args.test_preset:
                    test_args.extend(["--test-preset", args.test_preset])
                rc = run_tests(test_args)
            if rc == 0:
                _print("ok", "benchmark/tests completed successfully", G)
            else:
                _print("!", f"benchmark/tests finished with exit code {rc}", Y)
            return rc
        else:
            launch_gui()
            return 0

    except KeyboardInterrupt:
        _print("!", "stopped by user", Y)
        return 130
    except Exception as exc:
        error_code = _launcher_error_code(exc)
        _print("x", f"launcher error [{error_code}]: {exc}", R)
        return 1
    finally:
        if main_proc and main_proc.poll() is None:
            _print("i", "shutting down main llama-server", C)
            main_proc.terminate()
        if dllm_proc and dllm_proc.poll() is None:
            _print("i", "shutting down dllm llama-server", C)
            dllm_proc.terminate()
        close_process_log(main_proc)
        close_process_log(dllm_proc)


if __name__ == "__main__":
    raise SystemExit(main())
