from __future__ import annotations

import subprocess
import sys
import os
import struct
from pathlib import Path
import time
from typing import TYPE_CHECKING
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

import socket
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.cdss.runtime.resolver import RuntimeAssetResolution


# ---------------------------------------------------------------------------
# GGUF header validation (merged from gguf.py)
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc


@_dc(frozen=True)
class GGUFHeaderInfo:
    valid: bool
    reason: str = ""
    version: int | None = None
    tensor_count: int | None = None
    metadata_kv_count: int | None = None


def read_gguf_header(path: str | Path) -> GGUFHeaderInfo:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return GGUFHeaderInfo(valid=False, reason="missing")
    try:
        with file_path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        return GGUFHeaderInfo(valid=False, reason=f"io_error:{exc}")
    if len(header) < 24:
        return GGUFHeaderInfo(valid=False, reason="truncated_header")
    magic = header[:4]
    if magic != b"GGUF":
        return GGUFHeaderInfo(valid=False, reason="invalid_magic")
    try:
        version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", header[4:24])
    except struct.error:
        return GGUFHeaderInfo(valid=False, reason="invalid_header_layout")
    if version <= 0:
        return GGUFHeaderInfo(valid=False, reason="invalid_version")
    return GGUFHeaderInfo(
        valid=True, reason="ok", version=version,
        tensor_count=tensor_count, metadata_kv_count=metadata_kv_count,
    )


def has_valid_gguf_header(path: str | Path) -> bool:
    return read_gguf_header(path).valid
@dataclass
class ServerLaunchSpec:
    label: str
    model_path: Path
    port: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    fit_target_mib: int
    flash_attn: str
    cache_type_k: str | None
    cache_type_v: str | None
    parallel: int = 1
    cache_prompt: bool = False
    reasoning: str | None = None
    reasoning_budget: int | None = None

import json

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

