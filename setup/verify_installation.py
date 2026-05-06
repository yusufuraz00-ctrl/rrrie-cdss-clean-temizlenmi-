from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cdss.runtime.resolver import resolve_runtime_assets  # noqa: E402

ENV_FILE = PROJECT_ROOT / ".env"
GUI_PORT = 7860

REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("httpx", "httpx"),
    ("requests", "requests"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("structlog", "structlog"),
]

OPTIONAL_MODULES = [
    ("google.genai", "google-genai"),
]


def _read_pyvenv_home(pyvenv_cfg: Path) -> Path | None:
    if not pyvenv_cfg.exists():
        return None
    try:
        for line in pyvenv_cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip().lower() == "home":
                raw = value.strip().strip('"').strip("'")
                return Path(raw) if raw else None
    except Exception:
        return None
    return None


def inspect_project_venv(project_root: Path, current_executable: str | None = None) -> tuple[bool, str]:
    venv_dir = project_root / ".venv"
    venv_python = venv_dir / "Scripts" / "python.exe"
    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    current_path = Path(current_executable or sys.executable).resolve()

    if not venv_dir.exists():
        return False, "No .venv detected in project root."
    if not venv_python.exists():
        return False, f"Detected .venv at {venv_dir}, but .venv\\Scripts\\python.exe is missing. Run .\\setup\\install.ps1 -Repair."

    configured_home = _read_pyvenv_home(pyvenv_cfg)
    if configured_home is not None and not configured_home.exists():
        return False, f"Detected .venv at {venv_dir}, but its base interpreter is missing ({configured_home}). Run .\\setup\\install.ps1 -Repair."

    try:
        probe = subprocess.run(
            [str(venv_python), "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if probe.returncode != 0:
            return False, f"Detected .venv at {venv_dir}, but its Python launcher is unusable. Run .\\setup\\install.ps1 -Repair."
    except Exception:
        return False, f"Detected .venv at {venv_dir}, but its Python launcher is unusable. Run .\\setup\\install.ps1 -Repair."

    if venv_dir in current_path.parents:
        return True, "Running inside project .venv."
    return False, f"Detected a healthy .venv at {venv_dir}, but current interpreter is {current_path}. Use .\\setup\\start.ps1 or activate the environment."


def check_python_version() -> tuple[bool, str]:
    version = ".".join(map(str, sys.version_info[:3]))
    ok = sys.version_info >= (3, 11)
    status = f"Python {version} ({sys.executable})"
    return ok, status


def check_venv() -> tuple[bool, str]:
    return inspect_project_venv(PROJECT_ROOT, sys.executable)


def check_modules(module_specs: list[tuple[str, str]]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for module_name, package_name in module_specs:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return not missing, missing


def check_ports(ports: list[int]) -> list[tuple[int, bool]]:
    results: list[tuple[int, bool]] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            in_use = sock.connect_ex(("127.0.0.1", port)) == 0
        results.append((port, in_use))
    return results

def test_server_inference(port: int, label: str) -> tuple[bool, str]:
    import requests
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "warmup",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content.strip():
            return False, "EMPTY RESPONSE (Silent CUDA failure possible)"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def read_env_keys() -> list[str]:
    if not ENV_FILE.exists():
        return []

    content = ENV_FILE.read_text(encoding="utf-8", errors="replace")
    keys: list[str] = []
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        marker = f"{name}="
        if marker not in content:
            continue
        value = content.split(marker, 1)[1].splitlines()[0].strip()
        if value and "your-" not in value and "example" not in value:
            keys.append(name)
    return keys


def main() -> int:
    print("=" * 60)
    print("RRRIE-CDSS verification")
    print("=" * 60)

    critical_failures: list[str] = []
    warnings: list[str] = []

    python_ok, python_msg = check_python_version()
    print(f"[PYTHON] {python_msg}")
    if not python_ok:
        critical_failures.append("Python 3.11+ is required.")

    venv_ok, venv_msg = check_venv()
    print(f"[VENV] {venv_msg}")
    if not venv_ok:
        critical_failures.append(venv_msg)

    deps_ok, missing_deps = check_modules(REQUIRED_MODULES)
    if deps_ok:
        print("[DEPS] Core dependencies available.")
    else:
        critical_failures.append("Missing required packages: " + ", ".join(missing_deps))

    _, missing_optional = check_modules(OPTIONAL_MODULES)
    if missing_optional:
        warnings.append("Optional cloud package missing: " + ", ".join(missing_optional))

    state = resolve_runtime_assets(PROJECT_ROOT)

    if state.selected_server_exe:
        print(f"[LLAMA] Selected binary: {state.selected_server_exe}")
    else:
        critical_failures.append("No compatible llama-server binary could be resolved.")

    if state.main_model:
        size_gb = state.main_model.stat().st_size / (1024**3)
        print(f"[MODEL] Main: {state.main_model.name} ({size_gb:.2f} GB, GGUF ok)")
    else:
        critical_failures.append("Main 4B model invalid or missing.")

    if state.dllm_model:
        size_gb = state.dllm_model.stat().st_size / (1024**3)
        print(f"[MODEL] DLLM: {state.dllm_model.name} ({size_gb:.2f} GB, GGUF ok)")
    else:
        warnings.append("DLLM 0.8B/0.5B model invalid or missing.")

    print(
        "[GPU] "
        + f"name={state.hardware.name or 'unknown'} total={state.total_vram_gb or 'unknown'}GB "
        + f"free={state.free_vram_gb or 'unknown'}GB cc={state.compute_cap or 'unknown'} "
        + f"driver={state.driver_version or 'unknown'}"
    )
    print(
        f"[RUNTIME] recommended_profile={state.runtime_profile} "
        + f"dllm_mode={state.dllm_mode} topology={state.server_topology}"
    )
    print(f"[BINARY] compatibility={state.binary_compatibility}")
    print(f"[URLS] llama={state.llama_server_url} dllm={state.dllm_server_url}")

    warnings.extend(state.warnings)
    critical_failures.extend(state.fatal_issues)

    if ENV_FILE.exists():
        print("[ENV] .env present.")
    else:
        warnings.append(".env is missing. Defaults will be used until you create it.")

    keys = read_env_keys()
    if keys:
        print(f"[ENV] API keys configured: {', '.join(keys)}")
    else:
        warnings.append("No cloud API key detected. Local-only mode can still run if the local assets are present.")

    ports = [GUI_PORT, 8080]
    if state.server_topology == "dual_server":
        ports.append(8081)
        
    for port, in_use in check_ports(ports):
        label = "IN USE" if in_use else "free"
        msg = f"[PORT] Port {port}: {label}"
        
        if in_use and port in (8080, 8081):
            server_label = "main" if port == 8080 else "dllm"
            inference_ok, inference_msg = test_server_inference(port, server_label)
            if inference_ok:
                msg += f" (Inference: {inference_msg})"
            else:
                msg += f" (Inference FAILED: {inference_msg})"
                critical_failures.append(f"Server on port {port} is running but inference failed: {inference_msg}. Check GPU compatibility and Flash Attention.")
                
        print(msg)

    print("-" * 60)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if state.recommendations:
        print("Recommended actions:")
        for recommendation in state.recommendations:
            print(f"  - {recommendation}")

    if critical_failures:
        print("Critical issues:")
        for failure in critical_failures:
            print(f"  - {failure}")
        print("-" * 60)
        print("Verification failed.")
        return 1

    print("Verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
