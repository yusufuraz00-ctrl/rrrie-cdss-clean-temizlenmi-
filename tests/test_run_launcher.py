import sys
from pathlib import Path
import types

import run as launcher


class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


def test_run_tests_falls_back_to_pytest_when_legacy_script_missing(monkeypatch):
    original_exists = Path.exists

    def fake_exists(self):
        if self.name == "test_e2e_medical.py":
            return False
        return original_exists(self)

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _Completed(returncode=0)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    rc = launcher.run_tests([])

    assert rc == 0
    cmd = list(captured["cmd"])
    assert cmd[:3] == [sys.executable, "-m", "pytest"]
    assert "tests/test_cdss_smoke.py" in cmd


def test_main_returns_test_exit_code(monkeypatch):
    monkeypatch.setattr(launcher, "ensure_servers", lambda: (None, None))
    monkeypatch.setattr(launcher, "run_tests", lambda _args: 5)
    monkeypatch.setattr(launcher, "_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run.py", "--test"])

    rc = launcher.main()

    assert rc == 5


def test_run_tests_supports_retrieval_preset(monkeypatch):
    original_exists = Path.exists

    def fake_exists(self):
        if self.name == "test_e2e_medical.py":
            return False
        return original_exists(self)

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _Completed(returncode=0)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    rc = launcher.run_tests(["--test-preset", "retrieval"])

    assert rc == 0
    cmd = list(captured["cmd"])
    assert cmd[:3] == [sys.executable, "-m", "pytest"]
    assert "tests/test_cdss_mode_and_evidence.py" in cmd


def test_main_runs_preflight(monkeypatch):
    monkeypatch.setattr(launcher, "run_preflight_checks", lambda json_mode=False: 0)
    monkeypatch.setattr(launcher, "_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run.py", "--preflight"])

    rc = launcher.main()

    assert rc == 0


def test_main_returns_one_for_launcher_error(monkeypatch):
    monkeypatch.setattr(launcher, "ensure_servers", lambda: (_ for _ in ()).throw(launcher.LauncherError("launcher_bootstrap_failed", "boom")))
    monkeypatch.setattr(launcher, "_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run.py"])

    rc = launcher.main()

    assert rc == 1


def test_main_returns_one_for_unhandled_exception(monkeypatch):
    monkeypatch.setattr(launcher, "ensure_servers", lambda: (_ for _ in ()).throw(RuntimeError("unexpected")))
    monkeypatch.setattr(launcher, "_print", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run.py"])

    rc = launcher.main()

    assert rc == 1


def test_build_launch_spec_enables_reasoning_for_main_server():
    spec_standard = launcher._build_launch_spec(
        "standard_6gb",
        Path("models/Qwen3.5-4B-Q4_K_M.gguf"),
        launcher.MAIN_LLAMA_PORT,
        is_dllm=False,
    )
    spec_compact = launcher._build_launch_spec(
        "compact_4gb",
        Path("models/Qwen3.5-4B-Q4_K_M.gguf"),
        launcher.MAIN_LLAMA_PORT,
        is_dllm=False,
    )

    assert spec_standard.reasoning == "on"
    assert spec_standard.reasoning_budget >= 512
    assert spec_compact.reasoning == "on"
    assert spec_compact.reasoning_budget >= 384


def test_build_launch_spec_scales_reasoning_budget(monkeypatch):
    monkeypatch.setenv("RUNTIME_REASONING_BUDGET_SCALE", "1.5")

    spec = launcher._build_launch_spec(
        "standard_6gb",
        Path("models/Qwen3.5-4B-Q4_K_M.gguf"),
        launcher.MAIN_LLAMA_PORT,
        is_dllm=False,
    )

    assert spec.reasoning == "on"
    assert spec.reasoning_budget >= 768


def test_extract_warmup_message_accepts_reasoning_payload():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning": "internal reasoning trace",
                }
            }
        ]
    }

    text = launcher._extract_warmup_message(payload)

    assert text == "internal reasoning trace"


def test_warmup_server_accepts_generated_tokens_without_message_content(monkeypatch):
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": ""}}],
                "usage": {"completion_tokens": 6},
            }

    fake_requests = types.SimpleNamespace(post=lambda *args, **kwargs: FakeResponse())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setattr(launcher, "_print", lambda *args, **kwargs: None)

    assert launcher._warmup_server("127.0.0.1", 8080, "main") is True
