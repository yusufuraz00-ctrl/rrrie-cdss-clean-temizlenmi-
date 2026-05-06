"""Unit tests for the in-process tracing span emitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cdss.core import tracing


def test_span_records_duration_and_attributes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDSS_TRACE_DIR", str(tmp_path))
    tracing.set_run_id("run_simple")

    captured: list[tracing.SpanRecord] = []
    unsubscribe = tracing.subscribe(captured.append)

    with tracing.span("intake", stage="INTAKE") as rec:
        tracing.add_counter(rec, "llm_calls", 1)
        tracing.set_attribute(rec, "worker", "general")

    unsubscribe()
    tracing.close_run("run_simple")

    assert len(captured) == 1
    assert captured[0].name == "intake"
    assert captured[0].duration_ms is not None and captured[0].duration_ms >= 0.0
    assert captured[0].counters.get("llm_calls") == 1.0
    assert captured[0].attributes.get("worker") == "general"

    trace_file = tmp_path / "run_simple.jsonl"
    assert trace_file.exists()
    rec = json.loads(trace_file.read_text(encoding="utf-8").strip())
    assert rec["name"] == "intake"
    assert rec["status"] == "ok"


def test_nested_spans_set_parent_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDSS_TRACE_DIR", str(tmp_path))
    tracing.set_run_id("run_nested")

    captured: list[tracing.SpanRecord] = []
    unsubscribe = tracing.subscribe(captured.append)
    with tracing.span("outer") as outer:
        with tracing.span("inner"):
            pass
    unsubscribe()
    tracing.close_run("run_nested")

    by_name = {r.name: r for r in captured}
    assert by_name["inner"].parent_id == by_name["outer"].span_id
    assert by_name["outer"].parent_id is None


def test_span_records_error_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDSS_TRACE_DIR", str(tmp_path))
    tracing.set_run_id("run_err")

    captured: list[tracing.SpanRecord] = []
    unsubscribe = tracing.subscribe(captured.append)
    with pytest.raises(ValueError):
        with tracing.span("boom"):
            raise ValueError("kaboom")
    unsubscribe()
    tracing.close_run("run_err")

    assert captured and captured[0].status == "error"
    assert "kaboom" in (captured[0].error or "")


def test_disable_env_skips_file_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDSS_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("CDSS_TRACE_DISABLE", "1")
    tracing.set_run_id("run_disabled")

    with tracing.span("anything"):
        pass
    tracing.close_run("run_disabled")

    assert not (tmp_path / "run_disabled.jsonl").exists()


def teardown_function(_fn):
    tracing.set_run_id(None)
