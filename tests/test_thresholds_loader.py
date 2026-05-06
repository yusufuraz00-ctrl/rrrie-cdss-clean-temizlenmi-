"""Unit tests for the centralized clinical-thresholds loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.cdss.core import thresholds as ct


def test_known_value_loads_from_default_config():
    ct.reload()
    assert ct.get_float("vitals.spo2_critical_lt", -1.0) == pytest.approx(92.0)
    assert ct.get_int("fusion.rrf_k", -1) == 60


def test_missing_path_returns_default():
    ct.reload()
    assert ct.get_float("does.not.exist", 7.5) == pytest.approx(7.5)
    assert ct.get_int("nope.gone", 11) == 11


def test_metadata_includes_source_and_last_reviewed():
    ct.reload()
    meta = ct.metadata("vitals.spo2_critical_lt")
    assert meta.get("value") == 92
    assert "Surviving Sepsis Campaign" in meta.get("source", "")
    assert meta.get("last_reviewed", "")


def test_loader_falls_back_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CDSS_CLINICAL_THRESHOLDS_PATH", str(tmp_path / "absent.json"))
    ct.reload()
    assert ct.get_float("vitals.spo2_critical_lt", 99.9) == pytest.approx(99.9)


def test_loader_tolerates_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = tmp_path / "broken.json"
    bad.write_text("{ this is not json ")
    monkeypatch.setenv("CDSS_CLINICAL_THRESHOLDS_PATH", str(bad))
    ct.reload()
    assert ct.get_float("any.path", 3.14) == pytest.approx(3.14)


def test_loader_accepts_bare_numeric_legacy_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "legacy.json"
    cfg.write_text(json.dumps({"vitals": {"spo2_critical_lt": 95}}))
    monkeypatch.setenv("CDSS_CLINICAL_THRESHOLDS_PATH", str(cfg))
    ct.reload()
    assert ct.get_int("vitals.spo2_critical_lt", -1) == 95


def teardown_module(_module):
    if "CDSS_CLINICAL_THRESHOLDS_PATH" in os.environ:
        del os.environ["CDSS_CLINICAL_THRESHOLDS_PATH"]
    ct.reload()
