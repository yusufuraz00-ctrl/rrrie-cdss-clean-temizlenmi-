import pytest


@pytest.fixture(autouse=True)
def disable_external_side_effects_by_default(monkeypatch):
    monkeypatch.setenv("CDSS_LOCAL_LLM_EXTRACTION_ENABLED", "0")
    monkeypatch.setenv("CDSS_LOCAL_LLM_RANKER_ENABLED", "0")
    monkeypatch.setenv("CDSS_LOCAL_LLM_VERIFIER_ENABLED", "0")
    monkeypatch.setenv("CDSS_LOCAL_LLM_CHALLENGER_ENABLED", "0")
    monkeypatch.setenv("CDSS_EXTERNAL_EVIDENCE_ENABLED", "0")
