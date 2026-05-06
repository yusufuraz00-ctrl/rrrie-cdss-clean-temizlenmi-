from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_runtime_policy_exposes_reactive_search_limit(monkeypatch):
    from src.cdss.runtime.policy import load_runtime_policy

    monkeypatch.setenv("CDSS_REACTIVE_SEARCH_MAX_ITERATIONS", "4")

    policy = load_runtime_policy()

    assert policy.reactive_search_max_iterations == 4


def test_normalize_web_search_results_accepts_dict_and_list_shapes():
    from src.cdss.runtime.state_machine import _normalize_web_search_results

    assert _normalize_web_search_results(
        {"results": [{"title": "A", "content": "alpha"}], "_from_cache": True}
    ) == [{"title": "A", "content": "alpha"}]

    assert _normalize_web_search_results(
        [{"title": "B", "snippet": "beta"}, "not-a-result", {"title": "C"}]
    ) == [{"title": "B", "snippet": "beta"}, {"title": "C"}]


def test_web_search_falls_back_to_pubmed_without_tavily_key(monkeypatch):
    from src.tools import pubmed_tool
    from src.tools import web_search_tool

    web_search_tool.tool_cache.clear()
    monkeypatch.setattr(web_search_tool, "get_settings", lambda: SimpleNamespace(tavily_api_key=""))

    async def fake_pubmed_search(query: str, max_results: int):
        return {
            "articles": [
                {
                    "pmid": "123",
                    "title": "Clinical evidence title",
                    "abstract": "Clinical evidence abstract",
                }
            ]
        }

    monkeypatch.setattr(pubmed_tool, "search_pubmed", fake_pubmed_search)

    result = asyncio.run(web_search_tool.web_search("clinical evidence", max_results=1))

    assert result["_source"] == "pubmed_fallback"
    assert result["results"][0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert "Clinical evidence abstract" in result["answer"]


def test_web_search_uses_europe_pmc_before_wikipedia_when_pubmed_empty(monkeypatch):
    from src.tools import europe_pmc_tool
    from src.tools import pubmed_tool
    from src.tools import web_search_tool
    from src.tools import wikipedia_tool

    web_search_tool.tool_cache.clear()
    monkeypatch.setattr(web_search_tool, "get_settings", lambda: SimpleNamespace(tavily_api_key=""))

    async def empty_pubmed_search(query: str, max_results: int):
        return {"articles": []}

    async def fake_europe_pmc_search(query: str, max_results: int):
        return {
            "articles": [
                {
                    "pmid": "456",
                    "title": "Europe PMC evidence",
                    "abstract": "Europe PMC abstract",
                    "url": "https://europepmc.org/article/MED/456",
                }
            ]
        }

    async def wikipedia_should_not_run(query: str):
        raise AssertionError("Wikipedia fallback should not run before Europe PMC")

    monkeypatch.setattr(pubmed_tool, "search_pubmed", empty_pubmed_search)
    monkeypatch.setattr(europe_pmc_tool, "search_europe_pmc", fake_europe_pmc_search)
    monkeypatch.setattr(wikipedia_tool, "search_wikipedia_medical", wikipedia_should_not_run)

    result = asyncio.run(web_search_tool.web_search("clinical evidence", max_results=1))

    assert result["_source"] == "europe_pmc_fallback"
    assert result["results"][0]["url"] == "https://europepmc.org/article/MED/456"
    assert "Europe PMC abstract" in result["answer"]
