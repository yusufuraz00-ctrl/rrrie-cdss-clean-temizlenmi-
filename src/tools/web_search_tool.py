"""Web search tool - Tavily primary with biomedical fallbacks.

Fallback chain:
  1. Tavily (advanced search, trusted medical domains)
  2. PubMed (NCBI E-Utilities)
  3. Europe PMC (free, no key required)
  4. Wikipedia medical summary (free, no key required)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from config.api_config import TavilyConfig
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.rate_limiter import tool_cache, web_limiter

logger = get_logger(__name__)

_config = TavilyConfig()


def _pubmed_articles_to_web_result(pubmed_result: Dict, query: str) -> Dict:
    """Convert PubMed search result into the standard web_search result format."""
    articles = pubmed_result.get("articles") or []
    results = []
    for article in articles[:5]:
        title = str(article.get("title") or "").strip()
        pmid = str(article.get("pmid") or "").strip()
        abstract = str(article.get("abstract") or "").strip()
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        if not title:
            continue
        results.append({
            "title": title,
            "url": url,
            "content": abstract[:500] if abstract else title,
            "score": 0.75,
        })
    answer = results[0]["content"] if results else ""
    return {
        "query": query,
        "answer": answer,
        "results": results,
        "_source": "pubmed_fallback",
        "_from_cache": False,
    }


def _europe_pmc_articles_to_web_result(europe_pmc_result: Dict, query: str) -> Dict:
    """Convert Europe PMC search result into the standard web_search result format."""
    articles = europe_pmc_result.get("articles") or []
    results = []
    for article in articles[:5]:
        title = str(article.get("title") or "").strip()
        abstract = str(article.get("abstract") or "").strip()
        url = str(article.get("url") or "").strip()
        pmid = str(article.get("pmid") or "").strip()
        if not url and pmid:
            url = f"https://europepmc.org/article/MED/{pmid}"
        if not title:
            continue
        results.append({
            "title": title,
            "url": url,
            "content": abstract[:500] if abstract else title,
            "score": 0.70,
        })
    answer = results[0]["content"] if results else ""
    return {
        "query": query,
        "answer": answer,
        "results": results,
        "_source": "europe_pmc_fallback",
        "_from_cache": False,
    }


def _wikipedia_to_web_result(wiki_result: Dict, query: str) -> Dict:
    """Convert Wikipedia summary into the standard web_search result format."""
    title = str(wiki_result.get("title") or "").strip()
    summary = str(wiki_result.get("summary") or "").strip()
    url = str(wiki_result.get("url") or "").strip()
    if not title or not summary:
        return {"results": [], "answer": "", "error": "wikipedia_empty"}
    results = [{"title": title, "url": url, "content": summary[:500], "score": 0.60}]
    return {
        "query": query,
        "answer": summary[:300],
        "results": results,
        "_source": "wikipedia_fallback",
        "_from_cache": False,
    }


def _mark_cache_hit(result: Dict) -> Dict:
    cached = dict(result)
    cached["_from_cache"] = True
    return cached


def _build_payload(
    query: str,
    search_depth: str,
    include_domains: Optional[List[str]],
    max_results: int | None,
) -> Dict:
    payload: Dict = {
        "query": query,
        "search_depth": search_depth or _config.default_search_depth,
        "include_answer": True,
        "max_results": max_results or _config.default_max_results,
    }
    payload["include_domains"] = (
        include_domains if include_domains else list(_config.trusted_medical_domains)
    )
    return payload


def _web_search_cache_key(payload: Dict, search_depth: str, max_results: int | None) -> str:
    return tool_cache.make_key(
        "web_search",
        query=payload.get("query", ""),
        depth=search_depth,
        include_domains=tuple(payload.get("include_domains") or ()),
        max_results=max_results or _config.default_max_results,
    )


async def _tavily_search(payload: Dict, tavily_api_key: str, query: str) -> tuple[Dict, str]:
    if not tavily_api_key:
        logger.warning("tavily_no_api_key")
        return {}, "Tavily API key not configured"

    request_payload = dict(payload)
    request_payload["api_key"] = tavily_api_key
    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        try:
            async with web_limiter:
                resp = await client.post(_config.base_url, json=request_payload)
            resp.raise_for_status()
            return resp.json(), ""
        except Exception as exc:
            error = str(exc)
            logger.warning("web_search_tavily_error", error=error, query=query)
            return {}, error


def _tavily_to_web_result(data: Dict, query: str) -> Dict:
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
            "score": r.get("score", 0),
        })
    answer = data.get("answer", "")
    logger.info("web_search_results", count=len(results), has_answer=bool(answer))
    return {
        "query": query,
        "answer": answer,
        "results": results,
        "_source": "tavily",
        "_from_cache": False,
    }


async def _pubmed_fallback(query: str, max_results: int | None) -> Dict | None:
    try:
        from src.tools.pubmed_tool import search_pubmed
        pubmed_result = await search_pubmed(query=query, max_results=max_results or 3)
        if not pubmed_result.get("error") and pubmed_result.get("articles"):
            logger.info("web_search_pubmed_fallback", query=query)
            return _pubmed_articles_to_web_result(pubmed_result, query)
    except Exception as exc:
        logger.warning("web_search_pubmed_fallback_error", error=str(exc))
    return None


async def _europe_pmc_fallback(query: str, max_results: int | None) -> Dict | None:
    try:
        from src.tools.europe_pmc_tool import search_europe_pmc
        europe_pmc_result = await search_europe_pmc(query=query, max_results=max_results or 3)
        if not europe_pmc_result.get("error") and europe_pmc_result.get("articles"):
            logger.info("web_search_europe_pmc_fallback", query=query)
            return _europe_pmc_articles_to_web_result(europe_pmc_result, query)
    except Exception as exc:
        logger.warning("web_search_europe_pmc_fallback_error", error=str(exc))
    return None


async def _wikipedia_fallback(query: str) -> Dict | None:
    try:
        from src.tools.wikipedia_tool import search_wikipedia_medical
        wiki_query = query.replace("diagnosis", "").replace("treatment", "").strip()
        wiki_result = await search_wikipedia_medical(wiki_query)
        if wiki_result and not wiki_result.get("error") and wiki_result.get("summary"):
            logger.info("web_search_wikipedia_fallback", query=query)
            return _wikipedia_to_web_result(wiki_result, query)
    except Exception as exc:
        logger.warning("web_search_wikipedia_fallback_error", error=str(exc))
    return None


async def web_search(
    query: str,
    search_depth: str = "advanced",
    include_domains: Optional[List[str]] = None,
    max_results: int | None = None,
) -> Dict:
    """Search the web for medical information using Tavily and safe fallbacks."""
    settings = get_settings()
    payload = _build_payload(query, search_depth, include_domains, max_results)
    logger.info("web_search", query=query, depth=search_depth)

    cache_key = _web_search_cache_key(payload, search_depth, max_results)
    cached = tool_cache.get(cache_key)
    if cached is not None:
        logger.info("web_search_cache_hit", query=query)
        return _mark_cache_hit(cached)

    data, tavily_error = await _tavily_search(payload, settings.tavily_api_key, query)
    result = _tavily_to_web_result(data, query) if data else None
    result = result or await _pubmed_fallback(query, max_results)
    result = result or await _europe_pmc_fallback(query, max_results)
    result = result or await _wikipedia_fallback(query)

    if result:
        tool_cache.set(cache_key, result)
        return result

    logger.error("web_search_all_backends_failed", query=query, tavily_error=tavily_error)
    return {"results": [], "answer": "", "error": "all_search_backends_failed"}
