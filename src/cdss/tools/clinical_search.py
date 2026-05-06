"""Reactive clinical search tool — called when LLM emits NEEDS_SEARCH markers."""

from __future__ import annotations

from typing import Any


import logging
logger = logging.getLogger(__name__)

class ClinicalSearchTool:
    """Executes medical searches and returns text snippets for LLM context enrichment."""

    async def search(self, query: str, max_results: int = 5) -> list[str]:
        """Run query through available search backends; return list of text snippets."""
        snippets: list[str] = []
        try:
            from src.tools.web_search_tool import web_search
            result: dict[str, Any] = await web_search(query, max_results=max_results)
            for item in (result.get("results") or [])[:max_results]:
                content = str(item.get("content") or item.get("snippet") or "").strip()
                title = str(item.get("title") or "").strip()
                if content:
                    snippets.append(f"{title}: {content}" if title else content)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        if not snippets:
            snippets = await self._pubmed_fallback(query, max_results)

        return snippets[:max_results]

    async def _pubmed_fallback(self, query: str, max_results: int) -> list[str]:
        try:
            from src.tools.pubmed_tool import search_pubmed
            result: dict[str, Any] = await search_pubmed(query, max_results=max_results)
            snippets = []
            for article in (result.get("articles") or [])[:max_results]:
                title = str(article.get("title") or "").strip()
                abstract = str(article.get("abstract") or "").strip()[:300]
                if abstract:
                    snippets.append(f"{title}: {abstract}" if title else abstract)
                elif title:
                    snippets.append(title)
            return snippets
        except Exception:
            return []
