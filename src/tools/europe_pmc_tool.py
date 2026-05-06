"""Europe PMC API tool — searches European open-access biomedical literature.

Europe PMC aggregates ~40M articles from PubMed, PMC, patents, and preprints.
Key advantage over PubMed: returns FULL abstracts by default, no extra API call.
Free, no API key required.

API docs: https://europepmc.org/RestfulWebService
"""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TIMEOUT = 30.0


async def search_europe_pmc(
    query: str,
    max_results: int = 5,
    source: str = "MED",
) -> Dict:
    """Search Europe PMC for biomedical literature.

    Args:
        query: Search query (supports Boolean operators, MeSH terms).
        max_results: Maximum articles to return (default 5, max 25).
        source: Source database — "MED" (PubMed/MEDLINE), "PMC" (full-text), "PPR" (preprints).

    Returns:
        Dict with "articles" list and "total_found" count.
    """
    params = {
        "query": query,
        "resultType": "core",
        "pageSize": min(max_results, 25),
        "format": "json",
    }

    # Only apply source filter for specific non-default sources
    if source and source not in ("ALL", "MED"):
        params["query"] = f"(SRC:{source}) AND ({query})"

    logger.info("europe_pmc_search", query=query, source=source)

    async with httpx.AsyncClient(
        timeout=_TIMEOUT,
        headers={"User-Agent": "RRRIE-CDSS/1.0"},
    ) as client:
        try:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("europe_pmc_error", error=str(exc))
            return {"articles": [], "total_found": 0, "error": str(exc)}

    result_list = data.get("resultList", {}).get("result", [])
    total_found = int(data.get("hitCount", 0))

    articles: List[Dict] = []
    for item in result_list:
        pmid = item.get("pmid", "")
        articles.append({
            "pmid": pmid,
            "title": item.get("title", ""),
            "authors": _extract_authors(item),
            "journal": item.get("journalTitle", ""),
            "pub_date": item.get("firstPublicationDate", ""),
            "doi": item.get("doi", ""),
            "url": f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
            "abstract": item.get("abstractText", ""),
            "citation_count": item.get("citedByCount", 0),
            "is_open_access": item.get("isOpenAccess", "N") == "Y",
            "source": item.get("source", ""),
        })

    logger.info("europe_pmc_results", count=len(articles), total=total_found,
                 abstracts=sum(1 for a in articles if a.get("abstract")))
    return {"articles": articles, "total_found": total_found}


def _extract_authors(item: Dict) -> List[str]:
    """Extract author names from Europe PMC result."""
    author_list = item.get("authorList", {}).get("author", [])
    names = []
    for a in author_list[:5]:
        full = a.get("fullName", "")
        if full:
            names.append(full)
    # Fallback: authorString
    if not names:
        author_str = item.get("authorString", "")
        if author_str:
            names = [n.strip() for n in author_str.split(",")[:5]]
    return names
