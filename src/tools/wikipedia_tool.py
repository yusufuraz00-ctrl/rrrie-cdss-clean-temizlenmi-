"""Wikipedia Medical Summary tool — reliable fallback for disease overviews.

Uses the Wikimedia REST API to get structured disease summaries.
Always returns data for known medical conditions when PubMed/Europe PMC fail.
Free, no API key required.

API docs: https://en.wikipedia.org/api/rest_v1/
"""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://en.wikipedia.org/api/rest_v1"
_TIMEOUT = 15.0
_HEADERS = {
    "User-Agent": "RRRIE-CDSS/1.0 (Clinical Decision Support; https://github.com/rrrie-cdss)",
    "Accept": "application/json",
}


async def search_wikipedia_medical(
    disease_name: str,
    lang: str = "en",
) -> Dict:
    """Get a Wikipedia summary for a medical condition.

    Args:
        disease_name: Name of the disease/condition (e.g., "Botulism", "Pulmonary embolism").
        lang: Language code (default: "en").

    Returns:
        Dict with "title", "summary", "url", "description", and optionally "infobox" data.
    """
    # Normalize: capitalize first letter (Wikipedia convention)
    normalized = disease_name.strip()
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]

    # Replace spaces with underscores for REST API
    page_title = normalized.replace(" ", "_")

    logger.info("wikipedia_search", disease=disease_name, page=page_title)

    async with httpx.AsyncClient(
        timeout=_TIMEOUT,
        headers=_HEADERS,
        follow_redirects=True,
    ) as client:
        # Step 1: Try direct page summary
        result = await _fetch_summary(client, page_title, lang)

        # Step 2: If not found, try search API to find the right page
        if result.get("error") and "not found" in result.get("error", "").lower():
            search_result = await _search_page(client, disease_name, lang)
            if search_result:
                result = await _fetch_summary(client, search_result, lang)

    return result


async def _fetch_summary(client: httpx.AsyncClient, page_title: str, lang: str) -> Dict:
    """Fetch page summary from Wikipedia REST API."""
    base = f"https://{lang}.wikipedia.org/api/rest_v1"
    url = f"{base}/page/summary/{page_title}"

    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return {
                "title": page_title,
                "summary": "",
                "url": "",
                "error": "Page not found",
            }
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("wikipedia_error", error=str(exc))
        return {"title": page_title, "summary": "", "url": "", "error": str(exc)}

    summary = data.get("extract", "")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    result = {
        "title": data.get("title", page_title),
        "summary": summary,
        "url": page_url,
        "description": data.get("description", ""),
        "page_type": data.get("type", ""),  # "standard", "disambiguation", etc.
    }

    logger.info("wikipedia_result", title=result["title"],
                 summary_len=len(summary), has_url=bool(page_url))
    return result


async def _search_page(client: httpx.AsyncClient, query: str, lang: str) -> Optional[str]:
    """Search Wikipedia for the best matching page title."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 3,
        "format": "json",
    }

    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if results:
            # Return the first result's title
            return results[0].get("title", "").replace(" ", "_")
    except Exception as exc:
        logger.warning("wikipedia_search_fallback_error", error=str(exc))

    return None
