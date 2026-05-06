"""PubMed E-Utilities API tool — searches medical literature with abstract retrieval."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import httpx

from config.api_config import PubMedConfig
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.rate_limiter import pubmed_limiter, tool_cache

logger = get_logger(__name__)

_config = PubMedConfig()


def _compact_query(query: str, *, max_terms: int = 10) -> str:
    tokens = [token for token in str(query or "").replace("(", " ").replace(")", " ").split() if token]
    return " ".join(tokens[:max_terms]).strip()


async def search_pubmed(
    query: str,
    max_results: int = 5,
    date_range: Optional[str] = None,
    article_types: Optional[List[str]] = None,
) -> Dict:
    """Search PubMed for medical literature.

    Args:
        query: Search query (MeSH terms supported).
        max_results: Maximum articles to return (default 5).
        date_range: Date range filter, e.g. "2023/01/01:2025/12/31".
        article_types: Filter by article type: review, clinical-trial, meta-analysis, guideline.

    Returns:
        Dict with "articles" list and "total_found" count.
    """
    settings = get_settings()
    api_key = settings.ncbi_api_key
    email = settings.ncbi_email

    # ── TTL cache check ──
    cache_key = tool_cache.make_key(
        "pubmed", query=query, max_results=max_results,
        date_range=date_range, article_types=article_types,
    )
    cached = tool_cache.get(cache_key)
    if cached is not None:
        logger.info("pubmed_cache_hit", query=query)
        return cached

    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        # ── Step 1: eSearch — find PubMed IDs ────────────────────
        search_params: Dict = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }

        if api_key:
            search_params["api_key"] = api_key
        if email:
            search_params["email"] = email

        if date_range:
            parts = date_range.split(":")
            if len(parts) == 2:
                search_params["datetype"] = "pdat"
                search_params["mindate"] = parts[0]
                search_params["maxdate"] = parts[1]

        if article_types:
            type_filter = " OR ".join(f"{t}[pt]" for t in article_types)
            search_params["term"] = f"({query}) AND ({type_filter})"

        attempted_terms: list[str] = []
        search_data: Dict = {}
        last_error: Exception | None = None
        fallback_terms = [search_params["term"]]
        bare_query = _compact_query(query)
        if bare_query and bare_query not in fallback_terms:
            fallback_terms.append(bare_query)
        compact_query = _compact_query(query, max_terms=6)
        if compact_query and compact_query not in fallback_terms:
            fallback_terms.append(compact_query)

        for term in fallback_terms:
            attempted_terms.append(term)
            current_params = dict(search_params)
            current_params["term"] = term
            logger.info("pubmed_search", query=current_params["term"])
            try:
                async with pubmed_limiter:
                    search_resp = await client.get(
                        f"{_config.base_url}/esearch.fcgi",
                        params=current_params,
                    )
                search_resp.raise_for_status()
                search_data = search_resp.json()
                break
            except Exception as exc:
                last_error = exc
                continue

        if not search_data:
            logger.error("pubmed_search_error", error=str(last_error), attempted_terms=attempted_terms)
            return {"articles": [], "total_found": 0, "error": str(last_error or "pubmed_search_failed")}

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        total_found = int(search_data.get("esearchresult", {}).get("count", 0))

        if not id_list:
            return {"articles": [], "total_found": total_found}

        # ── Step 2: eSummary — get article metadata ──────────────────
        summary_params: Dict = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        }
        if api_key:
            summary_params["api_key"] = api_key
        if email:
            summary_params["email"] = email

        try:
            async with pubmed_limiter:
                summary_resp = await client.get(
                    f"{_config.base_url}/esummary.fcgi",
                    params=summary_params,
                )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
        except Exception as exc:
            logger.error("pubmed_summary_error", error=str(exc))
            return {"articles": [], "total_found": total_found, "error": str(exc)}

        # ── Parse eSummary results ────────────────────────────────────
        articles = []
        result_data = summary_data.get("result", {})
        for pmid in id_list:
            info = result_data.get(pmid, {})
            if not isinstance(info, dict):
                continue
            articles.append({
                "pmid": pmid,
                "title": info.get("title", ""),
                "authors": [a.get("name", "") for a in info.get("authors", [])[:5]],
                "journal": info.get("source", ""),
                "pub_date": info.get("pubdate", ""),
                "doi": info.get("elocationid", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "abstract": "",
            })

        # ── Step 3: eFetch — get abstracts (XML) ────────────────────
        if articles:
            fetch_params: Dict = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "rettype": "abstract",
                "retmode": "xml",
            }
            if api_key:
                fetch_params["api_key"] = api_key
            if email:
                fetch_params["email"] = email

            try:
                async with pubmed_limiter:
                    fetch_resp = await client.get(
                        f"{_config.base_url}/efetch.fcgi",
                        params=fetch_params,
                    )
                fetch_resp.raise_for_status()
                abstracts = _parse_efetch_abstracts(fetch_resp.text)
                for art in articles:
                    art["abstract"] = abstracts.get(art["pmid"], "")
            except Exception as exc:
                logger.warning("pubmed_efetch_error", error=str(exc))
                # Non-fatal: articles still have titles/metadata

        logger.info("pubmed_results", count=len(articles), total=total_found,
                     abstracts=sum(1 for a in articles if a.get("abstract")))
        result = {"articles": articles, "total_found": total_found}
        tool_cache.set(cache_key, result)
        return result


def _parse_efetch_abstracts(xml_text: str) -> Dict[str, str]:
    """Parse eFetch XML response to extract PMID → abstract text mapping."""
    abstracts: Dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
        for article_el in root.findall(".//PubmedArticle"):
            # Get PMID
            pmid_el = article_el.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text

            # Get abstract text (may have multiple AbstractText elements)
            abstract_parts = []
            for abs_text in article_el.findall(".//Abstract/AbstractText"):
                label = abs_text.get("Label", "")
                text = "".join(abs_text.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)

            if abstract_parts:
                abstracts[pmid] = " ".join(abstract_parts)
    except ET.ParseError as exc:
        logger.warning("pubmed_xml_parse_error", error=str(exc))
    return abstracts
