"""Medical code helpers for adaptive ICD-11 validation and WHO lookup."""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ICD-11 stem-code format:
#   - 1-2 uppercase letters followed by digits, optional dot + digits
#   - extension codes may follow separated by '&'
ICD11_PATTERN = re.compile(
    r"^(?:[A-Z]\d|[A-Z]{2}|\d[A-Z])[A-Z0-9]{1,4}(\.\d{1,3})?(\/[A-Z0-9.]+)?(&[A-Z0-9.]+)*$",
    re.IGNORECASE,
)

_WHO_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
_WHO_MMS_SEARCH = "https://id.who.int/icd/release/11/2024-01/mms/search"
_WHO_CODEINFO = "https://id.who.int/icd/release/11/2024-01/mms/codeinfo"

_cached_token: str | None = None
_token_expiry: float = 0.0


def is_valid_icd11(code: str) -> bool:
    """Check if a string looks like a valid ICD-11 stem code."""
    return bool(ICD11_PATTERN.match(code.strip()))


def normalize_icd11(code: str) -> Optional[str]:
    """Normalize an ICD-11 code to uppercase and return None if invalid."""
    cleaned = code.strip().upper()
    if is_valid_icd11(cleaned):
        return cleaned
    return None


def _strip_html(text: str) -> str:
    """Strip HTML tags from WHO API results."""
    return re.sub(r"<[^>]+>", "", text).strip()


async def _get_who_bearer_token(
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str | None:
    """Obtain and cache a WHO ICD-11 bearer token."""
    global _cached_token, _token_expiry

    if _cached_token and time.time() < (_token_expiry - 60):
        return _cached_token

    if not client_id or not client_secret:
        try:
            from config.settings import get_settings

            settings = get_settings()
            client_id = client_id or settings.icd11_client_id
            client_secret = client_secret or settings.icd11_client_secret
        except Exception:
            pass

    if not client_id or not client_secret:
        logger.debug("who_icd_no_credentials", msg="No ICD-11 client credentials configured")
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _WHO_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                    "scope": "icdapi_access",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if resp.status_code != 200:
            logger.warning(
                "who_icd_token_failed",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return None
        data = resp.json()
        _cached_token = str(data.get("access_token", "") or "")
        _token_expiry = time.time() + int(data.get("expires_in", 3600) or 3600)
        return _cached_token or None
    except Exception as exc:
        logger.error("who_icd_token_error", error=str(exc))
        return None


async def lookup_icd11_who(
    code: str,
    api_token: str | None = None,
) -> dict:
    """Look up an ICD-11 code via the WHO MMS codeinfo endpoint."""
    if not api_token:
        api_token = await _get_who_bearer_token()

    headers = {
        "Accept": "application/json",
        "Accept-Language": "en",
        "API-Version": "v2",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    url = f"{_WHO_CODEINFO}/{code.strip()}"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.warning("icd11_who_lookup_failed", code=code, status=resp.status_code)
                return {"code": code, "found": False, "status_code": resp.status_code}

            data = resp.json()
            stem_id = str(data.get("stemId", "") or "")
            result = {
                "code": data.get("code", code),
                "found": True,
                "stemId": stem_id,
                "title": "",
                "definition": "",
                "browserUrl": "",
            }
            if stem_id:
                try:
                    entity_resp = await client.get(stem_id, headers=headers)
                    if entity_resp.status_code == 200:
                        entity = entity_resp.json()
                        title_obj = entity.get("title", {})
                        definition_obj = entity.get("definition", {})
                        result["title"] = title_obj.get("@value", "") if isinstance(title_obj, dict) else str(title_obj)
                        result["definition"] = definition_obj.get("@value", "") if isinstance(definition_obj, dict) else str(definition_obj)
                        result["browserUrl"] = entity.get("browserUrl", "")
                        result["classKind"] = entity.get("classKind", "")
                except Exception as exc:
                    logger.debug("icd11_stemid_follow_error", stem_id=stem_id, error=str(exc))
            return result
    except Exception as exc:
        logger.error("icd11_who_lookup_error", code=code, error=str(exc))
        return {"code": code, "found": False, "error": str(exc)}


async def search_icd11_who(
    query: str,
    api_token: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search WHO ICD-11 MMS by free-text diagnosis."""
    if not api_token:
        api_token = await _get_who_bearer_token()

    headers = {
        "Accept": "application/json",
        "Accept-Language": "en",
        "API-Version": "v2",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    params = {
        "q": query,
        "useFlexisearch": "true",
        "flatResults": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_WHO_MMS_SEARCH, headers=headers, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results: list[dict] = []
        for item in data.get("destinationEntities", [])[:max_results]:
            results.append(
                {
                    "id": item.get("id", ""),
                    "title": _strip_html(str(item.get("title", "") or "")),
                    "theCode": item.get("theCode", ""),
                    "score": item.get("score", 0),
                }
            )
        return results
    except Exception as exc:
        logger.error("icd11_who_search_error", query=query, error=str(exc))
        return []
