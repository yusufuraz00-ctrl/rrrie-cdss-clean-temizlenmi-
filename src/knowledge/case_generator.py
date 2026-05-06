"""Source-grounded clinical case generation with multi-source auto-review."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from config.settings import PROJECT_ROOT
from src.llm.gemini_client import GeminiClient
from src.tools.europe_pmc_tool import search_europe_pmc
from src.tools.pubmed_tool import search_pubmed
from src.utils.json_payloads import parse_json_from_response, sanitize_json_text
from src.utils.medical_codes import search_icd11_who

logger = logging.getLogger("rrrie-cdss")


GENERATED_CASES_ROOT = PROJECT_ROOT / "data" / "generated_cases"
DRAFTS_DIR = GENERATED_CASES_ROOT / "drafts"
CURATED_DIR = GENERATED_CASES_ROOT / "curated"
REJECTED_DIR = GENERATED_CASES_ROOT / "rejected"
LAST_BATCH_SUMMARY_FILE = GENERATED_CASES_ROOT / "_last_batch_summary.json"
LEGACY_TEST_CASES_DIR = PROJECT_ROOT / "tests" / "test_cases"
TEXTBOOK_DIR = PROJECT_ROOT / "data" / "knowledge" / "textbooks"
APPROVED_BOOK_MANIFEST = TEXTBOOK_DIR / "approved_sources.json"
GENERATOR_VERSION = "casegen-v3"
DECISION_VERSION = "auto-review-v1"
WHO_DON_INDEX_URL = "https://www.who.int/emergencies/disease-outbreak-news"
DEFAULT_BATCH_SOURCES = ["pubmed", "europe_pmc", "who_don"]
MAX_STRUCTURED_ATTEMPTS = 3
MAX_BATCH_ATTEMPT_MULTIPLIER = 3
MAX_BATCH_ATTEMPT_BUFFER = 2
APPROVED_BOOK_HOST_ALLOWLIST = {
    "ncbi.nlm.nih.gov",
    "www.ncbi.nlm.nih.gov",
    "who.int",
    "www.who.int",
}
SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "pubmed": {
        "adapter": "pubmed_eutils",
        "case_kind": "real_report",
        "trust_tier": "high",
        "auto_approve_eligible": True,
        "official_endpoints": ["https://eutils.ncbi.nlm.nih.gov/"],
    },
    "europe_pmc": {
        "adapter": "europe_pmc_rest",
        "case_kind": "real_report",
        "trust_tier": "high",
        "auto_approve_eligible": True,
        "official_endpoints": ["https://www.ebi.ac.uk/europepmc/webservices/rest/search"],
    },
    "who_don": {
        "adapter": "who_don_html",
        "case_kind": "outbreak_teaching",
        "trust_tier": "public_health",
        "auto_approve_eligible": False,
        "official_endpoints": [WHO_DON_INDEX_URL],
    },
    "approved_book": {
        "adapter": "approved_book_allowlist",
        "case_kind": "textbook_teaching",
        "trust_tier": "teaching",
        "auto_approve_eligible": False,
        "official_endpoints": sorted(APPROVED_BOOK_HOST_ALLOWLIST),
    },
}


def _ensure_case_dirs() -> None:
    for path in (DRAFTS_DIR, CURATED_DIR, REJECTED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "")).strip("-").lower()
    return slug or "case"


def _case_filename(case_id: str) -> str:
    return f"{_slugify(case_id)}.json"


def _approved_book_content_available() -> bool:
    if APPROVED_BOOK_MANIFEST.exists():
        return True
    if not TEXTBOOK_DIR.exists():
        return False
    return any(path.is_file() and path.suffix.lower() in {".md", ".txt"} for path in TEXTBOOK_DIR.iterdir())


def _normalize_requested_sources(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        key = str(value or "").strip().lower()
        if key in SOURCE_REGISTRY and key not in normalized:
            normalized.append(key)
    return normalized


class CaseGenerationRequest(BaseModel):
    """User-facing generation controls."""

    model_config = ConfigDict(extra="ignore")

    language: Literal["tr", "en"] = "tr"
    difficulty: Literal["light", "moderate", "severe"] = "moderate"
    specialty: str | None = None
    source_mode: Literal["pubmed_primary", "multi_source_batch"] = "pubmed_primary"
    voice_style: Literal["patient_colloquial", "family_report", "triage_blend"] = "patient_colloquial"
    target_count: int = Field(default=1, ge=1, le=5)
    max_candidates: int = Field(default=20, ge=1, le=60)
    sources: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1, le=5)
    seed: int | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "CaseGenerationRequest":
        if self.count is not None:
            self.target_count = self.count
        if self.source_mode == "pubmed_primary":
            self.sources = ["pubmed"]
        else:
            normalized = _normalize_requested_sources(self.sources)
            if not normalized:
                normalized = list(DEFAULT_BATCH_SOURCES)
                if _approved_book_content_available():
                    normalized.append("approved_book")
            self.sources = normalized
        self.max_candidates = max(self.max_candidates, self.target_count)
        self.count = self.target_count
        return self


class VitalsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temperature: float | None = None
    heart_rate: int | None = None
    respiratory_rate: int | None = None
    blood_pressure: str | None = None
    spo2: float | None = None


class SourceArticle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    identifier: str
    title: str
    abstract: str
    journal: str = ""
    pub_date: str = ""
    url: str = ""
    query: str = ""
    specialty: str = "general"
    difficulty: str = "moderate"
    source_origin: Literal["pubmed", "europe_pmc", "who_don", "approved_book"]
    case_kind: Literal["real_report", "outbreak_teaching", "textbook_teaching"]
    trust_tier: str = "medium"
    auto_approve_eligible: bool = False
    extraction_method: str = "api"
    rights_status: Literal["approved", "unclear", "rejected"] = "approved"
    license_name: str = ""
    license_url: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    who_don_id: str = ""
    book_id: str = ""
    chapter_id: str = ""


class ClinicalFactGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    age: int | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"
    chief_complaint: str
    symptom_duration: str = ""
    symptom_course: str = ""
    symptoms: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    history: str = ""
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    vitals: VitalsModel = Field(default_factory=VitalsModel)
    lab_results: list[str] = Field(default_factory=list)
    primary_diagnosis_text: str
    diagnosis_hint_terms: list[str] = Field(default_factory=list)
    should_detect: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    uncertainty_or_gaps: list[str] = Field(default_factory=list)


class RenderedNarrative(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chief_complaint: str
    patient_text: str
    voice_style: Literal["patient_colloquial", "family_report", "triage_blend"]
    ambiguity_profile: list[str] = Field(default_factory=list)
    language: Literal["tr", "en"]


class NarrativeValidatorResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    voice_score: float = Field(default=0.5, ge=0.0, le=1.0)
    leakage_score: float = Field(default=0.5, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_fidelity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    clinician_voice_detected: bool = False
    diagnosis_leakage_detected: bool = False
    issues: list[str] = Field(default_factory=list)
    validator_notes: str = ""


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    voice_score: float = Field(ge=0.0, le=1.0)
    leakage_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    source_fidelity_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    validator_notes: str = ""
    review_required: bool = False


@dataclass
class GenerationAudit:
    repair_count: int = 0
    failure_count: int = 0
    validation_fallbacks: int = 0
    stage_notes: list[str] = field(default_factory=list)

    def note_repair(self, stage: str) -> None:
        self.repair_count += 1
        self.stage_notes.append(f"{stage}:repair")

    def note_failure(self, stage: str) -> None:
        self.failure_count += 1
        self.stage_notes.append(f"{stage}:failure")

    def note_validation_fallback(self) -> None:
        self.validation_fallbacks += 1
        self.stage_notes.append("validator:fallback")


def _extract_query_terms(query: str) -> list[str]:
    terms = re.findall(r'"([^"]+)"', str(query or ""))
    cleaned = [term.strip() for term in terms if term.strip()]
    if cleaned:
        return cleaned[:4]
    fallback = re.split(r"\bAND\b|\bOR\b", str(query or ""), flags=re.IGNORECASE)
    return [item.strip(" ()") for item in fallback if item.strip(" ()")][:4]


def _clip_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip(" ,;:.") + "..."


def _fallback_symptoms(article: SourceArticle) -> list[str]:
    terms = _extract_query_terms(article.query)
    combined = " ".join([article.title, article.abstract]).strip()
    if combined:
        sentence = re.split(r"[.!?]\s+", combined, maxsplit=1)[0]
        sentence = _clip_text(sentence, 120)
        if sentence:
            return [sentence]
    return terms[:3]


def _build_fact_fallback(article: SourceArticle) -> ClinicalFactGraph:
    title = _clip_text(article.title or "Source-grounded clinical case", 180)
    primary_diagnosis = _clip_text(article.title or "Unspecified clinical condition", 160)
    query_terms = _extract_query_terms(article.query)
    symptoms = _fallback_symptoms(article)
    chief = _clip_text(symptoms[0] if symptoms else (query_terms[0] if query_terms else article.title), 120)
    history = _clip_text(article.abstract or article.title, 420)
    red_flags = query_terms[:2]
    diagnosis_hints = _dedupe_preserve([primary_diagnosis, *query_terms])[:4]
    should_detect = _dedupe_preserve([*query_terms, primary_diagnosis])[:4]
    return ClinicalFactGraph(
        title=title or "Source-grounded clinical case",
        age=None,
        sex="unknown",
        chief_complaint=chief or "Acute clinical complaint",
        symptom_duration="",
        symptom_course="",
        symptoms=symptoms[:4],
        associated_symptoms=[],
        risk_factors=[],
        history=history,
        medications=[],
        allergies=[],
        vitals=VitalsModel(),
        lab_results=[],
        primary_diagnosis_text=primary_diagnosis or "Unspecified clinical condition",
        diagnosis_hint_terms=diagnosis_hints,
        should_detect=should_detect,
        red_flags=red_flags,
        uncertainty_or_gaps=["Deterministic fact fallback used after structured extraction failure."],
    )


def _build_narrative_fallback(
    facts: ClinicalFactGraph,
    *,
    language: str,
    voice_style: Literal["patient_colloquial", "family_report", "triage_blend"],
) -> RenderedNarrative:
    symptom_bits = _dedupe_preserve([facts.chief_complaint, *facts.symptoms[:2], *facts.associated_symptoms[:2]])
    lead = symptom_bits[0] if symptom_bits else "my symptoms"
    extras = ", ".join(item for item in symptom_bits[1:3] if item)
    duration = str(facts.symptom_duration or "").strip()
    if language == "tr":
        if voice_style == "family_report":
            body = f"Yakınımda {lead.lower()} var."
        elif voice_style == "triage_blend":
            body = f"Bir anda {lead.lower()} başladı."
        else:
            body = f"Bende {lead.lower()} var."
        if extras:
            body += f" Bir de {extras.lower()} oldu."
        if duration:
            body += f" {duration} kadar süredir devam ediyor."
        body += " Tam tarif etmesi zor ama iyi hissettirmiyor."
    else:
        if voice_style == "family_report":
            body = f"My relative suddenly developed {lead.lower()}."
        elif voice_style == "triage_blend":
            body = f"It started suddenly with {lead.lower()}."
        else:
            body = f"I've been dealing with {lead.lower()}."
        if extras:
            body += f" I also noticed {extras.lower()}."
        if duration:
            body += f" It has been going on for {duration}."
        body += " It's hard to explain, but it feels wrong."
    return RenderedNarrative(
        chief_complaint=facts.chief_complaint or lead or "Clinical complaint",
        patient_text=_clip_text(body, 480),
        voice_style=voice_style,
        ambiguity_profile=["deterministic_fallback"],
        language="tr" if language == "tr" else "en",
    )


QUERY_LIBRARY: list[dict[str, str]] = [
    {"difficulty": "light", "specialty": "urology", "query": "\"renal colic\" AND \"case report\""},
    {"difficulty": "light", "specialty": "neurology", "query": "\"migraine aura\" AND \"case report\""},
    {"difficulty": "light", "specialty": "respiratory", "query": "\"asthma exacerbation\" AND \"case report\""},
    {"difficulty": "light", "specialty": "infectious", "query": "\"cystitis\" AND \"case report\""},
    {"difficulty": "moderate", "specialty": "general_surgery", "query": "\"acute appendicitis\" AND atypical AND \"case report\""},
    {"difficulty": "moderate", "specialty": "cardiopulmonary", "query": "\"pulmonary embolism\" AND syncope AND \"case report\""},
    {"difficulty": "moderate", "specialty": "infectious", "query": "\"pulmonary tuberculosis\" AND hemoptysis AND \"case report\""},
    {"difficulty": "moderate", "specialty": "obgyn", "query": "\"ectopic pregnancy\" AND abdominal pain AND \"case report\""},
    {"difficulty": "severe", "specialty": "infectious", "query": "\"severe falciparum malaria\" AND \"case report\""},
    {"difficulty": "severe", "specialty": "infectious", "query": "\"rabies\" AND hydrophobia AND \"case report\""},
    {"difficulty": "severe", "specialty": "infectious", "query": "\"neonatal tetanus\" AND \"case report\""},
    {"difficulty": "severe", "specialty": "neurology", "query": "\"acute ischemic stroke\" AND aphasia AND \"case report\""},
]


CASE_FACT_EXTRACTION_PROMPT = """
You are an expert clinical source extractor.

Task:
- Read the provided source text.
- Build one clinically coherent structured case representation.

Rules:
1. If the source is a real case report, extract only directly supported facts.
2. If the source is an outbreak or textbook teaching source, build one representative single-patient teaching case using only source-supported facts and note uncertainty in uncertainty_or_gaps.
3. Do not invent ICD codes.
4. If age, sex, vitals, or labs are missing, use null or empty values.
5. Keep red_flags and should_detect concise and clinically meaningful.
6. primary_diagnosis_text should reflect the final diagnosis stated or strongly implied by the source text.
7. diagnosis_hint_terms should contain 1-4 English medical terms useful for ICD-11 lookup.
8. Output valid JSON only.
"""


CASE_NARRATIVE_RENDER_PROMPT_TR = """
You are a Turkish clinical complaint writer.

Task:
- Turn normalized clinical facts into a natural Turkish patient complaint.
- The complaint must sound like a real person speaking, not a doctor.

Rules:
1. Output Turkish only.
2. Main voice should be the patient's own words unless the patient is a newborn, unconscious, seizing, intubated, or otherwise unable to speak. In those cases use family_report or triage_blend naturally.
3. Never write in clinician style like:
   - "Patient is a 45-year-old male"
   - "presenting to the ED"
   - "brought by EMS"
4. Include 1-3 naturally vague phrases when clinically safe.
5. Do not reveal the diagnosis or obvious spoiler disease names.
6. Keep the complaint plausible, somewhat messy, and human.
7. Keep all major source-supported symptoms and timeline coherent.
8. Output valid JSON only.
"""


CASE_NARRATIVE_RENDER_PROMPT_EN = """
You are an English clinical complaint writer.

Task:
- Turn normalized clinical facts into a natural patient-style complaint.
- The complaint must sound like a real person or worried relative, not a chart note.

Rules:
1. Output English only.
2. Main voice should be the patient's own words unless the patient is a newborn, unconscious, seizing, intubated, or otherwise unable to speak. In those cases use family_report or triage_blend naturally.
3. Never write in clinician style like:
   - "Patient is a 45-year-old male"
   - "presenting to the ED"
   - "brought by EMS"
4. Include 1-3 naturally vague phrases when clinically safe.
5. Do not reveal the diagnosis or obvious spoiler disease names.
6. Keep the complaint plausible, a bit messy, and human.
7. Keep all major source-supported symptoms and timeline coherent.
8. Output valid JSON only.
"""


CASE_NARRATIVE_VALIDATOR_PROMPT = """
You are a clinical narrative quality reviewer.

Score the generated complaint for product quality, not diagnostic correctness.

Rules:
1. Penalize clinician voice heavily.
2. Penalize diagnosis leakage heavily.
3. Penalize if core symptoms, timeline, or risk factors are lost.
4. Penalize if the complaint sounds too polished, robotic, or textbook-like.
5. Favor narratives that are natural, imperfect, and still clinically useful.
6. Output valid JSON only.
"""


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _persist_last_batch_summary(summary: dict[str, Any]) -> None:
    _ensure_case_dirs()
    LAST_BATCH_SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _list_case_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def _normalize_text_for_hash(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return cleaned[:1000]


def _normalized_external_ids(article: SourceArticle) -> dict[str, str]:
    return {
        "pmid": article.pmid,
        "pmcid": article.pmcid,
        "doi": article.doi,
        "who_don_id": article.who_don_id,
        "book_id": article.book_id,
        "chapter_id": article.chapter_id,
    }


def _stable_source_hash(article: SourceArticle) -> str:
    payload = "|".join(
        [
            article.source_origin,
            article.identifier,
            article.title,
            article.pub_date,
            _normalize_text_for_hash(article.abstract),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _article_identifier(article: SourceArticle) -> str:
    for value in (article.pmid, article.pmcid, article.doi, article.who_don_id, article.chapter_id, article.identifier):
        value = str(value or "").strip()
        if value:
            return _slugify(value)
    return _stable_source_hash(article)


def _build_case_id(article: SourceArticle, request: CaseGenerationRequest) -> str:
    return f"DRAFT-{article.source_origin.upper()}-{_article_identifier(article).upper()}-{request.language.upper()}-{_stable_source_hash(article)}"


def _build_query_candidates(request: CaseGenerationRequest, rng: random.Random) -> list[dict[str, str]]:
    candidates = [item for item in QUERY_LIBRARY if item["difficulty"] == request.difficulty]
    if request.specialty:
        specialty = request.specialty.strip().lower()
        specialty_hits = [item for item in candidates if item["specialty"] == specialty]
        if specialty_hits:
            candidates = specialty_hits
    rng.shuffle(candidates)
    fallback = [item for item in QUERY_LIBRARY if item["difficulty"] != request.difficulty]
    rng.shuffle(fallback)
    return candidates + fallback


def _article_dedupe_keys(article: SourceArticle) -> list[str]:
    keys: list[str] = []
    if article.pmid:
        keys.append(f"pmid:{article.pmid.lower()}")
    if article.pmcid:
        keys.append(f"pmcid:{article.pmcid.lower()}")
    if article.doi:
        keys.append(f"doi:{article.doi.lower()}")
    if article.who_don_id:
        keys.append(f"who:{article.who_don_id.lower()}")
    if article.book_id or article.chapter_id:
        keys.append(f"book:{(article.book_id or '')}:{(article.chapter_id or '')}".lower())
    title_year = f"{_normalize_text_for_hash(article.title)}|{article.pub_date[:4]}"
    keys.append(f"title_year:{hashlib.sha1(title_year.encode('utf-8')).hexdigest()[:16]}")
    abstract_hash = _normalize_text_for_hash(article.abstract)
    if abstract_hash:
        keys.append(f"text:{hashlib.sha1(abstract_hash.encode('utf-8')).hexdigest()[:16]}")
    return list(dict.fromkeys(keys))


def _collect_existing_dedupe_keys() -> set[str]:
    keys: set[str] = set()
    roots = [DRAFTS_DIR, CURATED_DIR, REJECTED_DIR, LEGACY_TEST_CASES_DIR]
    for root in roots:
        for path in _list_case_paths(root):
            data = _read_json_file(path)
            if not data:
                continue
            bundle = data.get("source_bundle", {}) or {}
            for key in bundle.get("dedupe_keys", []) or []:
                if key:
                    keys.add(str(key))
            for pmid in bundle.get("pubmed_pmids", []) or []:
                if pmid:
                    keys.add(f"pmid:{str(pmid).lower()}")
            external_ids = bundle.get("external_ids", {}) or {}
            for prefix, field_name in (
                ("pmid", "pmid"),
                ("pmcid", "pmcid"),
                ("doi", "doi"),
                ("who", "who_don_id"),
            ):
                value = str(external_ids.get(field_name, "") or "").strip().lower()
                if value:
                    keys.add(f"{prefix}:{value}")
    return keys


def _collect_used_pubmed_pmids() -> set[str]:
    keys = _collect_existing_dedupe_keys()
    return {key.split(":", 1)[1] for key in keys if key.startswith("pmid:")}


def _load_reference_texts() -> list[str]:
    texts: list[str] = []
    roots = [DRAFTS_DIR, CURATED_DIR, LEGACY_TEST_CASES_DIR]
    for root in roots:
        for path in _list_case_paths(root):
            data = _read_json_file(path)
            if not data:
                continue
            narrative = data.get("patient_narrative", {}) or {}
            patient_text = str(narrative.get("text") or "")
            if not patient_text:
                patient_text = str((data.get("patient_data", {}) or {}).get("patient_text") or "")
            title = str(data.get("title", "") or "")
            if patient_text or title:
                texts.append(f"{title}\n{patient_text}".strip())
    return texts


def _tokenize_for_similarity(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return {token for token in text.split() if len(token) >= 4}


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize_for_similarity(a)
    tokens_b = _tokenize_for_similarity(b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return overlap / union if union else 0.0


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _extract_html_title(html: str) -> str:
    match = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if match:
        return _html_to_text(match.group(1))
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if match:
        return _html_to_text(match.group(1))
    return "Untitled source"


def _extract_html_paragraphs(html: str, *, limit: int = 16) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.findall(r"(?is)<p[^>]*>(.*?)</p>", html):
        text = _html_to_text(raw)
        if len(text) < 40:
            continue
        if any(marker in text.lower() for marker in ("cookie", "privacy", "copyright", "subscribe")):
            continue
        paragraphs.append(text)
        if len(paragraphs) >= limit:
            break
    return paragraphs


async def _fetch_text(url: str, *, timeout: float = 30.0) -> str:
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "RRRIE-CDSS/1.0"}) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _is_viable_candidate(article: SourceArticle) -> bool:
    if article.rights_status != "approved":
        return False
    text = str(article.abstract or "").strip()
    if article.source_origin in {"pubmed", "europe_pmc"}:
        return len(text) >= 120 and bool(article.identifier)
    if article.source_origin == "who_don":
        return len(text) >= 500
    return len(text) >= 300


async def _harvest_pubmed_candidates(
    request: CaseGenerationRequest,
    rng: random.Random,
    max_candidates: int,
    used_keys: set[str],
) -> list[SourceArticle]:
    candidates: list[SourceArticle] = []
    seen: set[str] = set()
    for query_info in _build_query_candidates(request, rng):
        result = await search_pubmed(query=query_info["query"], max_results=8)
        for article in result.get("articles", []) or []:
            pmid = str(article.get("pmid", "") or "").strip()
            abstract = str(article.get("abstract", "") or "").strip()
            if not pmid or not abstract:
                continue
            candidate = SourceArticle(
                identifier=pmid,
                title=str(article.get("title", "") or "Untitled case"),
                abstract=abstract,
                journal=str(article.get("journal", "") or ""),
                pub_date=str(article.get("pub_date", "") or ""),
                url=str(article.get("url", "") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                query=query_info["query"],
                specialty=query_info["specialty"],
                difficulty=query_info["difficulty"],
                source_origin="pubmed",
                case_kind="real_report",
                trust_tier="high",
                auto_approve_eligible=True,
                extraction_method="pubmed_eutils",
                rights_status="approved",
                license_name="PubMed abstract metadata",
                license_url="https://www.ncbi.nlm.nih.gov/home/tools/",
                pmid=pmid,
                doi=str(article.get("doi", "") or ""),
            )
            dedupe_keys = _article_dedupe_keys(candidate)
            if any(key in used_keys or key in seen for key in dedupe_keys):
                continue
            if not _is_viable_candidate(candidate):
                continue
            candidates.append(candidate)
            seen.update(dedupe_keys)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


async def _harvest_europe_pmc_candidates(
    request: CaseGenerationRequest,
    rng: random.Random,
    max_candidates: int,
    used_keys: set[str],
) -> list[SourceArticle]:
    candidates: list[SourceArticle] = []
    seen: set[str] = set()
    for query_info in _build_query_candidates(request, rng):
        for source in ("MED", "PMC"):
            result = await search_europe_pmc(query=query_info["query"], max_results=8, source=source)
            for article in result.get("articles", []) or []:
                pmid = str(article.get("pmid", "") or "").strip()
                doi = str(article.get("doi", "") or "").strip()
                title = str(article.get("title", "") or "Untitled case")
                identifier = pmid or doi or _slugify(f"{title}-{article.get('pub_date', '')}")
                candidate = SourceArticle(
                    identifier=identifier,
                    title=title,
                    abstract=str(article.get("abstract", "") or "").strip(),
                    journal=str(article.get("journal", "") or ""),
                    pub_date=str(article.get("pub_date", "") or ""),
                    url=str(article.get("url", "") or ""),
                    query=query_info["query"],
                    specialty=query_info["specialty"],
                    difficulty=query_info["difficulty"],
                    source_origin="europe_pmc",
                    case_kind="real_report",
                    trust_tier="high",
                    auto_approve_eligible=True,
                    extraction_method=f"europe_pmc_{source.lower()}",
                    rights_status="approved",
                    license_name="Europe PMC abstract metadata",
                    license_url="https://dev.europepmc.org/RestfulWebService",
                    pmid=pmid,
                    doi=doi,
                    pmcid=str(article.get("pmcid", "") or ""),
                )
                dedupe_keys = _article_dedupe_keys(candidate)
                if any(key in used_keys or key in seen for key in dedupe_keys):
                    continue
                if not _is_viable_candidate(candidate):
                    continue
                candidates.append(candidate)
                seen.update(dedupe_keys)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _who_case_signal(text: str) -> bool:
    lowered = text.lower()
    markers = ("patient", "patients", "year-old", "presented", "symptoms", "fever", "pain", "hospital", "admitted")
    return sum(1 for marker in markers if marker in lowered) >= 2


async def _harvest_who_candidates(
    request: CaseGenerationRequest,
    rng: random.Random,
    max_candidates: int,
    used_keys: set[str],
) -> list[SourceArticle]:
    candidates: list[SourceArticle] = []
    seen: set[str] = set()
    try:
        index_html = await _fetch_text(WHO_DON_INDEX_URL)
    except Exception as exc:
        logger.warning("[CASE GENERATOR] WHO DON index fetch failed: %s", exc)
        return candidates

    links = re.findall(r'href="([^"]*?/emergencies/disease-outbreak-news/item/[^"]+)"', index_html)
    normalized_links: list[str] = []
    for href in links:
        url = urljoin("https://www.who.int", href)
        if url not in normalized_links:
            normalized_links.append(url)
    rng.shuffle(normalized_links)

    for url in normalized_links[: max_candidates * 3]:
        try:
            html = await _fetch_text(url)
        except Exception as exc:
            logger.debug("[CASE GENERATOR] WHO DON item fetch failed %s: %s", url, exc)
            continue
        paragraphs = _extract_html_paragraphs(html, limit=18)
        body = "\n".join(paragraphs)
        if len(body) < 500 or not _who_case_signal(body):
            continue
        who_don_id = url.rstrip("/").split("/")[-1]
        title = _extract_html_title(html)
        candidate = SourceArticle(
            identifier=who_don_id,
            title=title,
            abstract=body,
            journal="WHO Disease Outbreak News",
            pub_date="",
            url=url,
            query=f"who_don:{request.difficulty}",
            specialty=request.specialty or "public_health",
            difficulty=request.difficulty,
            source_origin="who_don",
            case_kind="outbreak_teaching",
            trust_tier="public_health",
            auto_approve_eligible=False,
            extraction_method="who_don_html",
            rights_status="approved",
            license_name="WHO public web publication",
            license_url=WHO_DON_INDEX_URL,
            who_don_id=who_don_id,
        )
        dedupe_keys = _article_dedupe_keys(candidate)
        if any(key in used_keys or key in seen for key in dedupe_keys):
            continue
        candidates.append(candidate)
        seen.update(dedupe_keys)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _extract_textbook_passages(raw_text: str, *, limit: int = 4) -> list[str]:
    chunks = re.split(r"\n\s*\n+", raw_text)
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk).strip()
        if len(text) < 180:
            continue
        lowered = text.lower()
        score = 0
        for marker in ("patient", "presented", "history", "complaint", "symptom", "pain", "fever", "case"):
            if marker in lowered:
                score += 1
        if text.startswith("#"):
            score += 1
        scored.append((score, text))
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    return [text for _, text in scored[:limit]]


def _load_book_manifest_entries() -> list[dict[str, Any]]:
    if not APPROVED_BOOK_MANIFEST.exists():
        return []
    data = _read_json_file(APPROVED_BOOK_MANIFEST)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("sources"), list):
        return [item for item in data["sources"] if isinstance(item, dict)]
    return []


def _rights_ok(entry: dict[str, Any]) -> bool:
    status = str(entry.get("rights_status", "approved") or "approved").strip().lower()
    return status == "approved"


async def _harvest_approved_book_candidates(
    request: CaseGenerationRequest,
    rng: random.Random,
    max_candidates: int,
    used_keys: set[str],
) -> list[SourceArticle]:
    candidates: list[SourceArticle] = []
    seen: set[str] = set()

    if TEXTBOOK_DIR.exists():
        for path in sorted(TEXTBOOK_DIR.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            for idx, passage in enumerate(_extract_textbook_passages(raw_text)):
                candidate = SourceArticle(
                    identifier=f"{path.stem}-{idx}",
                    title=f"{path.stem.replace('_', ' ').title()} Teaching Case",
                    abstract=passage,
                    journal=path.name,
                    url="",
                    query=path.name,
                    specialty=request.specialty or "general",
                    difficulty=request.difficulty,
                    source_origin="approved_book",
                    case_kind="textbook_teaching",
                    trust_tier="teaching",
                    auto_approve_eligible=False,
                    extraction_method="local_textbook_file",
                    rights_status="approved",
                    license_name="Local approved content",
                    license_url="",
                    book_id=path.stem,
                    chapter_id=str(idx),
                )
                dedupe_keys = _article_dedupe_keys(candidate)
                if any(key in used_keys or key in seen for key in dedupe_keys):
                    continue
                if not _is_viable_candidate(candidate):
                    continue
                candidates.append(candidate)
                seen.update(dedupe_keys)
                if len(candidates) >= max_candidates:
                    return candidates

    for entry in _load_book_manifest_entries():
        if len(candidates) >= max_candidates:
            break
        url = str(entry.get("url", "") or "").strip()
        if not url or not _rights_ok(entry):
            continue
        host = urlparse(url).netloc.lower()
        if host not in APPROVED_BOOK_HOST_ALLOWLIST:
            continue
        try:
            html = await _fetch_text(url)
        except Exception as exc:
            logger.debug("[CASE GENERATOR] Approved book fetch failed %s: %s", url, exc)
            continue
        text = _html_to_text(html)
        passages = _extract_textbook_passages(text, limit=2)
        for idx, passage in enumerate(passages):
            candidate = SourceArticle(
                identifier=str(entry.get("id") or f"{host}-{idx}"),
                title=str(entry.get("title") or _extract_html_title(html) or "Approved teaching source"),
                abstract=passage,
                journal=str(entry.get("source_name") or host),
                url=url,
                query=str(entry.get("title") or host),
                specialty=request.specialty or "general",
                difficulty=request.difficulty,
                source_origin="approved_book",
                case_kind="textbook_teaching",
                trust_tier="teaching",
                auto_approve_eligible=False,
                extraction_method="approved_manifest_url",
                rights_status="approved",
                license_name=str(entry.get("license_name") or "Approved open/public source"),
                license_url=str(entry.get("license_url") or url),
                book_id=str(entry.get("book_id") or entry.get("id") or host),
                chapter_id=str(entry.get("chapter_id") or idx),
            )
            dedupe_keys = _article_dedupe_keys(candidate)
            if any(key in used_keys or key in seen for key in dedupe_keys):
                continue
            if not _is_viable_candidate(candidate):
                continue
            candidates.append(candidate)
            seen.update(dedupe_keys)
            if len(candidates) >= max_candidates:
                return candidates

    rng.shuffle(candidates)
    return candidates[:max_candidates]


async def _harvest_candidates(
    request: CaseGenerationRequest,
    rng: random.Random,
    used_keys: set[str],
) -> tuple[list[SourceArticle], dict[str, int]]:
    max_candidates = max(request.max_candidates, request.target_count)
    source_pool = list(request.sources)
    if request.source_mode == "pubmed_primary":
        source_pool = ["pubmed"]
    if not source_pool:
        source_pool = list(DEFAULT_BATCH_SOURCES)

    per_source_limit = max(3, max_candidates // max(1, len(source_pool)))
    all_candidates: list[SourceArticle] = []
    harvested_counts = {source: 0 for source in SOURCE_REGISTRY}
    seen_dedupe_keys: set[str] = set()

    for source in source_pool:
        if source == "pubmed":
            harvested = await _harvest_pubmed_candidates(request, rng, per_source_limit, used_keys | seen_dedupe_keys)
        elif source == "europe_pmc":
            harvested = await _harvest_europe_pmc_candidates(request, rng, per_source_limit, used_keys | seen_dedupe_keys)
        elif source == "who_don":
            harvested = await _harvest_who_candidates(request, rng, per_source_limit, used_keys | seen_dedupe_keys)
        elif source == "approved_book":
            harvested = await _harvest_approved_book_candidates(request, rng, per_source_limit, used_keys | seen_dedupe_keys)
        else:
            harvested = []
        harvested_counts[source] = len(harvested)
        for article in harvested:
            dedupe_keys = _article_dedupe_keys(article)
            if any(key in seen_dedupe_keys for key in dedupe_keys):
                continue
            all_candidates.append(article)
            seen_dedupe_keys.update(dedupe_keys)

    rng.shuffle(all_candidates)
    return all_candidates[:max_candidates], harvested_counts


async def _generate_structured_model(
    gemini_client: GeminiClient,
    model_cls: type[BaseModel],
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    use_pro: bool = False,
    context_defaults: dict[str, Any] | None = None,
    audit: GenerationAudit | None = None,
    stage_label: str = "structured",
) -> BaseModel:
    last_raw = ""
    errors: list[str] = []
    attempt_messages = list(messages)
    for attempt in range(MAX_STRUCTURED_ATTEMPTS):
        response = await gemini_client.chat_complete(
            messages=attempt_messages,
            max_tokens=max_tokens,
            temperature=temperature if attempt == 0 else 0.0,
            use_pro=use_pro,
            json_schema=model_cls,
        )
        last_raw = response.get("content", "")
        normalized = sanitize_json_text(last_raw)
        try:
            return model_cls.model_validate_json(normalized)
        except ValidationError as exc:
            errors.append(str(exc))
            salvaged = parse_json_from_response(normalized, strict=False)
            if salvaged:
                if audit is not None:
                    audit.note_repair(stage_label)
                salvaged = _repair_partial_structured_payload(model_cls, salvaged, context_defaults=context_defaults)
                try:
                    return model_cls.model_validate(salvaged)
                except ValidationError as repaired_exc:
                    errors.append(str(repaired_exc))
            if attempt < MAX_STRUCTURED_ATTEMPTS - 1:
                attempt_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "Your previous output did not match the schema exactly. "
                            "Return one valid JSON object only. Do not include commentary, markdown, or truncated strings. "
                            "Every required field must be present. Keep values short and complete. "
                            f"Validation errors: {' | '.join(errors[-2:])[:600]}"
                        ),
                    }
                ]

    if audit is not None:
        audit.note_failure(stage_label)
    raise RuntimeError(
        "Structured generation failed after retry. "
        f"Last raw output: {last_raw[:400]}. Errors: {' | '.join(errors[:3])}"
    )


def _repair_partial_structured_payload(
    model_cls: type[BaseModel],
    payload: dict[str, Any],
    *,
    context_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repaired = dict(payload)
    defaults = context_defaults or {}

    if model_cls is ClinicalFactGraph:
        title = str(repaired.get("title") or defaults.get("title") or "Source-grounded clinical case").strip()
        chief = str(
            repaired.get("chief_complaint")
            or defaults.get("chief_complaint")
            or repaired.get("primary_diagnosis_text")
            or title
        ).strip()
        diagnosis = str(
            repaired.get("primary_diagnosis_text")
            or defaults.get("primary_diagnosis_text")
            or title
        ).strip()
        repaired.setdefault("title", title)
        repaired.setdefault("sex", "unknown")
        repaired.setdefault("chief_complaint", chief)
        repaired.setdefault("symptom_duration", "")
        repaired.setdefault("symptom_course", "")
        repaired.setdefault("symptoms", _dedupe_preserve([chief, *(repaired.get("symptoms", []) or [])])[:4])
        repaired.setdefault("associated_symptoms", [])
        repaired.setdefault("risk_factors", [])
        repaired.setdefault("history", str(repaired.get("history") or defaults.get("history") or "")[:420])
        repaired.setdefault("medications", [])
        repaired.setdefault("allergies", [])
        repaired.setdefault("vitals", {})
        repaired.setdefault("lab_results", [])
        repaired.setdefault("primary_diagnosis_text", diagnosis)
        repaired.setdefault(
            "diagnosis_hint_terms",
            _dedupe_preserve(
                [
                    diagnosis,
                    *list(repaired.get("diagnosis_hint_terms", []) or []),
                    *list(defaults.get("diagnosis_hint_terms", []) or []),
                ]
            )[:4],
        )
        repaired.setdefault(
            "should_detect",
            _dedupe_preserve(
                [
                    chief,
                    diagnosis,
                    *list(repaired.get("should_detect", []) or []),
                ]
            )[:4],
        )
        repaired.setdefault("red_flags", list(repaired.get("red_flags", []) or []))
        repaired.setdefault("uncertainty_or_gaps", list(repaired.get("uncertainty_or_gaps", []) or []))

    if model_cls is RenderedNarrative:
        if not repaired.get("chief_complaint") and defaults.get("chief_complaint"):
            repaired["chief_complaint"] = defaults["chief_complaint"]
        if not repaired.get("voice_style") and defaults.get("voice_style"):
            repaired["voice_style"] = defaults["voice_style"]
        if not repaired.get("language") and defaults.get("language"):
            repaired["language"] = defaults["language"]
        if not repaired.get("ambiguity_profile"):
            repaired["ambiguity_profile"] = []
        patient_text = str(repaired.get("patient_text") or "").strip()
        if patient_text:
            repaired["patient_text"] = _clip_text(patient_text, 480)

    return repaired


def _effective_voice_style(
    requested_style: str,
    facts: ClinicalFactGraph,
) -> Literal["patient_colloquial", "family_report", "triage_blend"]:
    age = facts.age if facts.age is not None else 99
    text_blob = " ".join(
        [facts.chief_complaint, facts.history, " ".join(facts.symptoms), " ".join(facts.associated_symptoms)]
    ).lower()
    if age <= 8 or any(
        marker in text_blob
        for marker in (
            "unresponsive",
            "altered mental",
            "seizure",
            "confused",
            "coma",
            "newborn",
            "neonate",
            "unable to speak",
        )
    ):
        return "triage_blend" if requested_style == "triage_blend" else "family_report"
    return requested_style  # type: ignore[return-value]


async def _extract_clinical_facts(
    gemini_client: GeminiClient,
    article: SourceArticle,
    *,
    audit: GenerationAudit | None = None,
) -> ClinicalFactGraph:
    user_prompt = (
        f"Source origin: {article.source_origin}\n"
        f"Case kind: {article.case_kind}\n"
        f"Identifier: {article.identifier}\n"
        f"Title: {article.title}\n"
        f"Publication: {article.journal}\n"
        f"Published: {article.pub_date}\n"
        f"Query: {article.query}\n"
        f"Canonical URL: {article.url}\n\n"
        f"Source text:\n{article.abstract}"
    )
    try:
        result = await _generate_structured_model(
            gemini_client,
            ClinicalFactGraph,
            messages=[
                {"role": "system", "content": CASE_FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
            context_defaults={
                "title": article.title,
                "chief_complaint": article.title,
                "primary_diagnosis_text": article.title,
                "history": article.abstract,
                "diagnosis_hint_terms": _extract_query_terms(article.query),
            },
            audit=audit,
            stage_label="fact_extraction",
        )
        return result  # type: ignore[return-value]
    except Exception:
        if audit is not None:
            audit.note_validation_fallback()
        return _build_fact_fallback(article)


async def _resolve_icd_validation(facts: ClinicalFactGraph) -> dict[str, Any]:
    terms: list[str] = []
    if facts.primary_diagnosis_text:
        terms.append(facts.primary_diagnosis_text)
    for item in facts.diagnosis_hint_terms:
        value = str(item or "").strip()
        if value and value not in terms:
            terms.append(value)

    all_matches: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    codes: list[str] = []
    for term in terms[:4]:
        matches = await search_icd11_who(term, max_results=3)
        for match in matches:
            code = str(match.get("theCode", "") or "").strip()
            title = str(match.get("title", "") or "").strip()
            score = match.get("score", 0)
            entry = {"term": term, "code": code, "title": title, "score": score}
            all_matches.append(entry)
            if code and code not in seen_codes:
                seen_codes.add(code)
                codes.append(code)
            if len(codes) >= 3:
                break
        if len(codes) >= 3:
            break

    return {
        "primary_term": facts.primary_diagnosis_text,
        "lookup_terms": terms,
        "matches": all_matches,
        "codes": codes,
        "validated": bool(codes),
    }


def _render_prompt_for_language(language: str) -> str:
    return CASE_NARRATIVE_RENDER_PROMPT_TR if language == "tr" else CASE_NARRATIVE_RENDER_PROMPT_EN


async def _render_patient_narrative(
    gemini_client: GeminiClient,
    facts: ClinicalFactGraph,
    request: CaseGenerationRequest,
    *,
    feedback: str = "",
    audit: GenerationAudit | None = None,
) -> RenderedNarrative:
    effective_voice = _effective_voice_style(request.voice_style, facts)
    payload = {
        "language": request.language,
        "requested_voice_style": request.voice_style,
        "effective_voice_style": effective_voice,
        "difficulty": request.difficulty,
        "clinical_facts": facts.model_dump(mode="json"),
        "revision_feedback": feedback,
    }
    try:
        result = await _generate_structured_model(
            gemini_client,
            RenderedNarrative,
            messages=[
                {"role": "system", "content": _render_prompt_for_language(request.language)},
                {
                    "role": "user",
                    "content": "Generate the patient-facing complaint JSON for this case.\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            temperature=0.65,
            max_tokens=900,
            context_defaults={
                "chief_complaint": facts.chief_complaint,
                "voice_style": effective_voice,
                "language": request.language,
            },
            audit=audit,
            stage_label="narrative_render",
        )
        if result.voice_style != effective_voice:
            result.voice_style = effective_voice
        return result  # type: ignore[return-value]
    except Exception:
        if audit is not None:
            audit.note_validation_fallback()
        return _build_narrative_fallback(
            facts,
            language=request.language,
            voice_style=effective_voice,
        )


async def _validate_narrative_quality(
    gemini_client: GeminiClient,
    article: SourceArticle,
    facts: ClinicalFactGraph,
    narrative: RenderedNarrative,
    icd_validation: dict[str, Any],
    *,
    audit: GenerationAudit | None = None,
) -> NarrativeValidatorResult:
    payload = {
        "clinical_facts": facts.model_dump(mode="json"),
        "patient_narrative": narrative.model_dump(mode="json"),
        "icd_validation": icd_validation,
    }
    try:
        result = await _generate_structured_model(
            gemini_client,
            NarrativeValidatorResult,
            messages=[
                {"role": "system", "content": CASE_NARRATIVE_VALIDATOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.0,
            max_tokens=700,
            audit=audit,
            stage_label="narrative_validation",
        )
        return result  # type: ignore[return-value]
    except Exception as exc:
        logger.info("[CASE GENERATOR] Narrative validator fallback activated")
        if audit is not None:
            audit.note_validation_fallback()
        voice_score = _deterministic_voice_score(narrative.patient_text, narrative.language)
        leakage_score = _deterministic_leakage_score(facts.primary_diagnosis_text, narrative.patient_text)
        consistency_score = _deterministic_consistency_score(facts, narrative)
        source_fidelity_score = _deterministic_source_fidelity_score(article, facts)
        issues: list[str] = []
        if voice_score < 0.68:
            issues.append("Clinician voice detected")
        if leakage_score < 0.68:
            issues.append("Diagnosis leakage detected")
        if consistency_score < 0.68:
            issues.append("Narrative/facts consistency below target")
        if source_fidelity_score < 0.68:
            issues.append("Source fidelity below target")
        return NarrativeValidatorResult(
            voice_score=voice_score,
            leakage_score=leakage_score,
            consistency_score=consistency_score,
            source_fidelity_score=source_fidelity_score,
            clinician_voice_detected=voice_score < 0.68,
            diagnosis_leakage_detected=leakage_score < 0.68,
            issues=issues,
            validator_notes=f"Deterministic fallback used after validator failure: {str(exc)[:220]}",
        )


def _deterministic_leakage_score(primary_diagnosis: str, narrative_text: str) -> float:
    diagnosis = str(primary_diagnosis or "").lower()
    narrative = str(narrative_text or "").lower()
    if not diagnosis or not narrative:
        return 0.2
    normalized_dx = re.sub(r"[^a-z0-9\s]+", " ", diagnosis)
    dx_tokens = [token for token in normalized_dx.split() if len(token) >= 4]
    if diagnosis in narrative:
        return 0.0
    if any(token in narrative for token in dx_tokens):
        return 0.2
    return 1.0


def _deterministic_voice_score(narrative_text: str, language: str) -> float:
    text = str(narrative_text or "").lower()
    if not text:
        return 0.0
    clinician_markers = (
        "patient is a",
        "presenting to",
        "brought to the ed",
        "brought by ems",
        "currently post-ictal",
        "male presenting",
        "female presenting",
        "hasta ",
        "acile getirildi",
        "ed'e getirildi",
    )
    patient_markers = ("i ", "i'm", "it feels", "kind of", "my ", "hocam", "gibi", "sanki", "bir sey", "tarif", "midem")
    score = 0.65
    if any(marker in text for marker in clinician_markers):
        score -= 0.45
    if any(marker in text for marker in patient_markers):
        score += 0.25
    if language == "tr" and any(marker in text for marker in ("patient is a", "presenting to")):
        score -= 0.2
    return max(0.0, min(1.0, score))


def _deterministic_consistency_score(facts: ClinicalFactGraph, narrative: RenderedNarrative) -> float:
    narrative_text = str(narrative.patient_text or "").lower()
    if not narrative_text:
        return 0.0
    if narrative.language == "tr":
        score = 0.62
        if len(narrative_text.split()) >= 12:
            score += 0.1
        if facts.symptom_duration and any(marker in narrative_text for marker in ("gun", "gundur", "hafta", "ay", "dun", "bugun", "since", "for")):
            score += 0.08
        if len(facts.symptoms) + len(facts.associated_symptoms) >= 2 and any(marker in narrative_text for marker in (",", " ama ", " bir de ", " sanki ", " gibi ")):
            score += 0.08
        if facts.chief_complaint and len(narrative_text) > max(20, len(facts.chief_complaint) // 2):
            score += 0.08
        if facts.red_flags and any(marker in narrative_text for marker in ("bayil", "nefes", "ates", "titreme", "kan", "kus", "uyus", "konus")):
            score += 0.06
        return round(min(score, 1.0), 2)

    signals = []
    for item in [facts.chief_complaint, *facts.symptoms[:4], *facts.associated_symptoms[:3]]:
        cleaned = str(item or "").strip().lower()
        if not cleaned:
            continue
        tokens = [token for token in re.findall(r"[a-zA-Z]{4,}", cleaned)[:2] if token]
        if not tokens:
            continue
        signals.append(any(token.lower() in narrative_text for token in tokens))
    if not signals:
        return 0.45
    return round(sum(1 for matched in signals if matched) / len(signals), 2)


def _deterministic_source_fidelity_score(article: SourceArticle, facts: ClinicalFactGraph) -> float:
    score = 0.35
    if article.abstract:
        score += 0.2
    if facts.primary_diagnosis_text:
        score += 0.2
    if facts.symptoms:
        score += 0.15
    if facts.history:
        score += 0.05
    if facts.red_flags:
        score += 0.05
    return max(0.0, min(1.0, score))


def _estimate_novelty_score(
    article: SourceArticle,
    facts: ClinicalFactGraph,
    narrative: RenderedNarrative,
    used_pmids: set[str],
) -> float:
    if article.pmid and article.pmid in used_pmids:
        return 0.0
    candidate = f"{facts.title}\n{narrative.patient_text}"
    comparisons = _load_reference_texts()[-20:]
    if not comparisons:
        return 1.0
    max_similarity = max((_jaccard_similarity(candidate, ref) for ref in comparisons), default=0.0)
    return max(0.0, min(1.0, round(1.0 - max_similarity, 2)))


def _merge_quality_report(
    validator: NarrativeValidatorResult,
    *,
    narrative: RenderedNarrative,
    facts: ClinicalFactGraph,
    article: SourceArticle,
    used_pmids: set[str],
) -> QualityReport:
    voice_score = min(validator.voice_score, _deterministic_voice_score(narrative.patient_text, narrative.language))
    leakage_score = min(validator.leakage_score, _deterministic_leakage_score(facts.primary_diagnosis_text, narrative.patient_text))
    consistency_score = min(validator.consistency_score, _deterministic_consistency_score(facts, narrative))
    source_fidelity_score = min(validator.source_fidelity_score, _deterministic_source_fidelity_score(article, facts))
    novelty_score = _estimate_novelty_score(article, facts, narrative, used_pmids)

    issues = list(dict.fromkeys([*(validator.issues or [])]))
    if validator.clinician_voice_detected and "Clinician voice detected" not in issues:
        issues.append("Clinician voice detected")
    if validator.diagnosis_leakage_detected and "Diagnosis leakage detected" not in issues:
        issues.append("Diagnosis leakage detected")
    if novelty_score < 0.45 and "Narrative too similar to recent cases" not in issues:
        issues.append("Narrative too similar to recent cases")

    review_required = any(score < 0.68 for score in (voice_score, leakage_score, consistency_score, source_fidelity_score)) or novelty_score < 0.45

    return QualityReport(
        voice_score=round(voice_score, 2),
        leakage_score=round(leakage_score, 2),
        consistency_score=round(consistency_score, 2),
        novelty_score=round(novelty_score, 2),
        source_fidelity_score=round(source_fidelity_score, 2),
        issues=issues,
        validator_notes=validator.validator_notes,
        review_required=review_required,
    )


async def _render_validated_narrative(
    gemini_client: GeminiClient,
    article: SourceArticle,
    facts: ClinicalFactGraph,
    request: CaseGenerationRequest,
    icd_validation: dict[str, Any],
    used_pmids: set[str],
    *,
    audit: GenerationAudit | None = None,
) -> tuple[RenderedNarrative, QualityReport]:
    narrative = await _render_patient_narrative(gemini_client, facts, request, audit=audit)
    validator = await _validate_narrative_quality(gemini_client, article, facts, narrative, icd_validation, audit=audit)
    quality = _merge_quality_report(validator, narrative=narrative, facts=facts, article=article, used_pmids=used_pmids)
    if quality.review_required:
        feedback = "Fix these issues while keeping the same clinical facts: " + "; ".join(quality.issues[:4])
        narrative = await _render_patient_narrative(gemini_client, facts, request, feedback=feedback, audit=audit)
        validator = await _validate_narrative_quality(gemini_client, article, facts, narrative, icd_validation, audit=audit)
        quality = _merge_quality_report(validator, narrative=narrative, facts=facts, article=article, used_pmids=used_pmids)
    return narrative, quality


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _build_decision_meta(
    article: SourceArticle,
    quality: QualityReport,
    icd_validation: dict[str, Any],
    audit: GenerationAudit,
) -> dict[str, Any]:
    reasons: list[str] = []
    scores = {
        "voice_score": quality.voice_score,
        "leakage_score": quality.leakage_score,
        "consistency_score": quality.consistency_score,
        "source_fidelity_score": quality.source_fidelity_score,
        "novelty_score": quality.novelty_score,
        "repair_count": audit.repair_count,
        "failure_count": audit.failure_count,
        "validation_fallbacks": audit.validation_fallbacks,
        "who_validated": 1.0 if bool(icd_validation.get("validated")) else 0.0,
    }

    if article.rights_status != "approved":
        reasons.append("Provenance or license policy failed")
        decision = "auto_rejected"
    elif not article.abstract or len(article.abstract.strip()) < 120:
        reasons.append("Source text too thin for safe case generation")
        decision = "auto_rejected"
    elif quality.leakage_score < 0.60:
        reasons.append("Diagnosis leakage below rejection threshold")
        decision = "auto_rejected"
    elif quality.consistency_score < 0.60:
        reasons.append("Narrative consistency below rejection threshold")
        decision = "auto_rejected"
    elif quality.source_fidelity_score < 0.60:
        reasons.append("Source fidelity below rejection threshold")
        decision = "auto_rejected"
    elif quality.novelty_score < 0.35:
        reasons.append("Near-duplicate of existing cases")
        decision = "auto_rejected"
    elif audit.failure_count > 1:
        reasons.append("Repeated structured generation failures")
        decision = "auto_rejected"
    elif article.source_origin in {"who_don", "approved_book"}:
        reasons.append("Source policy routes outbreak/textbook cases to review in v1")
        decision = "needs_review"
    elif (
        article.auto_approve_eligible
        and bool(icd_validation.get("validated"))
        and quality.voice_score >= 0.85
        and quality.leakage_score >= 0.90
        and quality.consistency_score >= 0.85
        and quality.source_fidelity_score >= 0.85
        and quality.novelty_score >= 0.70
        and not quality.issues
        and audit.repair_count <= 1
        and audit.failure_count == 0
        and audit.validation_fallbacks == 0
    ):
        reasons.append("All conservative auto-approve thresholds passed")
        decision = "auto_approved"
    else:
        reasons.append("Requires manual review under conservative policy")
        if quality.issues:
            reasons.extend(quality.issues[:3])
        decision = "needs_review"

    return {
        "decision": decision,
        "decision_reasons": _dedupe_preserve(reasons),
        "decision_scores": scores,
        "decision_version": DECISION_VERSION,
        "stage_notes": audit.stage_notes,
    }


def _decision_to_status(decision: str) -> Literal["draft", "curated", "rejected"]:
    if decision == "auto_approved":
        return "curated"
    if decision == "auto_rejected":
        return "rejected"
    return "draft"


def _build_case_payload(
    *,
    request: CaseGenerationRequest,
    article: SourceArticle,
    facts: ClinicalFactGraph,
    narrative: RenderedNarrative,
    icd_validation: dict[str, Any],
    quality: QualityReport,
    decision_meta: dict[str, Any],
    status: Literal["draft", "curated", "rejected"],
) -> dict[str, Any]:
    source_hash = _stable_source_hash(article)
    dedupe_keys = _article_dedupe_keys(article)
    case_id = _build_case_id(article, request)
    symptom_list = _dedupe_preserve([*facts.symptoms, *facts.associated_symptoms])
    patient_narrative = narrative.model_dump(mode="json")
    patient_narrative["text"] = narrative.patient_text
    patient_data = {
        "age": facts.age,
        "sex": facts.sex,
        "chief_complaint": narrative.chief_complaint or facts.chief_complaint,
        "symptoms": symptom_list,
        "vitals": facts.vitals.model_dump(exclude_none=True),
        "history": facts.history,
        "medications": facts.medications,
        "allergies": facts.allergies,
        "lab_results": facts.lab_results,
        "patient_text": narrative.patient_text,
    }
    source_bundle = {
        "canonical_url": article.url,
        "external_ids": _normalized_external_ids(article),
        "rights": {
            "status": article.rights_status,
            "license_name": article.license_name,
            "license_url": article.license_url,
        },
        "fetch_timestamp": _utc_now_iso(),
        "fetched_at": _utc_now_iso(),
        "extraction_method": article.extraction_method,
        "source_trust_tier": article.trust_tier,
        "source_origin": article.source_origin,
        "case_kind": article.case_kind,
        "dedupe_keys": dedupe_keys,
        "medical_area": article.specialty,
        "pubmed_pmids": [article.pmid] if article.pmid else [],
        "pubmed_queries": [article.query] if article.query else [],
        "source_article": {
            "identifier": article.identifier,
            "title": article.title,
            "journal": article.journal,
            "pub_date": article.pub_date,
            "url": article.url,
            "abstract": article.abstract,
            "query": article.query,
        },
        "who_validation": icd_validation,
        "source_hash": source_hash,
    }
    quality.review_required = decision_meta.get("decision") == "needs_review"
    return {
        "case_id": case_id,
        "title": facts.title,
        "source_type": "generated",
        "source_origin": article.source_origin,
        "case_kind": article.case_kind,
        "synthetic": article.case_kind != "real_report",
        "case_meta": {
            "case_id": case_id,
            "status": status,
            "language": request.language,
            "difficulty": request.difficulty,
            "source_mode": request.source_mode,
            "voice_style": narrative.voice_style,
            "generator_version": GENERATOR_VERSION,
            "created_at": _utc_now_iso(),
        },
        "source_bundle": source_bundle,
        "clinical_facts": facts.model_dump(mode="json"),
        "patient_narrative": patient_narrative,
        "patient_data": patient_data,
        "expected_output": {
            "primary_diagnosis": facts.primary_diagnosis_text,
            "expected_icd11_codes": icd_validation.get("codes", []),
            "should_detect": _dedupe_preserve(facts.should_detect),
            "red_flags": _dedupe_preserve(facts.red_flags),
        },
        "quality_report": quality.model_dump(mode="json"),
        "review_required": quality.review_required,
        "decision_meta": decision_meta,
        "draft_id": case_id,
    }


def _build_failed_case_payload(
    *,
    request: CaseGenerationRequest,
    article: SourceArticle,
    error_message: str,
    audit: GenerationAudit,
) -> dict[str, Any]:
    dedupe_keys = _article_dedupe_keys(article)
    case_id = _build_case_id(article, request)
    decision_meta = {
        "decision": "auto_rejected",
        "decision_reasons": ["Repeated structured generation failure", error_message[:220]],
        "decision_scores": {
            "voice_score": 0.0,
            "leakage_score": 0.0,
            "consistency_score": 0.0,
            "source_fidelity_score": 0.0,
            "novelty_score": 0.0,
            "repair_count": audit.repair_count,
            "failure_count": max(1, audit.failure_count),
            "validation_fallbacks": audit.validation_fallbacks,
            "who_validated": 0.0,
        },
        "decision_version": DECISION_VERSION,
        "stage_notes": audit.stage_notes,
    }
    return {
        "case_id": case_id,
        "title": article.title,
        "source_type": "generated",
        "source_origin": article.source_origin,
        "case_kind": article.case_kind,
        "synthetic": True,
        "case_meta": {
            "case_id": case_id,
            "status": "rejected",
            "language": request.language,
            "difficulty": request.difficulty,
            "source_mode": request.source_mode,
            "voice_style": request.voice_style,
            "generator_version": GENERATOR_VERSION,
            "created_at": _utc_now_iso(),
        },
        "source_bundle": {
            "canonical_url": article.url,
            "external_ids": _normalized_external_ids(article),
            "rights": {
                "status": article.rights_status,
                "license_name": article.license_name,
                "license_url": article.license_url,
            },
            "fetch_timestamp": _utc_now_iso(),
            "fetched_at": _utc_now_iso(),
            "extraction_method": article.extraction_method,
            "source_trust_tier": article.trust_tier,
            "source_origin": article.source_origin,
            "case_kind": article.case_kind,
            "dedupe_keys": dedupe_keys,
            "medical_area": article.specialty,
            "pubmed_pmids": [article.pmid] if article.pmid else [],
            "pubmed_queries": [article.query] if article.query else [],
            "source_article": {
                "identifier": article.identifier,
                "title": article.title,
                "journal": article.journal,
                "pub_date": article.pub_date,
                "url": article.url,
                "abstract": article.abstract[:2500],
                "query": article.query,
            },
            "who_validation": {"validated": False, "codes": [], "matches": []},
            "source_hash": _stable_source_hash(article),
        },
        "clinical_facts": {},
        "patient_narrative": {"text": "", "language": request.language, "voice_style": request.voice_style},
        "patient_data": {"patient_text": ""},
        "expected_output": {},
        "quality_report": {
            "voice_score": 0.0,
            "leakage_score": 0.0,
            "consistency_score": 0.0,
            "novelty_score": 0.0,
            "source_fidelity_score": 0.0,
            "issues": ["Structured generation failed"],
            "validator_notes": error_message[:220],
            "review_required": False,
        },
        "review_required": False,
        "decision_meta": decision_meta,
        "draft_id": case_id,
    }


def _persist_case(payload: dict[str, Any], status: Literal["draft", "curated", "rejected"]) -> dict[str, Any]:
    _ensure_case_dirs()
    destination_dir = {"draft": DRAFTS_DIR, "curated": CURATED_DIR, "rejected": REJECTED_DIR}[status]
    payload.setdefault("case_meta", {})
    payload["case_meta"]["status"] = status
    file_path = destination_dir / _case_filename(str(payload.get("case_id", "") or str(uuid.uuid4())))
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["storage_path"] = str(file_path)
    return payload


def _decision_label(decision: str, status: str) -> str:
    mapping = {
        "auto_approved": "AUTO-APPROVED",
        "auto_rejected": "AUTO-REJECTED",
        "needs_review": "REVIEW",
        "manual_promoted": "MANUAL-PROMOTE",
        "manual_rejected": "MANUAL-REJECT",
    }
    if decision in mapping:
        return mapping[decision]
    return status.upper()


def _preview_case_payload(data: dict[str, Any], *, status: str) -> dict[str, Any]:
    narrative = data.get("patient_narrative", {}) or {}
    patient_data = data.get("patient_data", {}) or {}
    source_bundle = data.get("source_bundle", {}) or {}
    quality = data.get("quality_report", {}) or {}
    decision_meta = data.get("decision_meta", {}) or {}
    external_ids = source_bundle.get("external_ids", {}) or {}
    patient_text = str(narrative.get("text") or patient_data.get("patient_text") or "").strip()
    pmids = source_bundle.get("pubmed_pmids", []) or []
    source_origin = str(data.get("source_origin") or source_bundle.get("source_origin") or "generated")
    case_kind = str(data.get("case_kind") or source_bundle.get("case_kind") or "real_report")
    decision = str(decision_meta.get("decision") or ("needs_review" if status == "draft" else f"auto_{status}"))
    return {
        "id": data.get("case_id", ""),
        "title": data.get("title", "Untitled Draft"),
        "patient_text": patient_text,
        "diagnosis": (data.get("expected_output", {}) or {}).get("primary_diagnosis", "Unknown"),
        "expected_output": data.get("expected_output", {}) or {},
        "source_type": "generated",
        "source_origin": source_origin,
        "source_label": source_origin.replace("_", " ").upper(),
        "case_kind": case_kind,
        "case_kind_label": case_kind.replace("_", " ").upper(),
        "decision": decision,
        "decision_label": _decision_label(decision, status),
        "decision_reasons": decision_meta.get("decision_reasons", []) or [],
        "real_case": status == "curated",
        "language": ((data.get("case_meta", {}) or {}).get("language") or "en"),
        "difficulty": ((data.get("case_meta", {}) or {}).get("difficulty") or "moderate"),
        "voice_style": ((data.get("case_meta", {}) or {}).get("voice_style") or "patient_colloquial"),
        "pmids": pmids,
        "quality_report": quality,
        "review_required": bool(data.get("review_required", False)),
        "status": status,
        "canonical_url": source_bundle.get("canonical_url", ""),
        "external_ids": external_ids,
        "storage_path": data.get("storage_path", ""),
    }


def list_generated_cases(status: Literal["draft", "curated", "rejected"] = "draft") -> list[dict[str, Any]]:
    _ensure_case_dirs()
    root = {"draft": DRAFTS_DIR, "curated": CURATED_DIR, "rejected": REJECTED_DIR}[status]
    previews: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        data = _read_json_file(path)
        if not data:
            continue
        previews.append(_preview_case_payload(data, status=status))
    return previews


def list_generated_case_buckets(*, include_rejected: bool = False) -> dict[str, Any]:
    accepted = list_generated_cases(status="curated")
    review = list_generated_cases(status="draft")
    rejected = list_generated_cases(status="rejected")
    last_batch = _read_json_file(LAST_BATCH_SUMMARY_FILE) or {}
    meta_by_source = {source: {"accepted": 0, "review": 0, "rejected": 0} for source in SOURCE_REGISTRY}
    for bucket_name, cases in (("accepted", accepted), ("review", review), ("rejected", rejected)):
        for case in cases:
            source = str(case.get("source_origin", "") or "pubmed")
            if source not in meta_by_source:
                meta_by_source[source] = {"accepted": 0, "review": 0, "rejected": 0}
            meta_by_source[source][bucket_name] += 1
    return {
        "accepted": accepted,
        "review": review,
        "rejected": rejected if include_rejected else [],
        "meta": {
            "accepted_count": len(accepted),
            "review_count": len(review),
            "rejected_count": len(rejected),
            "hidden_rejected_count": int(last_batch.get("hidden_rejected_count", 0) or 0) if not include_rejected else len(rejected),
            "policy": "conservative_auto_review",
            "by_source": meta_by_source,
            "last_batch": last_batch,
        },
    }


def _locate_generated_case(case_id: str) -> tuple[Path, str]:
    for status, root in (("draft", DRAFTS_DIR), ("curated", CURATED_DIR), ("rejected", REJECTED_DIR)):
        path = root / _case_filename(case_id)
        if path.exists():
            return path, status
    raise FileNotFoundError(f"Generated case not found: {case_id}")


def _move_case(case_id: str, *, new_status: Literal["draft", "curated", "rejected"], new_decision: str, reason: str) -> dict[str, Any]:
    source_path, current_status = _locate_generated_case(case_id)
    data = _read_json_file(source_path)
    if not data:
        raise RuntimeError(f"Could not parse generated case: {case_id}")
    data.setdefault("case_meta", {})
    data["case_meta"]["status"] = new_status
    data["case_meta"][f"{new_status}_at"] = _utc_now_iso()
    decision_meta = data.setdefault("decision_meta", {})
    decision_meta["decision"] = new_decision
    reasons = list(decision_meta.get("decision_reasons", []) or [])
    reasons.insert(0, reason)
    decision_meta["decision_reasons"] = _dedupe_preserve(reasons)
    destination_dir = {"draft": DRAFTS_DIR, "curated": CURATED_DIR, "rejected": REJECTED_DIR}[new_status]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    destination_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if destination_path != source_path:
        source_path.unlink()
    data["storage_path"] = str(destination_path)
    return data


def promote_generated_case(case_id: str) -> dict[str, Any]:
    return _move_case(case_id, new_status="curated", new_decision="manual_promoted", reason="Manually promoted by reviewer")


def reject_generated_case(case_id: str) -> dict[str, Any]:
    return _move_case(case_id, new_status="rejected", new_decision="manual_rejected", reason="Manually rejected by reviewer")


async def generate_case_batch(
    gemini_client: GeminiClient,
    request: CaseGenerationRequest | dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_case_dirs()
    parsed_request = request if isinstance(request, CaseGenerationRequest) else CaseGenerationRequest.model_validate(request or {})
    rng = random.Random(parsed_request.seed if parsed_request.seed is not None else time.time_ns())
    used_dedupe_keys = _collect_existing_dedupe_keys()
    used_pmids = _collect_used_pubmed_pmids()

    candidates, harvested_counts = await _harvest_candidates(parsed_request, rng, used_dedupe_keys)
    if not candidates:
        raise RuntimeError("No eligible source candidates were harvested for the selected source set.")

    accepted_cases: list[dict[str, Any]] = []
    review_cases: list[dict[str, Any]] = []
    rejected_cases: list[dict[str, Any]] = []
    failure_samples: list[str] = []
    by_source = {
        source: {"harvested": harvested_counts.get(source, 0), "accepted": 0, "review": 0, "rejected": 0}
        for source in SOURCE_REGISTRY
    }

    kept_cases = 0
    attempted_cases = 0
    max_attempts = min(
        len(candidates),
        max(parsed_request.target_count * MAX_BATCH_ATTEMPT_MULTIPLIER, parsed_request.target_count + MAX_BATCH_ATTEMPT_BUFFER),
    )
    for article in candidates:
        if attempted_cases >= max_attempts:
            break
        attempted_cases += 1
        audit = GenerationAudit()
        try:
            facts = await _extract_clinical_facts(gemini_client, article, audit=audit)
            icd_validation = await _resolve_icd_validation(facts)
            narrative, quality = await _render_validated_narrative(
                gemini_client,
                article,
                facts,
                parsed_request,
                icd_validation,
                used_pmids,
                audit=audit,
            )
            decision_meta = _build_decision_meta(article, quality, icd_validation, audit)
            status = _decision_to_status(str(decision_meta["decision"]))
            payload = _build_case_payload(
                request=parsed_request,
                article=article,
                facts=facts,
                narrative=narrative,
                icd_validation=icd_validation,
                quality=quality,
                decision_meta=decision_meta,
                status=status,
            )
        except Exception as exc:
            failure_samples.append(f"{article.source_origin}:{article.identifier}")
            payload = _build_failed_case_payload(
                request=parsed_request,
                article=article,
                error_message=str(exc),
                audit=audit,
            )
            status = "rejected"

        source_origin = str(payload.get("source_origin", article.source_origin))
        if status == "curated":
            persisted = _persist_case(payload, status)  # type: ignore[arg-type]
            preview = _preview_case_payload(persisted, status=status)
            accepted_cases.append(preview)
            by_source[source_origin]["accepted"] += 1
            kept_cases += 1
        elif status == "draft":
            persisted = _persist_case(payload, status)  # type: ignore[arg-type]
            preview = _preview_case_payload(persisted, status=status)
            review_cases.append(preview)
            by_source[source_origin]["review"] += 1
            kept_cases += 1
        else:
            preview = _preview_case_payload(payload, status=status)
            rejected_cases.append(preview)
            by_source[source_origin]["rejected"] += 1

        used_dedupe_keys.update(_article_dedupe_keys(article))
        if article.pmid:
            used_pmids.add(article.pmid)
        if kept_cases >= parsed_request.target_count:
            break

    logger.info(
        "[CASE GENERATOR] batch completed target=%s attempted=%s kept=%s review=%s hidden_skips=%s sources=%s",
        parsed_request.target_count,
        attempted_cases,
        len(accepted_cases) + len(review_cases),
        len(review_cases),
        len(rejected_cases),
        ",".join(parsed_request.sources),
    )

    summary = {
        "status": "success",
        "source_mode": parsed_request.source_mode,
        "target_count": parsed_request.target_count,
        "attempted_count": attempted_cases,
        "accepted_count": len(accepted_cases),
        "review_count": len(review_cases),
        "rejected_count": len(rejected_cases),
        "hidden_rejected_count": len(rejected_cases),
        "accepted": accepted_cases,
        "review": review_cases,
        "rejected": [],
        "cases": [*accepted_cases, *review_cases],
        "meta": {
            "policy": "conservative_auto_review",
            "decision_version": DECISION_VERSION,
            "generator_version": GENERATOR_VERSION,
            "by_source": by_source,
            "harvested_candidates": len(candidates),
            "attempted_candidates": attempted_cases,
            "hidden_rejected_count": len(rejected_cases),
            "failure_samples": failure_samples[:5],
        },
        "message": (
            f"Processed {attempted_cases} source candidate(s): {len(accepted_cases)} ready, "
            f"{len(review_cases)} review, {len(rejected_cases)} hidden skip(s)."
        ),
    }
    _persist_last_batch_summary(
        {
            "generated_at": _utc_now_iso(),
            "source_mode": parsed_request.source_mode,
            "target_count": parsed_request.target_count,
            "attempted_count": attempted_cases,
            "accepted_count": len(accepted_cases),
            "review_count": len(review_cases),
            "hidden_rejected_count": len(rejected_cases),
            "by_source": by_source,
            "failure_samples": failure_samples[:5],
            "message": summary["message"],
        }
    )
    return summary


async def generate_new_case(
    gemini_client: GeminiClient,
    request: CaseGenerationRequest | dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_request = request if isinstance(request, CaseGenerationRequest) else CaseGenerationRequest.model_validate(request or {})
    batch = await generate_case_batch(gemini_client, parsed_request)
    if parsed_request.source_mode == "pubmed_primary" and parsed_request.target_count == 1 and batch.get("cases"):
        first_case = batch["cases"][0]
        full_path = first_case.get("storage_path")
        if full_path:
            stored = _read_json_file(Path(full_path))
            if stored:
                stored["storage_path"] = full_path
                return stored
        return first_case
    return batch
