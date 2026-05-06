"""API configuration — endpoint URLs and default parameters for medical APIs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PubMedConfig:
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    default_max_results: int = 5
    timeout: float = 30.0
    rate_limit_per_sec: int = 10  # with API key; 3 without


@dataclass(frozen=True)
class ClinicalTrialsConfig:
    base_url: str = "https://clinicaltrials.gov/api/v2"
    default_max_results: int = 5
    timeout: float = 30.0
    rate_limit_per_sec: int = 5


@dataclass(frozen=True)
class OpenFDAConfig:
    base_url: str = "https://api.fda.gov/drug"
    timeout: float = 30.0
    rate_limit_per_min: int = 240


@dataclass(frozen=True)
class MedlinePlusConfig:
    base_url: str = "https://connect.medlineplus.gov/service"
    timeout: float = 20.0


@dataclass(frozen=True)
class WHOConfig:
    base_url: str = "https://ghoapi.azureedge.net/api"
    timeout: float = 30.0


@dataclass(frozen=True)
class TavilyConfig:
    base_url: str = "https://api.tavily.com/search"
    default_search_depth: str = "advanced"
    default_max_results: int = 5
    timeout: float = 30.0
    trusted_medical_domains: tuple[str, ...] = (
        # Tier 1: Government & international health orgs
        "who.int",
        "cdc.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "fda.gov",
        "ema.europa.eu",
        # Tier 2: Clinical reference & guidelines
        "uptodate.com",
        "mayo.org",
        "mayoclinic.org",
        "cochranelibrary.com",
        "medscape.com",
        "emedicine.medscape.com",
        "merckmanuals.com",
        "drugs.com",
        # Tier 3: Top medical journals
        "bmj.com",
        "thelancet.com",
        "nejm.org",
        "nature.com",
        "springer.com",
        "wiley.com",
        "jamanetwork.com",
        "ahajournals.org",
        # Tier 4: Open access medical literature
        "europepmc.org",
        "scholar.google.com",
        "accessmedicine.mhmedical.com",
    )


@dataclass
class APIConfigs:
    """Container for all API configurations."""

    pubmed: PubMedConfig = field(default_factory=PubMedConfig)
    clinical_trials: ClinicalTrialsConfig = field(default_factory=ClinicalTrialsConfig)
    openfda: OpenFDAConfig = field(default_factory=OpenFDAConfig)
    medlineplus: MedlinePlusConfig = field(default_factory=MedlinePlusConfig)
    who: WHOConfig = field(default_factory=WHOConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)


def get_api_configs() -> APIConfigs:
    return APIConfigs()
