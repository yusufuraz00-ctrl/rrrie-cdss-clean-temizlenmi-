"""Knowledge package exports.

The package is imported by phenotype/prototype code during runtime bootstrap, so
exports stay lazy to avoid pulling ontology and registry modules into the same
initialization cycle.
"""

__all__ = [
    "DiagnosisCandidateResolver",
    "OntologyDecision",
    "SyndromeProfile",
    "SyndromeProfileRegistry",
    "build_syndrome_registry_artifact",
    "ensure_syndrome_registry_artifact",
    "is_plausible_diagnosis_label",
    "load_syndrome_registry",
    "normalize_candidate_label",
]


def __getattr__(name: str):
    if name in {
        "DiagnosisCandidateResolver",
        "OntologyDecision",
        "is_plausible_diagnosis_label",
        "normalize_candidate_label",
    }:
        from src.cdss.knowledge import ontology

        return getattr(ontology, name)
    if name in {
        "SyndromeProfile",
        "SyndromeProfileRegistry",
        "build_syndrome_registry_artifact",
        "ensure_syndrome_registry_artifact",
        "load_syndrome_registry",
    }:
        from src.cdss.knowledge import registry

        return getattr(registry, name)
    raise AttributeError(name)
