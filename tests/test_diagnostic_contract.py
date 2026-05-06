from src.cdss.runtime.diagnostic_contract import (
    DiagnosticContract,
    ResearchEvidenceItem,
    ResearchQueryPlan,
    contract_requires_research,
)


def test_contract_requires_research_when_source_disease_open():
    contract = DiagnosticContract(source_disease_resolved=False)
    assert contract_requires_research(contract) is True


def test_contract_requires_research_when_must_not_miss_open():
    contract = DiagnosticContract(must_not_miss=["aortic_dissection"])
    assert contract_requires_research(contract) is True


def test_contract_does_not_require_research_when_blocked():
    contract = DiagnosticContract(research_status="blocked")
    assert contract_requires_research(contract) is False


def test_research_query_plan_shape():
    query = ResearchQueryPlan(
        intent="confirmatory_test",
        query="central retinal artery occlusion urgent confirmation pathway",
        source_classes=["guideline", "ncbi"],
        rationale="Need urgent objective confirmation guidance.",
    )
    assert query.intent == "confirmatory_test"
    assert "guideline" in query.source_classes


def test_research_evidence_item_shape():
    evidence = ResearchEvidenceItem(
        claim="CRAO is a stroke-equivalent emergency.",
        citation="NCBI Bookshelf",
        source_class="ncbi",
        trust_tier="validated",
    )
    assert evidence.source_class == "ncbi"
