"""Real-world edge-case suite derived from PubMed case reports.

Run with pytest:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_cdss_pubmed_edge_cases.py -s

Run directly:
    .\\.venv\\Scripts\\python.exe tests\\test_cdss_pubmed_edge_cases.py
    .\\.venv\\Scripts\\python.exe tests\\test_cdss_pubmed_edge_cases.py --case anti_nmda_psychosis_mimic

Purpose:
    These are intentionally hard integration cases. They are meant to expose
    failure modes in diagnosis specificity, anchor-bias resistance, multilingual
    narrative understanding, and IE safety behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("CDSS_RUN_LIVE_LLM_TESTS") != "1",
    reason="requires a running local llama-server; set CDSS_RUN_LIVE_LLM_TESTS=1 to enable",
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdss_test_support import run_async
from src.cdss.app.service import CdssApplicationService
from src.cdss.contracts.models import DecisionPacket, PatientInput


@dataclass(frozen=True)
class EdgeCaseExpectation:
    diagnosis_keywords: tuple[str, ...]
    forbidden_top1_keywords: tuple[str, ...] = ()
    expected_status_keywords: tuple[str, ...] = ()
    expected_action_keywords: tuple[str, ...] = ()
    expected_required_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeCase:
    case_id: str
    title: str
    source_title: str
    source_url: str
    source_citation: str
    why_it_is_hard: str
    patient_text: str
    expectation: EdgeCaseExpectation


EDGE_CASES: tuple[EdgeCase, ...] = (
    EdgeCase(
        case_id="anti_nmda_psychosis_mimic",
        title="Acute Psychosis Plus Orofacial Dyskinesia Plus Seizure",
        source_title="Paraneoplastic anti-N-methyl-D-aspartate receptor encephalitis associated with ovarian teratoma",
        source_url="https://pubmed.ncbi.nlm.nih.gov/17262855/",
        source_citation="Dalmau et al., Ann Neurol. 2007;61(1):25-36. PMID: 17262855",
        why_it_is_hard=(
            "Tests whether the system can resist a primary-psychiatric anchor when the "
            "story contains a classic autoimmune encephalitis signature."
        ),
        patient_text=(
            "21 yasindaki kizimiz son bir haftada tamamen degisti. "
            "Odasinda bocekler var diye bagiriyor, televizyonun onunla konustugunu soyluyor. "
            "Dunden beri agziyla istemsiz cigeme ve yalanma hareketleri yapiyor. "
            "Bu aksam kasilip nobet gecirdi. Uyusturucu taramasi negatif denildi. "
            "Acil psikiyatri uzmani ilk sizofreni atagi veya agir psikotik kriz dusunup "
            "haloperidol yapilmasini ve kapali servise yatirisi planliyor. "
            "Nabiz 124, ates 37.9, solunum 24."
        ),
        expectation=EdgeCaseExpectation(
            diagnosis_keywords=("anti_nmda", "nmda", "autoimmune_encephalitis", "encephalitis"),
            forbidden_top1_keywords=("schizophrenia", "pseudobulbar", "acute_disseminated_encephalomyelitis"),
            expected_status_keywords=("urgent_escalation", "revise"),
            expected_action_keywords=("objective_testing", "review_treatment_assumptions", "state_aligned_objective_confirmation"),
            expected_required_keywords=("initial_labs", "objective", "contradiction"),
        ),
    ),
    EdgeCase(
        case_id="salicylate_psychosis_intubation_trap",
        title="Chronic Salicylate Toxicity Masquerading as Panic or Psychosis",
        source_title="\"I'm Seeing Dead People\": A Case Report on Salicylate Poisoning in a Patient with Hallucinations",
        source_url="https://pubmed.ncbi.nlm.nih.gov/39903610/",
        source_citation="Meyers et al., Clin Pract Cases Emerg Med. 2025;9(1):82-85. PMID: 39903610",
        why_it_is_hard=(
            "Tests whether the system can recover a toxic-metabolic diagnosis from a story "
            "that superficially looks psychiatric and can recognize the classic intubation trap."
        ),
        patient_text=(
            "19 yasinda kadin universite sinavina hazirlaniyor. Haftalardir bas agrisi icin "
            "her gun yuksek doz aspirin icmis. Son 12 saattir durmadan kusuyor, kulaklarim icinde "
            "cinlama var diyor ve olu insanlar gordugunu soyluyor. Cok hizli ve derin nefes alip veriyor. "
            "Ilk kan gazinda pCO2 dusuk bulunmus. Acil doktoru panik atak ile hiperventilasyon yorulmasi dusunup "
            "kas gevsetici vererek entubasyon planliyor, sonra laboratuvara bakariz diyor. "
            "Nabiz 128, solunum 36, ates 37.7."
        ),
        expectation=EdgeCaseExpectation(
            diagnosis_keywords=("salicylate", "aspirin_toxicity", "salicylate_toxicity"),
            forbidden_top1_keywords=("panic", "acute_psychosis", "schizophrenia"),
            expected_status_keywords=("urgent_escalation", "revise"),
            expected_action_keywords=("review_treatment_assumptions", "objective_testing", "state_aligned_workflow_review"),
            expected_required_keywords=("initial_labs", "objective", "contraindication"),
        ),
    ),
    EdgeCase(
        case_id="af_mesenteric_embolic_catastrophe",
        title="Abdominal Pain Out of Proportion in Atrial Fibrillation",
        source_title="Celiac trunk occlusion as a severe complication of atrial fibrillation: A case report",
        source_url="https://pubmed.ncbi.nlm.nih.gov/41438658/",
        source_citation="Bengrad et al., Radiol Case Rep. 2025;21(2):945-947. PMID: 41438658",
        why_it_is_hard=(
            "Tests whether the system can override a benign abdominal or anxiety anchor and "
            "surface embolic mesenteric ischemia from a sparse but lethal vascular pattern."
        ),
        patient_text=(
            "68 yasinda kadin, bilinen atriyal fibrilasyon ve anksiyete oykusu var. "
            "Uc gundur giderek artan cok siddetli karin agrisi var, bugun soguk terleme ve kusma eklenmis. "
            "Hasta karnim kopuyor, oluyorum diye kivraniyor. Muayenede karin beklenenden yumusak ama agrisi cok siddetli. "
            "Nabiz 136 ve duzensiz, tansiyon 86/54, solunum 28. Laktat 6.1 mmol/L ve lositoz var. "
            "Nobetteki doktor gastrit veya anksiyete alevlenmesi dusunup sakinlestirici, sivi ve gozlem planliyor."
        ),
        expectation=EdgeCaseExpectation(
            diagnosis_keywords=("mesenteric_ischemia", "celiac_trunk", "arterial_embol", "bowel_ischemia"),
            forbidden_top1_keywords=("anxiety", "gastritis", "gastroenteritis"),
            expected_status_keywords=("urgent_escalation", "revise"),
            expected_action_keywords=("objective_testing", "urgent_supervising_clinician_review", "state_aligned_objective_confirmation"),
            expected_required_keywords=("hemodynamic_assessment", "objective", "initial_labs"),
        ),
    ),
)


def _packet_text(packet: DecisionPacket) -> str:
    parts: list[str] = [
        packet.summary,
        packet.status.value if hasattr(packet.status, "value") else str(packet.status),
        *(candidate.label for candidate in packet.differential.candidates[:6]),
        *(node.label for node in packet.hypothesis_frontier.hypotheses[:6]),
        *packet.recommended_actions,
        *packet.required_data,
        *packet.dangerous_exclusions,
        *packet.missing_information,
        *packet.explanation,
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _top_label(packet: DecisionPacket) -> str:
    if packet.differential.candidates:
        return str(packet.differential.candidates[0].label or "").lower()
    if packet.hypothesis_frontier.hypotheses:
        return str(packet.hypothesis_frontier.hypotheses[0].label or "").lower()
    return ""


def _assert_edge_case(case: EdgeCase, packet: DecisionPacket) -> None:
    flat = _packet_text(packet)
    top_label = _top_label(packet)

    assert packet.differential.candidates or packet.hypothesis_frontier.hypotheses, (
        f"{case.case_id}: no ranked hypotheses returned"
    )
    assert any(keyword in flat for keyword in case.expectation.diagnosis_keywords), (
        f"{case.case_id}: expected one of {case.expectation.diagnosis_keywords} in output, "
        f"top_label={top_label!r}"
    )
    assert not any(keyword in top_label for keyword in case.expectation.forbidden_top1_keywords), (
        f"{case.case_id}: forbidden top-1 anchor {top_label!r}"
    )
    if case.expectation.expected_status_keywords:
        status_text = str(packet.status.value if hasattr(packet.status, "value") else packet.status).lower()
        assert any(keyword in status_text for keyword in case.expectation.expected_status_keywords), (
            f"{case.case_id}: unexpected status {status_text!r}"
        )
    if case.expectation.expected_action_keywords:
        assert any(keyword in flat for keyword in case.expectation.expected_action_keywords), (
            f"{case.case_id}: expected action keywords {case.expectation.expected_action_keywords} not found"
        )
    if case.expectation.expected_required_keywords:
        assert any(keyword in flat for keyword in case.expectation.expected_required_keywords), (
            f"{case.case_id}: expected required-data keywords {case.expectation.expected_required_keywords} not found"
        )


def _save_packet(case: EdgeCase, packet: DecisionPacket) -> Path:
    output_dir = ROOT / "output" / "edge_case_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{case.case_id}.json"
    target.write_text(json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _run_case(case: EdgeCase) -> DecisionPacket:
    service = CdssApplicationService()
    packet = run_async(
        service.analyze_case(
            PatientInput(
                case_id=case.case_id,
                patient_text=case.patient_text,
                language="tr",
            )
        )
    )
    _save_packet(case, packet)
    return packet


def test_pubmed_edge_case_catalog_has_sources() -> None:
    for case in EDGE_CASES:
        assert case.source_title
        assert case.source_url.startswith("https://")
        assert "PMID:" in case.source_citation
        assert case.patient_text
        assert case.expectation.diagnosis_keywords


@pytest.mark.parametrize("case", EDGE_CASES, ids=[case.case_id for case in EDGE_CASES])
def test_pubmed_edge_cases_live(case: EdgeCase) -> None:
    packet = _run_case(case)
    _assert_edge_case(case, packet)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-world PubMed edge cases against the adaptive CDSS.")
    parser.add_argument("--case", default="", help="Run only a single case_id.")
    args = parser.parse_args()

    selected = [case for case in EDGE_CASES if not args.case or case.case_id == args.case]
    if not selected:
        print(f"No edge case matched: {args.case}")
        return 1

    failures = 0
    for case in selected:
        print(f"\n=== {case.case_id} :: {case.title} ===")
        print(case.source_citation)
        print(case.source_url)
        packet = _run_case(case)
        top_label = _top_label(packet)
        print(f"status={packet.status.value if hasattr(packet.status, 'value') else packet.status}")
        print(f"top_label={top_label}")
        print(f"summary={packet.summary}")
        print(f"json_output={(ROOT / 'output' / 'edge_case_runs' / f'{case.case_id}.json')}")
        try:
            _assert_edge_case(case, packet)
            print("result=PASS")
        except AssertionError as exc:
            failures += 1
            print(f"result=FAIL :: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
