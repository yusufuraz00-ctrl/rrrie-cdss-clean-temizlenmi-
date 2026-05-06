r"""Sequential multi-case benchmark harness for the adaptive CDSS.

Run directly:
    .\.venv\Scripts\python.exe tests\benchmark_cdss_cases.py

Optional flags:
    --offline       Disable external evidence for reproducible local-only runs.
    --case <id>     Run only a single benchmark case id.
    --suite <name>  Benchmark suite name. Defaults to gold10. Available: gold10, fresh10, fresh50, legacy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cdss.app.service import CdssApplicationService
from src.llm.llama_cpp_client import LlamaCppClient
from src.cdss.contracts.models import DecisionPacket, DecisionStatus, PatientInput


LLM_ATTRIBUTION_SOURCES = {"llm_free_slate", "challenger_alt_hypothesis", "llm_or_hybrid"}
SUPPORT_ATTRIBUTION_SOURCES = {"helper_prior", "registry_profile_candidate", "mechanism_advisor"}


@dataclass(frozen=True)
class BenchmarkExpectation:
    keyword_hits: tuple[str, ...] = ()
    status_hits: tuple[str, ...] = ()
    blocked_hits: tuple[str, ...] = ()
    required_hits: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    difficulty: int
    disease_family: str
    note: str
    patient_text: str
    language: str = "en"
    expectations: BenchmarkExpectation = field(default_factory=BenchmarkExpectation)


def _json_benchmark_cases(filename: str) -> list[BenchmarkCase]:
    suite_path = Path(__file__).with_name(filename)
    payload = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    raw_cases = payload.get("cases", payload)
    cases: list[BenchmarkCase] = []
    for item in raw_cases:
        expectations = item.get("expectations", {}) or {}
        cases.append(
            BenchmarkCase(
                case_id=str(item.get("case_id", "") or ""),
                title=str(item.get("title", "") or ""),
                difficulty=int(item.get("difficulty", 0) or 0),
                disease_family=str(item.get("disease_family", "") or ""),
                note=str(item.get("note", "") or ""),
                patient_text=str(item.get("patient_text", "") or ""),
                language=str(item.get("language", "en") or "en"),
                expectations=BenchmarkExpectation(
                    keyword_hits=tuple(str(value) for value in (expectations.get("keyword_hits", []) or [])),
                    status_hits=tuple(str(value) for value in (expectations.get("status_hits", []) or [])),
                    blocked_hits=tuple(str(value) for value in (expectations.get("blocked_hits", []) or [])),
                    required_hits=tuple(str(value) for value in (expectations.get("required_hits", []) or [])),
                ),
            )
        )
    return cases


def _legacy_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="flu_common",
            title="Influenza-Like Viral URI",
            difficulty=2,
            disease_family="common_benign",
            note="Common outpatient respiratory illness; should avoid catastrophic over-escalation.",
            patient_text=(
                "26 year old teacher with 2 days of fever, sore throat, dry cough, runny nose, body aches, and sick contacts. "
                "Breathing is comfortable. SpO2 98%, pulse 92, blood pressure 118/74, temperature 38.4 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("influenza", "viral", "upper respiratory", "respiratory infection"),
                status_hits=("preliminary", "revise", "accept"),
            ),
        ),
        BenchmarkCase(
            case_id="dka_pediatric",
            title="Pediatric DKA",
            difficulty=6,
            disease_family="diabetes_emergency",
            note="Classic metabolic emergency with dehydration and respiratory compensation.",
            patient_text=(
                "12 year old with 2 days of vomiting, abdominal pain, intense thirst, frequent urination, lethargy, "
                "and deep rapid breathing. Glucose 420 mg/dL, ketones positive, blood pressure 88/54, pulse 142, "
                "respiratory rate 34, SpO2 92%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("dka", "ketoacidosis", "metabolic"),
                status_hits=("urgent_escalation", "revise"),
            ),
        ),
        BenchmarkCase(
            case_id="pneumonia_sepsis",
            title="Pneumonia With Sepsis Risk",
            difficulty=5,
            disease_family="infectious_critical",
            note="Common but dangerous infectious presentation.",
            patient_text=(
                "68 year old with productive cough, fever, confusion, and weakness for 3 days. "
                "Blood pressure 86/52, pulse 128, respiratory rate 30, temperature 39.3 C, SpO2 89%. "
                "Crackles at the right base and urine output is low."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("pneumonia", "sepsis", "infectious"),
                status_hits=("urgent_escalation", "revise"),
            ),
        ),
        BenchmarkCase(
            case_id="bacterial_meningitis",
            title="Meningitis / Encephalitis Danger",
            difficulty=7,
            disease_family="neuroinfectious",
            note="Must-not-miss CNS infection with high acuity.",
            patient_text=(
                "21 year old university student with high fever, severe headache, neck stiffness, vomiting, "
                "photophobia, and increasing confusion since this morning. Pulse 122, blood pressure 98/62, "
                "temperature 39.8 C, respiratory rate 24."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("meningitis", "encephalitis", "cns infection", "central nervous system infection"),
                status_hits=("urgent_escalation", "revise"),
            ),
        ),
        BenchmarkCase(
            case_id="guillain_barre",
            title="Guillain-Barre Style Ascending Weakness",
            difficulty=8,
            disease_family="rare_neuroimmune",
            note="Rare but high-risk neuromuscular pattern that should not be mistaken for simple weakness.",
            patient_text=(
                "34 year old with tingling in the feet after a diarrheal illness last week. Weakness has rapidly climbed "
                "from both legs to the thighs over 24 hours. He struggles to stand, reflexes are absent, and breathing "
                "feels slightly shallow. Pulse 108, blood pressure 126/78, SpO2 95%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("guillain", "ascending weakness", "neuromuscular", "acute flaccid paralysis"),
                status_hits=("urgent_escalation", "revise"),
            ),
        ),
        BenchmarkCase(
            case_id="pancreatic_cancer",
            title="Pancreatic Cancer / Obstructive Jaundice",
            difficulty=7,
            disease_family="oncology",
            note="Cancer-oriented benchmark to expose malignancy blind spots.",
            patient_text=(
                "67 year old smoker with 2 months of weight loss, painless jaundice, dark urine, pale stools, "
                "poor appetite, and dull epigastric discomfort radiating to the back. No fever. "
                "Blood pressure 124/76, pulse 98."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("pancreatic", "obstructive jaundice", "malignancy", "biliary obstruction", "cancer"),
                status_hits=("revise", "preliminary"),
            ),
        ),
        BenchmarkCase(
            case_id="rv_mi_trap",
            title="Inferior MI With RV Involvement Trap",
            difficulty=10,
            disease_family="cardiovascular_trap",
            note="Should block preload-dropping nitroglycerin and morphine.",
            patient_text=(
                "55 y male with chest pressure like an elephant on the chest, diaphoresis, and nausea. "
                "ED doctor says inferior MI and plans sublingual nitroglycerin plus morphine before angiography. "
                "Blood pressure 100/65 mmHg, pulse 55, temperature 36.5 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("inferior myocardial infarction", "rv involvement", "right ventricular"),
                status_hits=("urgent_escalation", "revise"),
                blocked_hits=("nitroglycerin", "morphine"),
                required_hits=("right_sided_ecg_v4r", "fluid"),
            ),
        ),
        BenchmarkCase(
            case_id="wernicke_hypoglycemia",
            title="Alcohol-Related Hypoglycemia / Wernicke Risk",
            difficulty=9,
            disease_family="toxic_metabolic_trap",
            note="Should recognize alcohol + nystagmus + coma as more than simple hypoglycemia.",
            patient_text=(
                "45 year old found unconscious, smelling strongly of alcohol, only groaning to pain. "
                "Eyes show constant nystagmus. Fingerstick glucose is 40 mg/dL. ED doctor plans two ampules of D50. "
                "Blood pressure 105/70, pulse 110, temperature 35.8 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("hypoglycemia", "wernicke", "thiamine", "alcohol"),
                status_hits=("revise", "urgent_escalation"),
            ),
        ),
        BenchmarkCase(
            case_id="massive_pe_trap",
            title="Massive PE S1Q3T3 Trap",
            difficulty=10,
            disease_family="cardiopulmonary_trap",
            note="Should break ACS anchoring and block nitroglycerin.",
            patient_text=(
                "60 year old woman after a long flight developed sudden shortness of breath and pleuritic chest pain. "
                "Right leg is swollen. ECG shows S1Q3T3. ED doctor says MI and plans nitroglycerin then angiography. "
                "Blood pressure 90/60, pulse 125, respiratory rate 30, SpO2 85%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("pulmonary embol", "massive pulmonary embolism", "obstructive risk", "s1q3t3"),
                status_hits=("urgent_escalation", "revise"),
                blocked_hits=("nitroglycerin",),
                required_hits=("ct_pulmonary_angiography", "pe_pathway"),
            ),
        ),
        BenchmarkCase(
            case_id="aortic_dissection_trap",
            title="Aortic Dissection Masquerading as MI",
            difficulty=10,
            disease_family="vascular_trap",
            note="Should flag tearing pain + arm pressure asymmetry and block anticoagulation.",
            patient_text=(
                "58 year old man with sudden chest pain tearing into the back between the shoulder blades and numbness in the right arm. "
                "ED doctor says diffuse ST changes mean a large MI and plans aspirin, high-dose heparin, and angiography. "
                "Right arm blood pressure 80/50, left arm blood pressure 160/95, pulse 110, temperature 36.6 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("aortic dissection", "acute aortic syndrome", "dissecting_aortic_aneurysm"),
                status_hits=("revise", "urgent_escalation"),
                blocked_hits=("heparin", "aspirin"),
                required_hits=("cta", "cardiothoracic surgery", "surgical"),
            ),
        ),
    ]


def _fresh_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="thyroid_storm_psychosis_trap",
            title="Thyroid Storm Hidden Behind Agitation",
            difficulty=10,
            disease_family="endocrine_trap",
            note="Hyperadrenergic endocrine crisis masquerading as psych/tox with aspirin danger.",
            patient_text=(
                "28 year old woman became severely agitated, shouting, febrile, and tremulous. "
                "Had diarrhea for 2 days. Husband says she used to take medicine for toxic goiter but stopped months ago. "
                "ED doctor says acute psychosis or stimulant intoxication and plans aspirin plus sedation. "
                "Blood pressure 150/60, pulse 165, temperature 39.8 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("thyroid storm", "thyrotoxic", "thyrotoxicosis", "hyperthyroidism"),
                status_hits=("urgent_escalation", "revise"),
                blocked_hits=("aspirin",),
                required_hits=("endocrine", "acetaminophen", "beta", "thionamide"),
            ),
        ),
        BenchmarkCase(
            case_id="anaphylaxis_silent_chest",
            title="Anaphylaxis With Near-Silent Chest",
            difficulty=10,
            disease_family="allergy_emergency",
            note="Should not be minimized as routine asthma; epinephrine pathway should surface.",
            patient_text=(
                "19 year old with peanut exposure developed lip swelling, hives, hoarse voice, and severe shortness of breath within minutes. "
                "Family says it is just asthma and wants inhaler puffs only. Breath sounds are very faint. "
                "Blood pressure 82/48, pulse 142, respiratory rate 32, SpO2 88%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("anaphylaxis", "allergic", "airway", "distributive shock"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("epinephrine", "airway", "oxygen"),
            ),
        ),
        BenchmarkCase(
            case_id="ectopic_pregnancy_shock",
            title="Ruptured Ectopic Pregnancy",
            difficulty=10,
            disease_family="gynecologic_emergency",
            note="Pelvic pain + syncope + missed period with shock physiology.",
            patient_text=(
                "27 year old with missed period, sudden severe lower abdominal pain, dizziness, and one syncopal episode. "
                "Small amount of vaginal bleeding. Pregnancy test positive at home. "
                "Blood pressure 84/50, pulse 136, temperature 36.8 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("ectopic pregnancy", "ruptured ectopic", "hemoperitoneum", "pregnancy related shock"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("pelvic ultrasound", "ob gyn", "crossmatch", "resuscitation"),
            ),
        ),
        BenchmarkCase(
            case_id="hyperkalemia_brady_arrhythmia",
            title="Hyperkalemia With Bradycardic Instability",
            difficulty=9,
            disease_family="metabolic_emergency",
            note="Electrolyte emergency with lethal rhythm risk.",
            patient_text=(
                "72 year old with kidney disease and weakness feels lightheaded and near fainting. "
                "Recently started new blood pressure medication. ECG reportedly shows very peaked T waves. "
                "Pulse 38, blood pressure 92/58, respiratory rate 18, SpO2 97%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("hyperkalemia", "electrolyte emergency", "bradyarrhythmia"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("calcium", "potassium", "ecg", "renal"),
            ),
        ),
        BenchmarkCase(
            case_id="subarachnoid_hemorrhage_thunderclap",
            title="Thunderclap Subarachnoid Hemorrhage",
            difficulty=10,
            disease_family="neurovascular_emergency",
            note="Classic catastrophic headache that should not be written off as migraine.",
            patient_text=(
                "41 year old with worst headache of life exploding suddenly during exercise, followed by vomiting and neck pain. "
                "Feels photophobic and briefly collapsed. "
                "Blood pressure 176/98, pulse 102, temperature 36.7 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("subarachnoid", "thunderclap headache", "aneurysmal bleed", "intracranial hemorrhage"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("ct head", "lumbar puncture", "neurosurgery"),
            ),
        ),
        BenchmarkCase(
            case_id="epidural_abscess_back_pain",
            title="Spinal Epidural Abscess Red-Flag Back Pain",
            difficulty=9,
            disease_family="spinal_infectious_emergency",
            note="Back pain plus fever and neuro deficits should force MRI pathway.",
            patient_text=(
                "52 year old with diabetes and recent skin infection has severe midline back pain, fever, and increasing leg weakness. "
                "Now has trouble urinating. Blood pressure 108/70, pulse 118, temperature 38.9 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("epidural abscess", "spinal infection", "cord compression", "cauda equina"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("spine mri", "neurosurgery", "broad spectrum antibiotics"),
            ),
        ),
        BenchmarkCase(
            case_id="giant_cell_arteritis_vision",
            title="Giant Cell Arteritis Threatening Vision",
            difficulty=8,
            disease_family="vasculitic_emergency",
            note="Headache and jaw claudication with visual symptoms should trigger steroid-first thinking.",
            patient_text=(
                "74 year old with new unilateral temporal headache, scalp tenderness, jaw pain while chewing, and brief episodes of blurry vision. "
                "Feels fatigued. Temperature 37.8 C, pulse 96, blood pressure 138/78."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("giant cell arteritis", "temporal arteritis", "vasculitis"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("steroids", "esr", "temporal artery", "ophthalmology"),
            ),
        ),
        BenchmarkCase(
            case_id="appendicitis_early",
            title="Early Appendicitis Surgical Abdomen",
            difficulty=6,
            disease_family="abdominal_urgent",
            note="Should avoid being flattened into generic abdominal process only.",
            patient_text=(
                "18 year old with abdominal pain starting near the belly button then moving to the right lower abdomen, with nausea and loss of appetite. "
                "Pain worsens with walking. Temperature 38.1 C, pulse 108, blood pressure 118/72."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("appendicitis", "right lower quadrant", "surgical abdomen"),
                status_hits=("urgent_escalation", "revise"),
                required_hits=("surgical", "abdominal imaging", "rebound"),
            ),
        ),
        BenchmarkCase(
            case_id="viral_pharyngitis_low_risk",
            title="Low-Risk Viral Pharyngitis",
            difficulty=2,
            disease_family="common_benign",
            note="Benign outpatient case should not demand emergency workup.",
            patient_text=(
                "24 year old with sore throat, runny nose, dry cough, low-grade fever, and sick roommate for 3 days. "
                "No breathing difficulty. SpO2 99%, pulse 88, blood pressure 116/70, temperature 37.9 C."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("viral pharyngitis", "pharyngitis", "upper respiratory infection", "viral uri"),
                status_hits=("preliminary", "revise", "accept"),
            ),
        ),
        BenchmarkCase(
            case_id="self_limited_gastroenteritis_low_risk",
            title="Self-Limited Gastroenteritis",
            difficulty=3,
            disease_family="common_benign",
            note="Should allow benign closure when dehydration and surgical red flags are absent.",
            patient_text=(
                "31 year old with one day of vomiting, loose stools, crampy abdominal discomfort, and sick family members with the same illness. "
                "Still drinking fluids. No blood in stool. Blood pressure 118/76, pulse 94, temperature 37.8 C, SpO2 98%."
            ),
            expectations=BenchmarkExpectation(
                keyword_hits=("gastroenteritis", "self limited", "supportive care", "diarrheal illness"),
                status_hits=("preliminary", "revise", "accept"),
            ),
        ),
    ]


def _benchmark_cases(suite: str) -> list[BenchmarkCase]:
    suites = {
        "fresh10": _fresh_benchmark_cases,
        "fresh50": lambda: _json_benchmark_cases("benchmark_suite_fresh50.json"),
        "gold10": lambda: _json_benchmark_cases("benchmark_suite_gold10.json"),
        "legacy": _legacy_benchmark_cases,
        "stress26": lambda: _json_benchmark_cases("benchmark_suite_stress26.json"),
    }
    try:
        return suites[suite]()
    except KeyError as exc:
        raise SystemExit(f"Unknown benchmark suite: {suite}") from exc


def _packet_stage_metrics(packet: DecisionPacket) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for item in packet.trace:
        stage_metrics = dict((item.payload or {}).get("metrics", {}) or {})
        if not stage_metrics:
            continue
        bucket = metrics.setdefault(item.stage, {"time_s": 0.0, "total_tokens": 0.0})
        bucket["time_s"] += float(stage_metrics.get("time_s", 0) or 0)
        bucket["total_tokens"] += float(
            stage_metrics.get(
                "total_tokens",
                (stage_metrics.get("prompt_tokens", 0) or 0) + (stage_metrics.get("completion_tokens", 0) or 0),
            )
            or 0
        )
    for bucket in metrics.values():
        bucket["time_s"] = round(bucket["time_s"], 2)
        bucket["total_tokens"] = int(bucket["total_tokens"])
    return metrics


def _packet_hypothesis_payload(packet: DecisionPacket) -> dict[str, Any]:
    for item in reversed(packet.trace):
        if item.stage == "hypothesis_generation":
            return dict(item.payload or {})
    return {}


def _attribution_breakdown(packet: DecisionPacket) -> dict[str, Any]:
    payload = _packet_hypothesis_payload(packet)
    histogram = dict(payload.get("source_histogram", {}) or {})
    total = sum(int(value or 0) for value in histogram.values())
    if total <= 0:
        return {
            "llm_share": None,
            "support_share": None,
            "helper_origin_share": packet.model_support.helper_origin_share,
            "effective_support_influence": packet.model_support.effective_support_influence,
            "support_proposer_correlation": packet.model_support.support_proposer_correlation,
            "other_share": None,
            "source_histogram": histogram,
            "helper_prior_used": bool(payload.get("supporting_helper_prior_used", False)),
            "llm_free_slate_used": bool(payload.get("llm_free_slate_used", False)),
            "llm_hypothesis_count": int(payload.get("llm_hypothesis_count", 0) or 0),
            "llm_error": dict(payload.get("llm_error", {}) or {}),
        }
    llm_count = sum(int(histogram.get(source, 0) or 0) for source in LLM_ATTRIBUTION_SOURCES)
    support_count = sum(int(histogram.get(source, 0) or 0) for source in SUPPORT_ATTRIBUTION_SOURCES)
    other_count = max(0, total - llm_count - support_count)
    return {
        "llm_share": round(llm_count / total, 2),
        "support_share": round(support_count / total, 2),
        "helper_origin_share": packet.model_support.helper_origin_share,
        "effective_support_influence": packet.model_support.effective_support_influence,
        "support_proposer_correlation": packet.model_support.support_proposer_correlation,
        "other_share": round(other_count / total, 2),
        "source_histogram": histogram,
        "helper_prior_used": bool(payload.get("supporting_helper_prior_used", False)),
        "llm_free_slate_used": bool(payload.get("llm_free_slate_used", False)),
        "llm_hypothesis_count": int(payload.get("llm_hypothesis_count", 0) or 0),
        "llm_error": dict(payload.get("llm_error", {}) or {}),
    }


def _flatten_packet_text(packet: DecisionPacket) -> str:
    values: list[str] = [
        packet.summary,
        packet.anchor_hypothesis,
        packet.verification.recommended_disposition,
        *packet.must_not_miss,
        *packet.blocked_interventions,
        *packet.required_concurrent_actions,
        *packet.required_data,
        *packet.reasoning_trace,
    ]
    for candidate in packet.differential.candidates:
        values.append(candidate.label)
        values.extend(candidate.rationale)
    for issue in packet.verification.issues:
        values.append(issue.issue_type)
        values.append(issue.detail)
    for action in packet.recommended_actions:
        values.append(action)
    return " ".join(str(value or "") for value in values).lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    normalized_text = _normalize_label_text(text)
    return any(_normalize_label_text(pattern) in normalized_text for pattern in patterns)


def _normalize_label_text(value: str) -> str:
    normalized = str(value or "").replace("-", " ").replace("_", " ").lower().strip()
    normalized = normalized.replace("cellarteritis", "cell arteritis")
    normalized = normalized.replace("subarachnoidhemorrhage", "subarachnoid hemorrhage")
    normalized = " ".join(normalized.split())
    return normalized


def _match_list(items: list[str], patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    flat = _normalize_label_text(" ".join(str(item or "") for item in items))
    return any(_normalize_label_text(pattern) in flat for pattern in patterns)


def _label_matches(label: str, patterns: tuple[str, ...]) -> bool:
    value = _normalize_label_text(label)
    if not patterns:
        return True
    return any(_normalize_label_text(pattern) in value for pattern in patterns)


def _score_case(case: BenchmarkCase, packet: DecisionPacket) -> dict[str, Any]:
    packet_text = _flatten_packet_text(packet)
    top_label = packet.differential.candidates[0].label if packet.differential.candidates else ""
    anchor_label = packet.anchor_hypothesis or top_label
    low_risk_case = "preliminary" in set(case.expectations.status_hits)
    state_reasoning = bool(packet.explanation_graph.state_frames) and packet.reliability.state_coherence >= 0.4
    explanation_reasoning = (
        packet.reliability.explanation_completeness >= 0.42
        and (packet.reliability.critical_unexplained_ratio <= 0.65 or packet.status != DecisionStatus.PRELIMINARY)
    )
    safety_reasoning = packet.verification.safety_gate
    closure_behavior = packet.verification.closure_gate or (
        packet.reliability.closure_readiness >= 0.52 if low_risk_case else packet.status != DecisionStatus.PRELIMINARY or packet.reliability.closure_readiness <= 0.72
    )
    blocked_pool = [
        *packet.blocked_interventions,
        *packet.risk_profile.blocked_actions,
    ]
    required_pool = [
        *packet.required_concurrent_actions,
        *packet.recommended_actions,
        *packet.risk_profile.required_actions,
        *packet.required_data,
    ]
    criteria: list[tuple[str, float, bool]] = [
        ("diagnosis_top", 0.32, _label_matches(top_label, case.expectations.keyword_hits)),
        ("diagnosis_anchor_or_support", 0.08, _label_matches(anchor_label, case.expectations.keyword_hits) or _contains_any(packet_text, case.expectations.keyword_hits)),
        ("status", 0.1, packet.status.value in set(case.expectations.status_hits or (packet.status.value,))),
        ("state_reasoning", 0.14, state_reasoning),
        ("safety_reasoning", 0.14, safety_reasoning),
        ("explanation_reasoning", 0.14, explanation_reasoning),
        ("closure_behavior", 0.06, closure_behavior),
        ("blocked", 0.03, _match_list(blocked_pool, case.expectations.blocked_hits)),
        ("required", 0.02, _match_list(required_pool, case.expectations.required_hits)),
    ]
    applicable = [
        (name, weight, hit)
        for name, weight, hit in criteria
        if (
            name in {
                "diagnosis_top",
                "diagnosis_anchor_or_support",
                "status",
                "state_reasoning",
                "safety_reasoning",
                "explanation_reasoning",
                "closure_behavior",
            }
            or (name == "blocked" and case.expectations.blocked_hits)
            or (name == "required" and case.expectations.required_hits)
        )
    ]
    total_weight = sum(weight for _, weight, _ in applicable) or 1.0
    normalized_score = round(sum(weight for _, weight, hit in applicable if hit) / total_weight, 2)
    diagnosis_top = _label_matches(top_label, case.expectations.keyword_hits)
    diagnosis_family = _label_matches(anchor_label, case.expectations.keyword_hits) or _contains_any(packet_text, case.expectations.keyword_hits)
    if not safety_reasoning and not diagnosis_top:
        normalized_score = min(normalized_score, 0.34)
    elif not diagnosis_top and not diagnosis_family:
        normalized_score = min(normalized_score, 0.34)
    elif not diagnosis_top:
        normalized_score = min(normalized_score, 0.58)
    elif not safety_reasoning:
        normalized_score = min(normalized_score, 0.58)
    if not low_risk_case and packet.status == DecisionStatus.PRELIMINARY:
        normalized_score = min(normalized_score, 0.58)
    if safety_reasoning and diagnosis_top and normalized_score >= 0.75:
        grade = "pass"
    elif (diagnosis_top or diagnosis_family) and normalized_score >= 0.45:
        grade = "partial"
    else:
        grade = "fail"
    overflow = any(candidate.score > 0.99 for candidate in packet.differential.candidates)
    return {
        "score": normalized_score,
        "grade": grade,
        "checks": {name: hit for name, _, hit in applicable},
        "confidence_overflow": overflow,
        "top_label": top_label,
        "top_score": packet.differential.candidates[0].score if packet.differential.candidates else 0.0,
    }


def _to_patient_input(case: BenchmarkCase) -> PatientInput:
    return PatientInput(
        case_id=case.case_id,
        patient_text=case.patient_text,
        language=case.language,
        source="benchmark_harness",
    )


def _unsupported_claim_count(packet: DecisionPacket) -> int:
    unsupported_issue_types = {
        "reasoning_incomplete",
        "evidence_gap",
        "specificity_gap",
        "no_ranked_differential",
    }
    issue_count = sum(
        1
        for item in packet.verification.issues
        if str(item.issue_type or "").strip().lower() in unsupported_issue_types
    )
    return max(issue_count, int(packet.explanation_graph.critical_unexplained_count or 0))


def _unsafe_green(packet: DecisionPacket) -> bool:
    if packet.status != DecisionStatus.PRELIMINARY:
        return False
    blocking_issue_types = {
        "unsafe_closure",
        "missed_critical_risk",
        "must_not_miss_gap",
        "contraindicated_intervention",
        "unsafe_plan_blocked",
    }
    return any(str(item.issue_type or "").strip().lower() in blocking_issue_types for item in packet.verification.issues)


def _generic_top_label(packet: DecisionPacket) -> bool:
    top_label = packet.differential.candidates[0].label if packet.differential.candidates else ""
    return str(top_label or "").strip().lower().endswith("_process")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return round(float(ordered[index]), 2)


async def _run_case(service: CdssApplicationService, case: BenchmarkCase) -> dict[str, Any]:
    t0 = time.perf_counter()
    packet = await service.analyze_case(_to_patient_input(case))
    elapsed = round(time.perf_counter() - t0, 2)
    scorecard = _score_case(case, packet)
    attribution = _attribution_breakdown(packet)
    return {
        "case_id": case.case_id,
        "title": case.title,
        "difficulty": case.difficulty,
        "disease_family": case.disease_family,
        "note": case.note,
        "elapsed_s": elapsed,
        "language": case.language,
        "status": packet.status.value,
        "anchor_hypothesis": packet.anchor_hypothesis,
        "top_differential": scorecard["top_label"],
        "top_score": scorecard["top_score"],
        "generic_top_label": _generic_top_label(packet),
        "summary": packet.summary,
        "score": scorecard["score"],
        "grade": scorecard["grade"],
        "checks": scorecard["checks"],
        "confidence_overflow": scorecard["confidence_overflow"],
        "unsupported_claims": _unsupported_claim_count(packet),
        "unsafe_green": _unsafe_green(packet),
        "reliability": packet.reliability.overall,
        "ood_score": packet.ood_assessment.ood_score,
        "model_disagreement": packet.model_support.model_disagreement,
        "complexity_score": packet.complexity.score,
        "complexity_route": packet.complexity.route,
        "retrieval_specificity_gain": packet.retrieval_stats.specificity_gain,
        "state_coherence": packet.reliability.state_coherence,
        "explanation_completeness": packet.reliability.explanation_completeness,
        "closure_readiness": packet.reliability.closure_readiness,
        "state_frames": packet.explanation_graph.state_frames,
        "must_not_miss": packet.must_not_miss,
        "blocked_interventions": packet.blocked_interventions,
        "required_concurrent_actions": packet.required_concurrent_actions,
        "required_data": packet.required_data,
        "recommended_actions": packet.recommended_actions,
        "verification_issues": [issue.model_dump(mode="json") for issue in packet.verification.issues],
        "loop_history": packet.loop_history,
        "trace_stage_metrics": _packet_stage_metrics(packet),
        "trace_stages": [item.stage for item in packet.trace],
        "attribution": attribution,
        "query_hygiene_score": packet.retrieval_stats.query_hygiene_score,
        "hazard_leakage_rate": packet.retrieval_stats.hazard_leakage_rate,
        "evidence_coverage": packet.evidence.coverage,
        "contradiction_mass": packet.evidence.contradiction_mass,
        "retrieval_queries": [item.query_hint for item in packet.evidence.retrieval_intents],
        "evidence_needs": [item.model_dump(mode="json") for item in packet.evidence.evidence_needs],
        "differential": [candidate.model_dump(mode="json") for candidate in packet.differential.candidates[:5]],
        "retrospective": packet.retrospective_stub.model_dump(mode="json"),
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    elapsed_values = [float(item["elapsed_s"]) for item in results]
    passes = sum(1 for item in results if item["grade"] == "pass")
    partials = sum(1 for item in results if item["grade"] == "partial")
    fails = sum(1 for item in results if item["grade"] == "fail")
    mean_score = round(sum(item["score"] for item in results) / max(1, total), 2)
    mean_time = round(sum(item["elapsed_s"] for item in results) / max(1, total), 2)
    median_time = round(float(statistics.median(elapsed_values)) if elapsed_values else 0.0, 2)
    p95_time = _p95(elapsed_values)
    mean_reliability = round(sum(item["reliability"] for item in results) / max(1, total), 2)
    mean_state_coherence = round(sum(item["state_coherence"] for item in results) / max(1, total), 2)
    mean_explanation_completeness = round(sum(item["explanation_completeness"] for item in results) / max(1, total), 2)
    mean_closure_readiness = round(sum(item["closure_readiness"] for item in results) / max(1, total), 2)
    mean_ood = round(sum(item["ood_score"] for item in results) / max(1, total), 2)
    mean_retrieval_gain = round(sum(item["retrieval_specificity_gain"] for item in results) / max(1, total), 2)
    diagnosis_hit_rate = round(sum(1 for item in results if item["checks"].get("diagnosis_top")) / max(1, total), 2)
    diagnosis_family_rate = round(sum(1 for item in results if item["checks"].get("diagnosis_anchor_or_support")) / max(1, total), 2)
    llm_share_values = [item["attribution"]["llm_share"] for item in results if item["attribution"]["llm_share"] is not None]
    helper_origin_values = [item["attribution"]["helper_origin_share"] for item in results if item["attribution"].get("helper_origin_share") is not None]
    effective_support_values = [item["attribution"]["effective_support_influence"] for item in results if item["attribution"].get("effective_support_influence") is not None]
    query_hygiene_values = [item["query_hygiene_score"] for item in results]
    hazard_leakage_values = [item["hazard_leakage_rate"] for item in results]
    safety_cases = [item for item in results if "blocked" in item["checks"]]
    safety_hit_rate = round(
        sum(1 for item in safety_cases if item["checks"].get("blocked")) / max(1, len(safety_cases)),
        2,
    )
    required_cases = [item for item in results if "required" in item["checks"]]
    required_hit_rate = round(
        sum(1 for item in required_cases if item["checks"].get("required")) / max(1, len(required_cases)),
        2,
    )
    loop_counts = [len(item.get("loop_history") or []) for item in results]
    mean_loop_density = round(sum(loop_counts) / max(1, total), 2)
    loop_over_budget_cases = [item["case_id"] for item in results if len(item.get("loop_history") or []) > 2]
    overflow_cases = [item["case_id"] for item in results if item["confidence_overflow"]]
    unsafe_green_cases = [item["case_id"] for item in results if item.get("unsafe_green")]
    unsupported_claim_cases = [item["case_id"] for item in results if int(item.get("unsupported_claims", 0) or 0) > 0]
    generic_top_label_cases = [item["case_id"] for item in results if item.get("generic_top_label")]
    unsupported_claim_total = sum(int(item.get("unsupported_claims", 0) or 0) for item in results)
    weakest = sorted(results, key=lambda item: (item["score"], -item["difficulty"]))[:3]
    return {
        "cases_total": total,
        "pass_count": passes,
        "partial_count": partials,
        "fail_count": fails,
        "mean_score": mean_score,
        "mean_elapsed_s": mean_time,
        "median_elapsed_s": median_time,
        "p95_elapsed_s": p95_time,
        "mean_reliability": mean_reliability,
        "mean_state_coherence": mean_state_coherence,
        "mean_explanation_completeness": mean_explanation_completeness,
        "mean_closure_readiness": mean_closure_readiness,
        "mean_ood_score": mean_ood,
        "mean_retrieval_specificity_gain": mean_retrieval_gain,
        "diagnosis_hit_rate": diagnosis_hit_rate,
        "diagnosis_top1_rate": diagnosis_hit_rate,
        "diagnosis_family_rate": diagnosis_family_rate,
        "mean_llm_share": round(sum(llm_share_values) / max(1, len(llm_share_values)), 2) if llm_share_values else None,
        "mean_support_share": round(sum(helper_origin_values) / max(1, len(helper_origin_values)), 2) if helper_origin_values else None,
        "mean_helper_origin_share": round(sum(helper_origin_values) / max(1, len(helper_origin_values)), 2) if helper_origin_values else None,
        "mean_effective_support_influence": round(sum(effective_support_values) / max(1, len(effective_support_values)), 2) if effective_support_values else None,
        "mean_query_hygiene_score": round(sum(query_hygiene_values) / max(1, len(query_hygiene_values)), 2) if query_hygiene_values else None,
        "hazard_leakage_rate": round(sum(hazard_leakage_values) / max(1, len(hazard_leakage_values)), 2) if hazard_leakage_values else None,
        "safety_hit_rate": safety_hit_rate,
        "required_action_hit_rate": required_hit_rate,
        "mean_loop_density": mean_loop_density,
        "loop_over_budget_rate": round(len(loop_over_budget_cases) / max(1, total), 2),
        "loop_over_budget_cases": loop_over_budget_cases,
        "confidence_overflow_cases": overflow_cases,
        "unsafe_green_rate": round(len(unsafe_green_cases) / max(1, total), 2),
        "unsafe_green_cases": unsafe_green_cases,
        "unsupported_claim_case_rate": round(len(unsupported_claim_cases) / max(1, total), 2),
        "unsupported_claim_total": unsupported_claim_total,
        "unsupported_claim_cases": unsupported_claim_cases,
        "generic_top_label_rate": round(len(generic_top_label_cases) / max(1, total), 2),
        "generic_top_label_cases": generic_top_label_cases,
        "weakest_cases": [
            {
                "case_id": item["case_id"],
                "title": item["title"],
                "score": item["score"],
                "top_differential": item["top_differential"],
                "status": item["status"],
            }
            for item in weakest
        ],
    }


def _render_markdown(summary: dict[str, Any], results: list[dict[str, Any]], suite: str) -> str:
    lines: list[str] = []
    lines.append("# CDSS 10-Case Benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Suite: `{suite}`")
    lines.append(f"- Cases: `{summary['cases_total']}`")
    lines.append(f"- Pass / Partial / Fail: `{summary['pass_count']}` / `{summary['partial_count']}` / `{summary['fail_count']}`")
    lines.append(f"- Mean score: `{summary['mean_score']}`")
    lines.append(f"- Mean runtime: `{summary['mean_elapsed_s']} s`")
    lines.append(f"- Median runtime: `{summary['median_elapsed_s']} s`")
    lines.append(f"- P95 runtime: `{summary['p95_elapsed_s']} s`")
    lines.append(f"- Mean reliability: `{summary['mean_reliability']}`")
    lines.append(f"- Mean state coherence: `{summary['mean_state_coherence']}`")
    lines.append(f"- Mean explanation completeness: `{summary['mean_explanation_completeness']}`")
    lines.append(f"- Mean closure readiness: `{summary['mean_closure_readiness']}`")
    lines.append(f"- Mean OOD score: `{summary['mean_ood_score']}`")
    lines.append(f"- Mean retrieval specificity gain: `{summary['mean_retrieval_specificity_gain']}`")
    lines.append(f"- Diagnosis top-1 rate: `{summary['diagnosis_top1_rate']}`")
    lines.append(f"- Diagnosis family rate: `{summary['diagnosis_family_rate']}`")
    if summary.get("mean_llm_share") is not None:
        lines.append(f"- Mean LLM frontier share: `{summary['mean_llm_share']}`")
    if summary.get("mean_support_share") is not None:
        lines.append(f"- Mean helper-origin share: `{summary['mean_support_share']}`")
    if summary.get("mean_effective_support_influence") is not None:
        lines.append(f"- Mean effective support influence: `{summary['mean_effective_support_influence']}`")
    if summary.get("mean_query_hygiene_score") is not None:
        lines.append(f"- Mean query hygiene score: `{summary['mean_query_hygiene_score']}`")
    if summary.get("hazard_leakage_rate") is not None:
        lines.append(f"- Hazard leakage rate: `{summary['hazard_leakage_rate']}`")
    lines.append(f"- Safety hit rate (Blocked Avoidance): `{summary['safety_hit_rate']}`")
    lines.append(f"- Required action hit rate: `{summary['required_action_hit_rate']}`")
    lines.append(f"- Mean Loop Density (Autonomy persistence): `{summary['mean_loop_density']}`")
    lines.append(f"- Loop-over-budget rate (>2): `{summary['loop_over_budget_rate']}`")
    lines.append(f"- Unsafe green rate: `{summary['unsafe_green_rate']}`")
    lines.append(f"- Unsupported-claim case rate: `{summary['unsupported_claim_case_rate']}`")
    lines.append(f"- Unsupported-claim total: `{summary['unsupported_claim_total']}`")
    lines.append(f"- Generic top-label rate: `{summary['generic_top_label_rate']}`")
    if summary["confidence_overflow_cases"]:
        lines.append(f"- Confidence overflow cases: `{', '.join(summary['confidence_overflow_cases'])}`")
    if summary["loop_over_budget_cases"]:
        lines.append(f"- Loop-over-budget cases: `{', '.join(summary['loop_over_budget_cases'])}`")
    if summary["unsafe_green_cases"]:
        lines.append(f"- Unsafe green cases: `{', '.join(summary['unsafe_green_cases'])}`")
    if summary["unsupported_claim_cases"]:
        lines.append(f"- Unsupported-claim cases: `{', '.join(summary['unsupported_claim_cases'])}`")
    if summary["generic_top_label_cases"]:
        lines.append(f"- Generic top-label cases: `{', '.join(summary['generic_top_label_cases'])}`")
    lines.append("")
    lines.append("## Case Results")
    lines.append("")
    for item in results:
        lines.append(f"### {item['title']} ({item['case_id']})")
        lines.append("")
        lines.append(f"- Difficulty: `{item['difficulty']}`")
        lines.append(f"- Language: `{item['language']}`")
        lines.append(f"- Family: `{item['disease_family']}`")
        lines.append(f"- Grade: `{item['grade']}`")
        lines.append(f"- Score: `{item['score']}`")
        lines.append(f"- Runtime: `{item['elapsed_s']} s`")
        lines.append(f"- Status: `{item['status']}`")
        lines.append(f"- Anchor: `{item['anchor_hypothesis'] or 'n/a'}`")
        lines.append(f"- Top differential: `{item['top_differential']}` ({item['top_score']})")
        lines.append(f"- Generic top label: `{item['generic_top_label']}`")
        lines.append(f"- Unsupported claims: `{item['unsupported_claims']}`")
        lines.append(f"- Unsafe green: `{item['unsafe_green']}`")
        lines.append(f"- Reliability: `{item['reliability']}`")
        lines.append(f"- OOD score: `{item['ood_score']}`")
        lines.append(f"- Model disagreement: `{item['model_disagreement']}`")
        lines.append(f"- Complexity route: `{item['complexity_route']}` ({item['complexity_score']})")
        lines.append(f"- Retrieval specificity gain: `{item['retrieval_specificity_gain']}`")
        if item["attribution"]["llm_share"] is not None:
            lines.append(
                f"- Attribution: `llm={item['attribution']['llm_share']}` / "
                f"`helper_origin={item['attribution']['helper_origin_share']}` / "
                f"`other={item['attribution']['other_share']}`"
            )
        if item["attribution"].get("effective_support_influence") is not None:
            lines.append(f"- Effective support influence: `{item['attribution']['effective_support_influence']}`")
        lines.append(f"- Query hygiene: `{item['query_hygiene_score']}` / hazard leakage `{item['hazard_leakage_rate']}`")
        lines.append(
            f"- LLM slate: `used={item['attribution'].get('llm_free_slate_used', False)}` / "
            f"`count={item['attribution'].get('llm_hypothesis_count', 0)}`"
        )
        if item["attribution"].get("llm_error"):
            lines.append(
                f"- LLM error: `{item['attribution']['llm_error'].get('type', 'unknown')}: "
                f"{item['attribution']['llm_error'].get('message', '')}`"
            )
        if item["attribution"]["helper_prior_used"]:
            lines.append("- Helper prior: `used`")
        lines.append(f"- Checks: `{json.dumps(item['checks'], ensure_ascii=False)}`")
        if item["blocked_interventions"]:
            lines.append(f"- Blocked interventions: `{', '.join(item['blocked_interventions'])}`")
        if item["required_concurrent_actions"]:
            lines.append(f"- Required concurrent actions: `{', '.join(item['required_concurrent_actions'])}`")
        if item["verification_issues"]:
            issue_tags = ", ".join(f"{issue['issue_type']}:{issue['severity']}" for issue in item["verification_issues"][:6])
            lines.append(f"- Verification issues: `{issue_tags}`")
        lines.append(f"- Loop history: `{', '.join(item['loop_history']) if item['loop_history'] else 'none'}`")
        lines.append(f"- Trace stages: `{', '.join(item['trace_stages'])}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


async def _run_all(selected_case_id: str | None = None, suite: str = "gold10") -> dict[str, Any]:
    service = CdssApplicationService()
    cases = _benchmark_cases(suite)
    if selected_case_id:
        cases = [case for case in cases if case.case_id == selected_case_id]
        if not cases:
            raise SystemExit(f"Unknown case id: {selected_case_id}")

    results: list[dict[str, Any]] = []
    # --- Intelligence Validation Gate v3 (Retry Loop) ---
    if suite == "stress26" and cases:
        print(f"[*] VALIDATION GATE: Checking LLM connectivity...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                LlamaCppClient.reset_server_state()
                client = LlamaCppClient.get_instance()
                
                print(f"  -> Attempt {attempt}/{max_retries}: Pinging LLM server for life...")
                test_resp = client.chat([{"role": "user", "content": "HI"}], max_tokens=10)
                if not getattr(test_resp, "completion_tokens", 0) > 0:
                    raise ConnectionError("LLM server responded but generated 0 tokens.")
                
                print(f"  -> Validation passed. Proceeding with suite...\n")
                break
            except Exception as exc:
                print(f"[!] Validation attempt {attempt} failed: {exc}")
                if attempt < max_retries:
                    print("    Check: Is llama-server.exe running on port 8080?")
                    print(f"    Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("\n" + "!" * 80)
                    print(f"CRITICAL ERROR: LLM INTELLIGENCE LAYER OFFLINE")
                    print("Stress26 benchmarks require a functional LLM server (port 8080).")
                    print("Please run: .\\run_llm.bat in a separate terminal.")
                    print("!" * 80 + "\n")
                    raise SystemExit("Benchmark aborted: Failed to initialize LLM intelligence.") from exc
        remaining_cases = cases
    else:
        remaining_cases = cases

    for index, case in enumerate(remaining_cases, start=1):
        print(f"[{index}/{len(cases)}] Running {case.case_id} :: {case.title}")
        result = await _run_case(service, case)
        results.append(result)
        print(
            f"  -> grade={result['grade']} score={result['score']} status={result['status']} "
            f"top={result['top_differential']} runtime={result['elapsed_s']}s"
        )
    summary = _build_summary(results)
    return {"suite": suite, "summary": summary, "cases": results}


def _write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite = str(report.get("suite") or "gold10")
    json_path = output_dir / f"cdss_case_benchmark_{suite}_{stamp}.json"
    md_path = output_dir / f"cdss_case_benchmark_{suite}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(report["summary"], report["cases"], suite), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CDSS benchmark harness.")
    parser.add_argument("--offline", action="store_true", help="Disable external evidence calls for reproducible local-only runs.")
    parser.add_argument("--case", dest="case_id", default=None, help="Run only a single case id.")
    parser.add_argument(
        "--suite",
        default="gold10",
        help="Benchmark suite name: fresh10, gold10, fresh50, legacy, or stress26.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output" / "benchmarks"),
        help="Directory for JSON and Markdown benchmark reports.",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["CDSS_EXTERNAL_EVIDENCE_ENABLED"] = "0"

    report = asyncio.run(_run_all(selected_case_id=args.case_id, suite=args.suite))
    output_dir = Path(args.output_dir)
    json_path, md_path = _write_outputs(report, output_dir)

    print("")
    print("Benchmark summary")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print("")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()

