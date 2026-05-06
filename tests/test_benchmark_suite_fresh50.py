from __future__ import annotations

import json
from pathlib import Path


SUITE_PATH = Path(__file__).with_name("benchmark_suite_fresh50.json")


def _load_suite() -> dict:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def test_fresh50_suite_has_expected_shape_and_unique_ids():
    payload = _load_suite()
    cases = payload["cases"]
    ids = [item["case_id"] for item in cases]

    assert payload["suite"] == "fresh50"
    assert len(cases) == 50
    assert len(set(ids)) == 50
    assert all(item["title"] for item in cases)
    assert all(item["patient_text"] for item in cases)


def test_fresh50_suite_has_mixed_language_and_difficulty_coverage():
    cases = _load_suite()["cases"]
    turkish_markers = (" yasinda", "bogaz", "ates", "agri", "nefes", "bulanti", "geldi")
    english_markers = ("year old", "shortness of breath", "blood pressure", "room air", "worst headache")

    turkish_case_count = sum(
        1
        for item in cases
        if any(marker in f"{item['title']} {item['patient_text']}".lower() for marker in turkish_markers)
    )
    english_case_count = sum(
        1
        for item in cases
        if any(marker in f"{item['title']} {item['patient_text']}".lower() for marker in english_markers)
    )
    difficulties = [int(item["difficulty"]) for item in cases]
    families = {item["disease_family"] for item in cases}

    assert turkish_case_count >= 15
    assert english_case_count >= 15
    assert min(difficulties) <= 2
    assert max(difficulties) >= 9
    assert sum(1 for value in difficulties if value <= 3) >= 8
    assert sum(1 for value in difficulties if value >= 7) >= 20
    assert len(families) >= 15


def test_fresh50_suite_expectations_are_populated():
    cases = _load_suite()["cases"]

    assert all(item.get("expectations", {}).get("keyword_hits") for item in cases)
    assert all(item.get("expectations", {}).get("status_hits") for item in cases)
