import asyncio

from src.cdss.contracts.models import PatientInput


def run_async(coro):
    return asyncio.run(coro)


def collect_async(async_iterable):
    async def _inner():
        return [item async for item in async_iterable]

    return asyncio.run(_inner())


def make_high_risk_input() -> PatientInput:
    return PatientInput(
        case_id="case-high-risk",
        patient_text=(
            "12 year old with vomiting, abdominal pain, thirst, polyuria, deep breathing, "
            "lethargy, and difficulty breathing for 2 days."
        ),
        language="en",
        demographics={"age": 12},
        vitals={
            "heart_rate": 142,
            "respiratory_rate": 34,
            "spo2": 90,
            "sbp": 88,
            "temperature_c": 37.2,
        },
        labs={"glucose": 325, "ketones": "positive"},
        medications=[],
    )


def make_stable_metabolic_input() -> PatientInput:
    return PatientInput(
        case_id="case-stable-metabolic",
        patient_text="Vomiting and abdominal pain for 2 days with increasing thirst.",
        language="en",
        demographics={"age": 24},
        vitals={
            "heart_rate": 98,
            "respiratory_rate": 18,
            "spo2": 98,
            "sbp": 118,
            "temperature_c": 37.0,
        },
        labs={"glucose": 118, "anion_gap": 9},
        medications=[],
    )


def make_acs_high_risk_input() -> PatientInput:
    return PatientInput(
        case_id="case-acs-high-risk",
        patient_text=(
            "56 year old male with sudden crushing chest pain radiating to left arm, diaphoresis, "
            "dyspnea, and near-syncope for 45 minutes. Clinician considers acute coronary syndrome."
        ),
        language="en",
        demographics={"age": 56, "sex": "male"},
        vitals={
            "heart_rate": 118,
            "respiratory_rate": 24,
            "spo2": 94,
            "sbp": 92,
            "dbp": 58,
            "temperature_c": 36.9,
        },
        labs={"troponin_i": "elevated", "lactate": 3.1},
        medications=[],
    )
