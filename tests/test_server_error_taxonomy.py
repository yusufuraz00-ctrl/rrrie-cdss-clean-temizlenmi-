from gui.server import _api_error_payload, _http_error, _ws_error_payload


def test_api_error_payload_has_structured_fields():
    payload = _api_error_payload(
        "api_vnext_invalid_input",
        "Invalid vNext patient input.",
        incident_id="inc123",
        retryable=False,
        details={"reason": "bad field"},
    )

    assert payload["error_code"] == "api_vnext_invalid_input"
    assert payload["message"] == "Invalid vNext patient input."
    assert payload["incident_id"] == "inc123"
    assert payload["retryable"] is False
    assert payload["details"]["reason"] == "bad field"


def test_http_error_wraps_structured_payload():
    err = _http_error(422, "api_v2_invalid_input", "Invalid v2 patient input.")

    assert err.status_code == 422
    assert err.detail["error_code"] == "api_v2_invalid_input"
    assert err.detail["message"] == "Invalid v2 patient input."


def test_ws_error_payload_has_error_code_and_type():
    payload = _ws_error_payload("ws_invalid_json", "Invalid JSON payload.")

    assert payload["type"] == "error"
    assert payload["error_code"] == "ws_invalid_json"
    assert payload["content"] == "Invalid JSON payload."
