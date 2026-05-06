from src.llm.llama_cpp_client import _coerce_text_block, _extract_reasoning_text


def test_extract_reasoning_text_reads_reasoning_content_fields():
    message = {
        "content": "",
        "reasoning": "deliberate intermediate trace",
    }

    assert _extract_reasoning_text(message) == "deliberate intermediate trace"


def test_coerce_text_block_joins_string_lists():
    value = ["line1", "line2", 7]

    assert _coerce_text_block(value) == "line1\nline2"
