"""Tests for the LLM JSON response parser."""

from __future__ import annotations

import pytest

from backend.features.llm_analysis.parser import parse_llm_json


def test_parse_llm_json_valid_json():
    """Valid JSON should parse immediately."""
    text = '{"summary": "test", "risk": "low"}'
    result = parse_llm_json(text)
    assert result == {"summary": "test", "risk": "low"}


def test_parse_llm_json_array():
    """Valid JSON array should parse immediately."""
    text = '[{"path": "test.py"}]'
    result = parse_llm_json(text)
    assert result == [{"path": "test.py"}]


def test_parse_llm_json_with_markdown_wrapper():
    """JSON inside a markdown code block should be extracted."""
    text = """Here is the analysis:
```json
{"summary": "test", "risk": "high"}
```
Hope this helps!
"""
    result = parse_llm_json(text)
    assert result == {"summary": "test", "risk": "high"}


def test_parse_llm_json_with_generic_code_block():
    """JSON inside a generic code block (no 'json' hint) should be extracted."""
    text = """```
[
  {"path": "file1.py"},
  {"path": "file2.py"}
]
```"""
    result = parse_llm_json(text)
    assert result == [{"path": "file1.py"}, {"path": "file2.py"}]


def test_parse_llm_json_with_leading_trailing_text():
    """JSON with conversational filler but NO code blocks should extract outermost braces."""
    text = """
Sure, here is the result:
{
  "summary": "a complex refactor",
  "risk": "medium"
}
Please review it.
"""
    result = parse_llm_json(text)
    assert result == {"summary": "a complex refactor", "risk": "medium"}


def test_parse_llm_json_invalid_fallback():
    """If neither the raw string nor the extracted block is valid JSON, raise ValueError."""
    # Extracted block is not valid JSON
    text = """```json
{ summary: "missing quotes around keys" }
```"""
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        parse_llm_json(text)


def test_parse_llm_json_no_json_present():
    """If there are no braces or brackets at all, raise ValueError."""
    text = "The commit looks good. No issues found."
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        parse_llm_json(text)
