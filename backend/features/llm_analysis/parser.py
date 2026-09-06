"""Utilities for parsing LLM responses."""

import json
import re
from typing import Any


def parse_llm_json(response_text: str) -> Any:
    """
    Parse a structured JSON explanation from the LLM.
    Uses a regex fallback to gracefully handle malformed JSON wrappers
    (like markdown code blocks) before failing.
    """
    text = response_text.strip()

    # 1. Try a direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try to extract from markdown code blocks (e.g. ```json ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Try to extract the outermost braces or brackets to ignore conversational filler
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 4. If all fail, raise ValueError
    raise ValueError("Failed to parse LLM response as JSON even with regex fallback.")
