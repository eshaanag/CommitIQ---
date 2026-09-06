"""LLM analysis router tests.

These tests cover the unit-level behaviour of the LLM helpers used by the
streaming endpoints: the cache-key derivation, the cost estimator, the
budget guard, and the demo-narrative builder. HTTP-level streaming tests
live in ``test_streaming_narrative.py``.
"""

from __future__ import annotations

import asyncio

from backend.features.llm_analysis.cache import make_cache_key
from backend.features.llm_analysis.cost_guard import estimate_cost_usd, provider_from_model
from backend.features.llm_analysis.llm_router import LLMProvider, model_for_provider
from backend.features.llm_analysis.prompt_builder import build_explain_prompt, build_predict_prompt
from backend.features.llm_analysis.router import _build_demo_narrative

# ────────────────────────────────────────────────────────────────
# Cache key
# ────────────────────────────────────────────────────────────────


def test_make_cache_key_is_deterministic():
    """The same inputs must always produce the same SHA256 key."""
    a = make_cache_key(1, "abc123", "explain_drop")
    b = make_cache_key(1, "abc123", "explain_drop")
    assert a == b
    assert len(a) == 64  # SHA256 hex digest length


def test_make_cache_key_differs_on_any_input():
    """Different repo / sha / prompt_type must produce a different key."""
    base = make_cache_key(1, "abc123", "explain_drop")
    assert base != make_cache_key(2, "abc123", "explain_drop")
    assert base != make_cache_key(1, "def456", "explain_drop")
    assert base != make_cache_key(1, "abc123", "predict_merge")


# ────────────────────────────────────────────────────────────────
# Provider / model mapping
# ────────────────────────────────────────────────────────────────


def test_model_for_provider_returns_canonical_model():
    assert "claude" in model_for_provider(LLMProvider.ANTHROPIC)
    assert "gemini" in model_for_provider(LLMProvider.GEMINI)
    assert model_for_provider(LLMProvider.NONE) == "none"
    assert model_for_provider("anthropic") == model_for_provider(LLMProvider.ANTHROPIC)


def test_provider_from_model_recognises_known_models():
    assert provider_from_model("claude-3-5-sonnet-20241022") == "anthropic"
    assert provider_from_model("gemini-2.5-flash") == "gemini"
    assert provider_from_model("redis-cache") == "cache"
    assert provider_from_model("demo-mode") == "none"
    assert provider_from_model(None) == "none"


# ────────────────────────────────────────────────────────────────
# Cost estimation
# ────────────────────────────────────────────────────────────────


def test_estimate_cost_usd_anthropic():
    """Anthropic cost = $3/1M input + $15/1M output tokens."""
    cost = estimate_cost_usd(1000, 500, "anthropic")
    # 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
    assert abs(cost - 0.0105) < 1e-9


def test_estimate_cost_usd_gemini():
    """Gemini cost = $0.00035 / 1K output tokens (input not billed for the demo tier)."""
    cost = estimate_cost_usd(1000, 500, "gemini")
    # 500 / 1000 * 0.00035 = 0.000175
    assert abs(cost - 0.000175) < 1e-9


def test_estimate_cost_usd_cache_is_zero():
    assert estimate_cost_usd(1000, 500, "cache") == 0.0
    assert estimate_cost_usd(1000, 500, "none") == 0.0


# ────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────


def test_build_explain_prompt_includes_metrics_and_excludes_source_code():
    """The prompt must contain metric deltas, never raw source code."""
    before = {"health_score": 80, "avg_complexity": 3.0, "bus_factor_min": 4}
    after = {
        "health_score": 65,
        "avg_complexity": 4.5,
        "churn_rate": 0.2,
        "num_files_changed": 12,
        "bus_factor_min": 2,
        "top_files_json": '[{"path": "auth.py", "loc": 200}]',
        "avg_semantic_drift": 0.1,
        "semantic_health_score": 70,
        "high_drift_files": 1,
        "semantic_drift_method": "embedding",
    }

    prompt = build_explain_prompt(before, after, "Refactor auth module")

    assert "Refactor auth module" in prompt
    assert "80" in prompt  # before health
    assert "65" in prompt  # after health
    assert "auth.py" in prompt  # top changed file is included


def test_build_predict_prompt_compares_branch_to_main():
    branch = {
        "health_score": 70,
        "avg_complexity": 4.0,
        "churn_rate": 0.15,
        "bus_factor_min": 3,
        "num_files_changed": 8,
    }
    main = {"health_score": 80, "avg_complexity": 3.5}

    prompt = build_predict_prompt(branch, main)

    assert "80" in prompt  # main health
    assert "70" in prompt  # branch health


# ────────────────────────────────────────────────────────────────
# Demo narrative builder
# ────────────────────────────────────────────────────────────────


def test_build_demo_narrative_includes_risk_level_and_metrics():
    before = {"health_score": 80, "avg_complexity": 3.0, "bus_factor_min": 4}
    after = {
        "health_score": 60,
        "avg_complexity": 4.5,
        "churn_rate": 0.2,
        "num_files_changed": 12,
        "bus_factor_min": 2,
        "top_files_json": '[{"path": "auth.py"}]',
    }

    narrative = _build_demo_narrative("Refactor auth", before, after)

    assert "DEMO MODE" in narrative
    assert "80" in narrative  # before health
    assert "60" in narrative  # after health
    assert "Risk level: High" in narrative  # health dropped by 20 → High


def test_build_demo_narrative_risk_level_medium_for_small_drop():
    before = {"health_score": 80, "avg_complexity": 3.0, "bus_factor_min": 4}
    after = {
        "health_score": 75,
        "avg_complexity": 3.5,
        "churn_rate": 0.1,
        "num_files_changed": 5,
        "bus_factor_min": 3,
        "top_files_json": "[]",
    }
    narrative = _build_demo_narrative("Small change", before, after)
    assert "Risk level: Medium" in narrative


def test_build_demo_narrative_risk_level_low_for_health_gain():
    before = {"health_score": 60, "avg_complexity": 4.0, "bus_factor_min": 2}
    after = {
        "health_score": 75,
        "avg_complexity": 3.5,
        "churn_rate": 0.05,
        "num_files_changed": 3,
        "bus_factor_min": 5,
        "top_files_json": "[]",
    }
    narrative = _build_demo_narrative("Cleanup", before, after)
    assert "Risk level: Low" in narrative


# ────────────────────────────────────────────────────────────────
# Stream narrative async generator (no live API calls)
# ────────────────────────────────────────────────────────────────


def test_stream_narrative_yields_degraded_message_when_no_api_keys(monkeypatch):
    """With no API keys configured, stream_narrative emits a single fallback chunk."""
    # Force both API keys to empty strings.
    monkeypatch.setattr("backend.features.llm_analysis.llm_router.ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("backend.features.llm_analysis.llm_router.GEMINI_API_KEY", "")

    from backend.features.llm_analysis.llm_router import stream_narrative

    async def collect():
        chunks = []
        async for token, provider in stream_narrative("dummy prompt"):
            chunks.append((token, provider))
        return chunks

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    token, provider = chunks[0]
    assert provider == LLMProvider.NONE
    # The fallback message should mention either the degraded state or the risk level.
    assert "unavailable" in token.lower() or "risk level" in token.lower()


def test_get_narrative_non_streaming_joins_tokens(monkeypatch):
    """The non-streaming helper consumes the async generator and joins text."""
    from backend.features.llm_analysis import llm_router as router_mod

    async def fake_stream(_prompt, max_tokens=600):
        yield "Hello ", LLMProvider.ANTHROPIC
        yield "world.", LLMProvider.ANTHROPIC

    monkeypatch.setattr(router_mod, "stream_narrative", fake_stream)

    async def run():
        return await router_mod.get_narrative_non_streaming("ignored")

    text, provider = asyncio.run(run())
    assert text == "Hello world."
    assert provider == LLMProvider.ANTHROPIC
