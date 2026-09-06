"""LLM call wrapper for on-demand CommitIQ narratives."""

from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import ANTHROPIC_API_KEY
from backend.features.llm_analysis.cache import make_cache_key
from backend.features.llm_analysis.cost_guard import check_budget, estimate_cost_usd
from backend.features.llm_analysis.llm_router import get_narrative_non_streaming, model_for_provider
from backend.features.llm_analysis.prompt_builder import (
    EXPLAIN_DROP_SYSTEM,
    PREDICT_MERGE_SYSTEM,
    build_explain_prompt,
    build_predict_prompt,
)
from backend.shared.models import Commit, HealthSnapshot, LLMNarrative

# Sonnet pricing (as of 2025):
# Input:  $3.00 per 1M tokens = $0.000003 per token
# Output: $15.00 per 1M tokens = $0.000015 per token
INPUT_COST_PER_TOKEN = 0.000003
OUTPUT_COST_PER_TOKEN = 0.000015
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"


def _get_anthropic():
    try:
        import anthropic

        return anthropic
    except ImportError:
        return None


def _get_genai():
    try:
        import google.generativeai as genai

        return genai
    except ImportError:
        return None


def _sync_anthropic_client():
    anthropic = _get_anthropic()
    if not anthropic or not ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _async_anthropic_client():
    anthropic = _get_anthropic()
    if not anthropic or not ANTHROPIC_API_KEY:
        return None
    return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def stream_narrative(prompt: str) -> AsyncGenerator[str, None]:
    """Stream text chunks from Claude. Raises when Anthropic is unavailable."""
    client = _async_anthropic_client()
    if client is None:
        raise RuntimeError("anthropic not installed or ANTHROPIC_API_KEY is not configured.")

    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are a code health analyst. Given metric deltas for a Git commit, "
            "explain in plain English why the health score changed. Be specific about "
            "which files and metrics drove the change. Use bullet points. Max 4 sentences. "
            'Always end with: "Estimated impact: [Low/Medium/High] risk to next sprint." '
            "Never mention token costs or that you're an AI."
        ),
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def generate_narrative(prompt: str) -> str:
    chunks = []
    async for chunk in stream_narrative(prompt):
        chunks.append(chunk)
    return "".join(chunks)


async def get_or_create_narrative(
    repo_id: int,
    commit_sha: str,
    prompt_type: str,
    db: AsyncSession,
) -> dict:
    """
    Main entry point. Returns narrative dict.
    Checks cache → checks budget → fires API → stores result.
    """
    commit_result = await db.execute(
        select(Commit)
        .where(
            Commit.repo_id == repo_id,
            Commit.sha == commit_sha[:12],
        )
        .limit(1)
    )
    commit = commit_result.scalar_one_or_none()
    if not commit:
        raise ValueError(f"Commit {commit_sha} not found in repo {repo_id}")

    cache_key = make_cache_key(repo_id, commit.full_sha, prompt_type)

    # 1.5. Check Redis cache first
    from backend.features.llm_analysis.cache import get_cached_narrative, set_cached_narrative

    redis_cached = await get_cached_narrative(cache_key)
    if redis_cached:
        return {
            "repo_id": repo_id,
            "commit_sha": commit.sha,
            "prompt_type": prompt_type,
            "explanation": redis_cached,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "cached": True,
            "model": "redis-cache",
            "provider": "cache",
            "demo_mode": False,
        }

    # 2. Check Postgres cache
    cached_result = await db.execute(
        select(LLMNarrative).where(LLMNarrative.cache_key == cache_key)
    )
    cached = cached_result.scalar_one_or_none()
    if cached:
        # Backfill Redis
        await set_cached_narrative(cache_key, cached.response_text)
        return {
            "repo_id": repo_id,
            "commit_sha": commit.sha,
            "prompt_type": prompt_type,
            "explanation": cached.response_text,
            "tokens_used": cached.tokens_input + cached.tokens_output,
            "cost_usd": cached.cost_usd,
            "cached": True,
            "model": cached.model_used,
            "provider": "cache",
            "demo_mode": False,
        }

    # 3. Check budget — hard limit

    if not await check_budget(repo_id, db):
        raise PermissionError("LLM budget exhausted for this repository.")

    snapshot_result = await db.execute(
        select(HealthSnapshot).where(HealthSnapshot.commit_id == commit.id)
    )
    snapshot = snapshot_result.scalar_one_or_none()
    if not snapshot:
        raise ValueError(f"No health snapshot for commit {commit_sha}")

    after_dict = {
        "health_score": snapshot.health_score,
        "avg_complexity": snapshot.avg_complexity,
        "churn_rate": snapshot.churn_rate,
        "num_files_changed": snapshot.num_files_changed,
        "bus_factor_min": snapshot.bus_factor_min,
        "top_files_json": snapshot.top_files_json,
        "avg_semantic_drift": snapshot.avg_semantic_drift,
        "semantic_health_score": snapshot.semantic_health_score,
        "high_drift_files": snapshot.high_drift_files,
        "semantic_drift_method": snapshot.semantic_drift_method,
    }

    prev_commit_result = await db.execute(
        select(Commit)
        .where(
            Commit.repo_id == repo_id,
            Commit.committed_at < commit.committed_at,
        )
        .order_by(Commit.committed_at.desc())
        .limit(1)
    )
    prev_commit = prev_commit_result.scalar_one_or_none()

    if prev_commit:
        prev_snap_result = await db.execute(
            select(HealthSnapshot).where(HealthSnapshot.commit_id == prev_commit.id)
        )
        prev_snap = prev_snap_result.scalar_one_or_none()
        before_dict = {
            "health_score": prev_snap.health_score if prev_snap else 0,
            "avg_complexity": prev_snap.avg_complexity if prev_snap else 0,
            "bus_factor_min": prev_snap.bus_factor_min if prev_snap else 1,
        }
    else:
        before_dict = {"health_score": 0, "avg_complexity": 0, "bus_factor_min": 1}

    if prompt_type == "explain_drop":
        system_prompt = EXPLAIN_DROP_SYSTEM
        user_prompt = build_explain_prompt(before_dict, after_dict, commit.message or "")
    else:
        system_prompt = PREDICT_MERGE_SYSTEM
        user_prompt = build_predict_prompt(after_dict, before_dict)

    # 5. Fire the API call through Claude-first router with Gemini fallback.
    response_text = ""
    tokens_in = int(len((system_prompt + user_prompt).split()) * 1.3)
    tokens_out = 0
    cost = 0.0
    model_used = ""
    provider_used = "none"
    demo_mode = False

    try:
        response_text, provider = await get_narrative_non_streaming(user_prompt, max_tokens=600)
        provider_used = provider.value
        model_used = model_for_provider(provider)
        tokens_out = int(len(response_text.split()) * 1.3)
        cost = estimate_cost_usd(tokens_in, tokens_out, provider_used)
    except Exception:
        model_used = "demo-mode"
        demo_mode = True

    if demo_mode:
        response_text = (
            f"DEMO MODE: LLM API keys are not configured or calls failed. "
            f"Preview narrative for commit '{(commit.message or '')[:40]}...': "
            f"Health score is {after_dict['health_score']} (delta {after_dict['health_score'] - before_dict['health_score']:.1f}). "
            f"The codebase average complexity is {after_dict['avg_complexity']} (previously {before_dict['avg_complexity']}). "
            f"Please configure ANTHROPIC_API_KEY or GEMINI_API_KEY in your .env file."
        )

    # 6. Store in cache — immutable once written
    narrative = LLMNarrative(
        repo_id=repo_id,
        commit_id=commit.id,
        full_sha=commit.full_sha,
        prompt_type=prompt_type,
        cache_key=cache_key,
        prompt_input=user_prompt,
        response_text=response_text,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=round(cost, 6),
        model_used=model_used,
        is_pre_cached=False,
    )
    db.add(narrative)
    await db.commit()

    # Update Redis cache
    if not demo_mode:
        await set_cached_narrative(cache_key, response_text)

    return {
        "repo_id": repo_id,
        "commit_sha": commit.sha,
        "prompt_type": prompt_type,
        "explanation": response_text,
        "tokens_used": tokens_in + tokens_out,
        "cost_usd": round(cost, 5),
        "cached": False,
        "model": model_used,
        "provider": provider_used,
        "demo_mode": demo_mode,
    }
