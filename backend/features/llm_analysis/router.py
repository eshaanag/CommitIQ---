"""FastAPI router for LLM narrative generation (Issue #394).

Endpoints
---------
- ``POST /api/explain``            non-streaming narrative (cached or fresh)
- ``POST /api/predict``            non-streaming merge-impact narrative
- ``POST /api/explain/stream``     SSE token-by-token narrative stream
- ``POST /api/predict/stream``     SSE token-by-token merge-impact stream

The streaming endpoints emit ``text/event-stream`` payloads of the form::

    data: {"token": "<chunk>", "done": false}\\n\\n
    data: {"done": true, "explanation": "...", "tokens_total": 42, ...}\\n\\n

Both cache-hit and live-provider paths stream the response so the frontend
UI always renders a typing animation regardless of the underlying source.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.features.llm_analysis.cache import (
    get_cached_narrative,
    make_cache_key,
    set_cached_narrative,
)
from backend.features.llm_analysis.claude_client import get_or_create_narrative
from backend.features.llm_analysis.cost_guard import check_budget, estimate_cost_usd
from backend.features.llm_analysis.llm_router import (
    LLMProvider,
    model_for_provider,
    stream_narrative,
)
from backend.features.llm_analysis.prompt_builder import (
    EXPLAIN_DROP_SYSTEM,
    build_explain_prompt,
    build_predict_prompt,
)
from backend.shared.models import Commit, HealthSnapshot, LLMNarrative
from backend.shared.schemas import NarrativeRequest, NarrativeResponse, PredictRequest

router = APIRouter(prefix="", tags=["llm"])
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _build_demo_narrative(commit_message: str, before: dict, after: dict) -> str:
    """Produce a deterministic fallback narrative when all LLM providers fail.

    No external API is called - this lets the UI show something useful even
    in demo mode (no API keys configured) or during outages.
    """
    health_delta = float(after.get("health_score", 0) or 0) - float(
        before.get("health_score", 0) or 0
    )
    complexity_delta = float(after.get("avg_complexity", 0) or 0) - float(
        before.get("avg_complexity", 0) or 0
    )
    risk_level = (
        "High"
        if health_delta <= -15 or after.get("bus_factor_min", 1) <= 1
        else "Medium" if health_delta < 0 else "Low"
    )
    top_files = after.get("top_files_json") or "[]"

    return (
        "DEMO MODE: Configure ANTHROPIC_API_KEY or GEMINI_API_KEY for provider-backed narratives.\n"
        f"- Commit: {(commit_message or 'No commit message')[:90]}\n"
        f"- Health moved from {before.get('health_score', 0)} to {after.get('health_score', 0)} "
        f"({health_delta:+.1f}).\n"
        f"- Average complexity changed by {complexity_delta:+.2f}; churn is "
        f"{after.get('churn_rate', 0)} across {after.get('num_files_changed', 0)} changed files.\n"
        f"- Bus factor minimum is {after.get('bus_factor_min', 1)}. "
        f"Top changed-file metrics: {top_files[:220]}.\n"
        f"Risk level: {risk_level}"
    )


def _map_error(exc: Exception) -> HTTPException:
    """Translate an internal exception into an HTTP error response."""
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=503, detail=f"LLM call failed: {str(exc)[:200]}")


async def _resolve_commit(db: AsyncSession, repo_id: int, commit_sha: str) -> Commit:
    """Look up a commit by short or full SHA, raising 404 if not found."""
    result = await db.execute(
        select(Commit)
        .where(
            Commit.repo_id == repo_id,
            (Commit.sha == commit_sha[:12]) | (Commit.full_sha == commit_sha),
        )
        .limit(1)
    )
    commit = result.scalar_one_or_none()
    if not commit:
        raise HTTPException(status_code=404, detail=f"Commit {commit_sha} not found")
    return commit


async def _build_before_after(
    db: AsyncSession, repo_id: int, commit: Commit
) -> tuple[dict, dict, HealthSnapshot]:
    """Build the before/after metric dicts that feed the prompt builder."""
    snapshot_result = await db.execute(
        select(HealthSnapshot).where(HealthSnapshot.commit_id == commit.id)
    )
    snapshot = snapshot_result.scalar_one_or_none()
    if not snapshot:
        raise HTTPException(status_code=404, detail="Health snapshot not found")

    prev_commit_result = await db.execute(
        select(Commit)
        .where(Commit.repo_id == repo_id, Commit.committed_at < commit.committed_at)
        .order_by(Commit.committed_at.desc())
        .limit(1)
    )
    prev_commit = prev_commit_result.scalar_one_or_none()
    prev_snapshot = None
    if prev_commit:
        prev_snapshot_result = await db.execute(
            select(HealthSnapshot).where(HealthSnapshot.commit_id == prev_commit.id)
        )
        prev_snapshot = prev_snapshot_result.scalar_one_or_none()

    before = {
        "health_score": prev_snapshot.health_score if prev_snapshot else 0,
        "avg_complexity": prev_snapshot.avg_complexity if prev_snapshot else 0,
        "bus_factor_min": prev_snapshot.bus_factor_min if prev_snapshot else 1,
        "avg_semantic_drift": prev_snapshot.avg_semantic_drift if prev_snapshot else 0,
        "semantic_health_score": prev_snapshot.semantic_health_score if prev_snapshot else 100,
    }
    after = {
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
    return before, after, snapshot


def _sse(payload: dict) -> str:
    """Format a dict as a single SSE ``data:`` line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _replay_cached_stream(
    cached_text: str,
    model_used: str,
    tokens_total: int,
) -> StreamingResponse:
    """Stream an already-cached narrative word-by-word to preserve the typing UX."""

    async def replay():
        for word in cached_text.split(" "):
            yield _sse({"token": word + " ", "done": False})
            await asyncio.sleep(0.03)
        yield _sse(
            {
                "done": True,
                "explanation": cached_text,
                "tokens_total": tokens_total,
                "cost_usd": 0.0,
                "cached": True,
                "model": model_used,
                "provider": "cache",
                "demo_mode": False,
            }
        )

    return StreamingResponse(
        replay(),
        media_type="text/event-stream",
        headers={"X-LLM-Provider": "cache", "X-Cache-Hit": "true", "X-LLM-Cost-USD": "0.0000"},
    )


async def _generate_live_stream(
    *,
    repo_id: int,
    commit: Commit,
    prompt: str,
    system_prompt_header: str,
    prompt_type: str,
    cache_key: str,
    before: dict,
    after: dict,
    request_repo_id: int,
) -> StreamingResponse:
    """Stream tokens live from the LLM provider, persisting the result on completion."""

    # Imported lazily to avoid a circular import at module load time.
    from backend.database import AsyncSessionLocal

    async def event_generator():
        full_text: list[str] = []
        provider_used = LLMProvider.NONE
        try:
            async for chunk, provider in stream_narrative(prompt):
                full_text.append(chunk)
                provider_used = provider
                yield _sse({"token": chunk, "done": False})

            response_text = "".join(full_text)
            tokens_in = int(len((system_prompt_header + prompt).split()) * 1.3)
            tokens_out = int(len(response_text.split()) * 1.3)
            provider_value = provider_used.value
            cost = estimate_cost_usd(tokens_in, tokens_out, provider_value)
            model_used = model_for_provider(provider_used)

            async with AsyncSessionLocal() as local_db:
                narrative = LLMNarrative(
                    repo_id=repo_id,
                    commit_id=commit.id,
                    full_sha=commit.full_sha,
                    prompt_type=prompt_type,
                    cache_key=cache_key,
                    prompt_input=prompt,
                    response_text=response_text,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    cost_usd=cost,
                    model_used=model_used,
                )
                local_db.add(narrative)
                await local_db.commit()
            # Also seed Redis so the next request hits cache fast.
            await set_cached_narrative(cache_key, response_text)

            yield _sse(
                {
                    "done": True,
                    "explanation": response_text,
                    "tokens_total": tokens_in + tokens_out,
                    "cost_usd": cost,
                    "cached": False,
                    "model": model_used,
                    "provider": provider_value,
                    "demo_mode": False,
                }
            )
        except Exception as exc:
            logger.warning("Narrative stream provider unavailable, using demo mode: %s", exc)
            response_text = _build_demo_narrative(commit.message or "", before, after)
            tokens_in = int(len((system_prompt_header + prompt).split()) * 1.3)
            tokens_out = int(len(response_text.split()) * 1.3)
            async with AsyncSessionLocal() as local_db:
                narrative = LLMNarrative(
                    repo_id=repo_id,
                    commit_id=commit.id,
                    full_sha=commit.full_sha,
                    prompt_type=prompt_type,
                    cache_key=cache_key,
                    prompt_input=prompt,
                    response_text=response_text,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    cost_usd=0.0,
                    model_used="demo-mode",
                    is_pre_cached=False,
                )
                local_db.add(narrative)
                await local_db.commit()
            yield _sse(
                {
                    "done": True,
                    "explanation": response_text,
                    "tokens_total": tokens_in + tokens_out,
                    "cost_usd": 0.0,
                    "cached": False,
                    "model": "demo-mode",
                    "provider": LLMProvider.NONE.value,
                    "demo_mode": True,
                }
            )

    # Reference request_repo_id to satisfy lint of unused-looking param (kept for callers
    # that pass it explicitly). It is intentionally unused inside this helper.
    _ = request_repo_id

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ────────────────────────────────────────────────────────────────────
# Non-streaming endpoints (existing, unchanged behaviour)
# ────────────────────────────────────────────────────────────────────


@router.post("/explain", response_model=NarrativeResponse)
async def explain_commit(
    request: NarrativeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate (or fetch from cache) an explanation for a commit's health delta."""
    try:
        return await get_or_create_narrative(
            repo_id=request.repo_id,
            commit_sha=request.commit_sha,
            prompt_type=request.prompt_type,
            db=db,
        )
    except Exception as exc:
        raise _map_error(exc)


@router.post("/predict", response_model=NarrativeResponse)
async def predict_merge(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate (or fetch from cache) a merge-impact prediction narrative."""
    try:
        return await get_or_create_narrative(
            repo_id=request.repo_id,
            commit_sha=request.commit_sha,
            prompt_type="predict_merge",
            db=db,
        )
    except Exception as exc:
        raise _map_error(exc)


# ────────────────────────────────────────────────────────────────────
# Streaming endpoints (SSE) - Issue #394
# ────────────────────────────────────────────────────────────────────


@router.post("/explain/stream")
async def explain_commit_stream(
    request: NarrativeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream an ``explain_drop`` narrative token-by-token via Server-Sent Events.

    Flow:
      1. Resolve commit by SHA (404 if missing).
      2. If a Redis or DB cache entry exists, replay it word-by-word.
      3. If the repo's LLM budget is exhausted, return HTTP 429.
      4. Otherwise stream live tokens from Claude/Gemini, then persist
         the resulting narrative.
    """
    commit = await _resolve_commit(db, request.repo_id, request.commit_sha)
    cache_key = make_cache_key(request.repo_id, commit.full_sha, request.prompt_type)

    # 1. Redis cache fast-path
    redis_cached_text = await get_cached_narrative(cache_key)
    if redis_cached_text:
        return await _replay_cached_stream(redis_cached_text, "redis-cache", 0)

    # 2. DB cache
    cached_result = await db.execute(
        select(LLMNarrative).where(LLMNarrative.cache_key == cache_key)
    )
    cached = cached_result.scalar_one_or_none()
    if cached:
        await set_cached_narrative(cache_key, cached.response_text)
        return await _replay_cached_stream(
            cached.response_text,
            cached.model_used,
            cached.tokens_input + cached.tokens_output,
        )

    # 3. Budget guard
    if not await check_budget(request.repo_id, db):
        raise HTTPException(
            status_code=429, detail="LLM budget exceeded for this repo. Cache will be used."
        )

    # 4. Live stream
    before, after, _ = await _build_before_after(db, request.repo_id, commit)
    prompt = build_explain_prompt(before, after, commit.message or "")
    return await _generate_live_stream(
        repo_id=request.repo_id,
        commit=commit,
        prompt=prompt,
        system_prompt_header=EXPLAIN_DROP_SYSTEM,
        prompt_type=request.prompt_type,
        cache_key=cache_key,
        before=before,
        after=after,
        request_repo_id=request.repo_id,
    )


@router.post("/predict/stream")
async def predict_merge_stream(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream a ``predict_merge`` narrative token-by-token via Server-Sent Events.

    Mirrors :func:`explain_commit_stream` but builds the prompt from the
    branch-vs-main metrics comparison via :func:`build_predict_prompt`.
    """
    commit = await _resolve_commit(db, request.repo_id, request.commit_sha)
    cache_key = make_cache_key(request.repo_id, commit.full_sha, "predict_merge")

    redis_cached_text = await get_cached_narrative(cache_key)
    if redis_cached_text:
        return await _replay_cached_stream(redis_cached_text, "redis-cache", 0)

    cached_result = await db.execute(
        select(LLMNarrative).where(LLMNarrative.cache_key == cache_key)
    )
    cached = cached_result.scalar_one_or_none()
    if cached:
        await set_cached_narrative(cache_key, cached.response_text)
        return await _replay_cached_stream(
            cached.response_text,
            cached.model_used,
            cached.tokens_input + cached.tokens_output,
        )

    if not await check_budget(request.repo_id, db):
        raise HTTPException(
            status_code=429, detail="LLM budget exceeded for this repo. Cache will be used."
        )

    before, after, _ = await _build_before_after(db, request.repo_id, commit)
    prompt = build_predict_prompt(after, before)
    return await _generate_live_stream(
        repo_id=request.repo_id,
        commit=commit,
        prompt=prompt,
        system_prompt_header=EXPLAIN_DROP_SYSTEM,  # tokens estimate only; provider has its own
        prompt_type="predict_merge",
        cache_key=cache_key,
        before=before,
        after=after,
        request_repo_id=request.repo_id,
    )
