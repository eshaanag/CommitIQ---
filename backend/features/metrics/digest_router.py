"""FastAPI router for the weekly repository health digest.

Endpoint
--------
GET /api/repos/{repo_id}/digest?weeks=1
    Returns a single JSON payload aggregating all health signals for the
    requested look-back window.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.features.metrics.weekly_digest import compute_weekly_digest

router = APIRouter(prefix="/repos", tags=["digest"])


@router.get("/{repo_id}/digest")
async def get_weekly_digest(
    repo_id: int,
    weeks: int = Query(default=1, ge=1, le=52, description="Look-back window in weeks"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return an aggregated weekly health digest for *repo_id*."""
    try:
        return await compute_weekly_digest(db, repo_id, weeks=weeks)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
