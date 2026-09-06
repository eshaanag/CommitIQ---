"""FastAPI router for the actionable health recommendations endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.features.metrics.recommendations import generate_recommendations

router = APIRouter(prefix="/repos", tags=["recommendations"])


@router.get("/{repo_id}/recommendations")
async def get_recommendations(
    repo_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return prioritised health recommendations for *repo_id*."""
    try:
        return await generate_recommendations(db, repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
