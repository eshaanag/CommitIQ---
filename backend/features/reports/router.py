"""
backend/features/reports/router.py

FastAPI router for the unified PDF report export (Issue #389).

Endpoint:
    GET /api/repos/{repo_id}/report
        Returns a PDF file with aggregated DORA, Cycle Time, and
        Team Health metrics.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.features.reports.deployment_service import get_deployment_timeline
from backend.features.reports.pdf_service import generate_health_report
from backend.shared.models import Repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


@router.get("/repos/{repo_id}/report")
async def get_health_report(
    repo_id: int,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Download a unified PDF report for Developer Health metrics.

    Aggregates DORA, Cycle Time, and Team Health into a single PDF
    suitable for sharing with stakeholders.
    """
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        )

    try:
        pdf_bytes, filename = await generate_health_report(db, repo_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found.",
        ) from exc
    except RuntimeError as exc:
        logger.warning("ReportLab dependency missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="PDF generation is unavailable because reportlab is not installed.",
        ) from exc
    except Exception:
        logger.exception("Failed to generate PDF report")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF report.",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers=headers,
    )


@router.get("/repos/{repo_id}/deployments")
async def get_deployments(
    repo_id: int,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return deployment timeline and summary stats."""
    try:
        return await get_deployment_timeline(db, repo_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
