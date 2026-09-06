"""FastAPI router for scheduled health report management.

Endpoints allow creating, listing, updating, deleting, and manually
triggering scheduled reports with full delivery history tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.features.reports.scheduler_service import (
    compute_next_run,
    describe_cron,
    execute_scheduled_report,
    generate_report_payload,
    validate_cron_expression,
)
from backend.shared.models import Repo, ReportDelivery, ReportSchedule

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report-schedules", tags=["report-schedules"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ReportScheduleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    cron_expression: str = Field(..., min_length=5, max_length=100)
    timezone: str = Field(default="UTC", max_length=50)
    report_type: str = Field(default="health_summary")
    webhook_url: str | None = None
    webhook_secret: str | None = None
    notification_email: str | None = None
    include_narrative: bool = False

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        allowed = {"health_summary", "dora_metrics", "team_health", "full_analysis"}
        if v not in allowed:
            raise ValueError(f"report_type must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        ok, msg = validate_cron_expression(v)
        if not ok:
            raise ValueError(msg)
        return v


class ReportScheduleUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    cron_expression: str | None = Field(None, min_length=5, max_length=100)
    timezone: str | None = None
    report_type: str | None = None
    is_active: bool | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    notification_email: str | None = None
    include_narrative: bool | None = None

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v: str | None) -> str | None:
        if v is None:
            return v
        ok, msg = validate_cron_expression(v)
        if not ok:
            raise ValueError(msg)
        return v

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"health_summary", "dora_metrics", "team_health", "full_analysis"}
        if v not in allowed:
            raise ValueError(f"report_type must be one of: {', '.join(sorted(allowed))}")
        return v


class ScheduleResponse(BaseModel):
    id: int
    repo_id: int
    name: str
    description: str | None
    cron_expression: str
    cron_description: str
    timezone: str
    report_type: str
    is_active: bool
    webhook_url: str | None
    notification_email: str | None
    include_narrative: bool
    last_run_at: str | None
    next_run_at: str | None
    last_delivery_status: str | None
    consecutive_failures: int
    max_retry_count: int
    created_at: str | None
    updated_at: str | None


class DeliveryResponse(BaseModel):
    id: int
    schedule_id: int
    repo_id: int
    status: str
    report_type: str
    triggered_at: str | None
    completed_at: str | None
    duration_seconds: float | None
    webhook_status_code: int | None
    error_message: str | None
    retry_count: int
    snapshot_health_score: float | None
    snapshot_commits_analyzed: int | None
    snapshot_latest_sha: str | None


class TriggerResponse(BaseModel):
    delivery_id: int
    status: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_schedule(s: ReportSchedule) -> dict[str, Any]:
    return {
        "id": s.id,
        "repo_id": s.repo_id,
        "name": s.name,
        "description": s.description,
        "cron_expression": s.cron_expression,
        "cron_description": describe_cron(s.cron_expression),
        "timezone": s.timezone,
        "report_type": s.report_type,
        "is_active": s.is_active,
        "webhook_url": s.webhook_url,
        "notification_email": s.notification_email,
        "include_narrative": s.include_narrative,
        "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
        "next_run_at": s.next_run_at.isoformat() if s.next_run_at else None,
        "last_delivery_status": s.last_delivery_status,
        "consecutive_failures": s.consecutive_failures,
        "max_retry_count": s.max_retry_count,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _serialize_delivery(d: ReportDelivery) -> dict[str, Any]:
    return {
        "id": d.id,
        "schedule_id": d.schedule_id,
        "repo_id": d.repo_id,
        "status": d.status,
        "report_type": d.report_type,
        "triggered_at": d.triggered_at.isoformat() if d.triggered_at else None,
        "completed_at": d.completed_at.isoformat() if d.completed_at else None,
        "duration_seconds": d.duration_seconds,
        "webhook_status_code": d.webhook_status_code,
        "error_message": d.error_message,
        "retry_count": d.retry_count,
        "snapshot_health_score": d.snapshot_health_score,
        "snapshot_commits_analyzed": d.snapshot_commits_analyzed,
        "snapshot_latest_sha": d.snapshot_latest_sha,
    }


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/repos/{repo_id}/schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    repo_id: int,
    body: ReportScheduleCreate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a new scheduled health report for a repository."""
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    next_run = compute_next_run(body.cron_expression, body.timezone)

    schedule = ReportSchedule(
        repo_id=repo_id,
        name=body.name,
        description=body.description,
        cron_expression=body.cron_expression,
        timezone=body.timezone,
        report_type=body.report_type,
        is_active=True,
        webhook_url=body.webhook_url,
        webhook_secret=body.webhook_secret,
        notification_email=body.notification_email,
        include_narrative=body.include_narrative,
        next_run_at=next_run,
    )
    db.add(schedule)
    await db.flush()

    logger.info(
        "Created report schedule %d for repo %d (%s), next run: %s",
        schedule.id,
        repo_id,
        body.name,
        next_run.isoformat(),
    )
    return _serialize_schedule(schedule)


@router.get("/repos/{repo_id}/schedules")
async def list_schedules(
    repo_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List all report schedules for a repository."""
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    stmt = (
        select(ReportSchedule)
        .where(ReportSchedule.repo_id == repo_id)
        .order_by(desc(ReportSchedule.created_at))
    )
    result = await db.execute(stmt)
    schedules = result.scalars().all()
    return [_serialize_schedule(s) for s in schedules]


@router.get("/repos/{repo_id}/schedules/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    repo_id: int,
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get a single report schedule by ID."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return _serialize_schedule(schedule)


@router.patch("/repos/{repo_id}/schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    repo_id: int,
    schedule_id: int,
    body: ReportScheduleUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update an existing report schedule."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    update_data = body.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(schedule, field_name, value)

    # Recompute next_run_at if cron changed
    if "cron_expression" in update_data or "timezone" in update_data:
        schedule.next_run_at = compute_next_run(
            schedule.cron_expression, schedule.timezone, after=datetime.now(timezone.utc)
        )

    logger.info("Updated report schedule %d: %s", schedule_id, list(update_data.keys()))
    return _serialize_schedule(schedule)


@router.delete("/repos/{repo_id}/schedules/{schedule_id}", status_code=204)
async def delete_schedule(
    repo_id: int,
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a report schedule and all associated delivery records."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")
    await db.delete(schedule)
    logger.info("Deleted report schedule %d for repo %d", schedule_id, repo_id)


@router.post("/repos/{repo_id}/schedules/{schedule_id}/toggle", response_model=ScheduleResponse)
async def toggle_schedule(
    repo_id: int,
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Toggle a report schedule between active and paused."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule.is_active = not schedule.is_active
    if schedule.is_active and schedule.next_run_at is None:
        schedule.next_run_at = compute_next_run(schedule.cron_expression, schedule.timezone)

    state = "activated" if schedule.is_active else "paused"
    logger.info("Schedule %d %s for repo %d", schedule_id, state, repo_id)
    return _serialize_schedule(schedule)


# ---------------------------------------------------------------------------
# Trigger & delivery history
# ---------------------------------------------------------------------------


@router.post(
    "/repos/{repo_id}/schedules/{schedule_id}/trigger",
    response_model=TriggerResponse,
)
async def trigger_schedule(
    repo_id: int,
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Manually trigger an immediate execution of a scheduled report."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    delivery = await execute_scheduled_report(db, schedule_id)
    await db.commit()

    return {
        "delivery_id": delivery.id,
        "status": delivery.status,
        "message": f"Report {'delivered' if delivery.status == 'success' else 'execution failed'} "
        f"(delivery #{delivery.id})",
    }


@router.get("/repos/{repo_id}/schedules/{schedule_id}/deliveries")
async def list_deliveries(
    repo_id: int,
    schedule_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List delivery history for a specific schedule with pagination."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    count_stmt = select(sa_func.count(ReportDelivery.id)).where(
        ReportDelivery.schedule_id == schedule_id
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = (
        select(ReportDelivery)
        .where(ReportDelivery.schedule_id == schedule_id)
        .order_by(desc(ReportDelivery.triggered_at))
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    deliveries = result.scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "deliveries": [_serialize_delivery(d) for d in deliveries],
    }


@router.post(
    "/repos/{repo_id}/schedules/{schedule_id}/deliveries/{delivery_id}/retry",
    response_model=TriggerResponse,
)
async def retry_delivery(
    repo_id: int,
    schedule_id: int,
    delivery_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retry a failed delivery by re-executing the schedule."""
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule or schedule.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    original = await db.get(ReportDelivery, delivery_id)
    if not original or original.schedule_id != schedule_id:
        raise HTTPException(status_code=404, detail="Delivery not found")

    if original.status != "failed":
        raise HTTPException(status_code=400, detail="Only failed deliveries can be retried")

    delivery = await execute_scheduled_report(db, schedule_id)
    await db.commit()

    return {
        "delivery_id": delivery.id,
        "status": delivery.status,
        "message": f"Retry {'succeeded' if delivery.status == 'success' else 'failed'} "
        f"(new delivery #{delivery.id})",
    }


@router.get("/repos/{repo_id}/deliveries/recent")
async def recent_deliveries(
    repo_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get the most recent deliveries across all schedules for a repo."""
    stmt = (
        select(ReportDelivery)
        .where(ReportDelivery.repo_id == repo_id)
        .order_by(desc(ReportDelivery.triggered_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    deliveries = result.scalars().all()
    return [_serialize_delivery(d) for d in deliveries]


@router.get("/repos/{repo_id}/reports/preview")
async def preview_report(
    repo_id: int,
    report_type: str = Query(default="health_summary"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Preview what a report would look like without scheduling it."""
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    payload = await generate_report_payload(db, repo_id, report_type)
    return payload
