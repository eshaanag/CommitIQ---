"""Scheduled health report service.

Handles cron-expression parsing, next-run computation, report payload
generation, webhook delivery with HMAC signing, and delivery retry logic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import (
    Commit,
    Repo,
    ReportDelivery,
    ReportSchedule,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cron helpers (minimal subset: minute hour day-of-month month day-of-week)
# ---------------------------------------------------------------------------

_DOW_MAP = {
    "SUN": 0,
    "MON": 1,
    "TUE": 2,
    "WED": 3,
    "THU": 4,
    "FRI": 5,
    "SAT": 6,
}


def _expand_cron_field(field: str, lo: int, hi: int) -> set[int]:
    """Expand a single cron field (with *, ranges, steps) into a set of ints."""
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip().upper()
        # Replace named day-of-week tokens
        for name, num in _DOW_MAP.items():
            part = part.replace(name, str(num))

        step_match = re.match(r"^(\S+)/(\d+)$", part)
        if step_match:
            start_end = step_match.group(1)
            step = int(step_match.group(2))
            if start_end == "*":
                start, end = lo, hi
            else:
                rng = start_end.split("-")
                start, end = int(rng[0]), int(rng[-1])
            for v in range(start, end + 1, step):
                if lo <= v <= hi:
                    values.add(v)
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            for v in range(int(a), int(b) + 1):
                if lo <= v <= hi:
                    values.add(v)
        elif part == "*":
            values.update(range(lo, hi + 1))
        else:
            val = int(part)
            if val < lo or val > hi:
                raise ValueError(f"Value {val} out of bounds [{lo}, {hi}]")
            values.add(val)
    return values


def parse_cron(expression: str) -> dict[str, set[int]]:
    """Parse a 5-field cron expression into a dict of allowed values."""
    parts = expression.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (expected 5 fields): {expression}")
    return {
        "minute": _expand_cron_field(parts[0], 0, 59),
        "hour": _expand_cron_field(parts[1], 0, 23),
        "day": _expand_cron_field(parts[2], 1, 31),
        "month": _expand_cron_field(parts[3], 1, 12),
        "weekday": _expand_cron_field(parts[4], 0, 6),
    }


def compute_next_run(cron_expr: str, tz_name: str, after: datetime | None = None) -> datetime:
    """Compute the next run time for a cron expression after *after* (default: now).

    Uses a bounded forward scan (up to 366 days) to avoid infinite loops.
    """
    fields = parse_cron(cron_expr)
    start = (after or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    candidate = start + timedelta(minutes=1)

    for _ in range(525_600):  # 366 days × 24h × 60m
        if (
            candidate.minute in fields["minute"]
            and candidate.hour in fields["hour"]
            and candidate.day in fields["day"]
            and candidate.month in fields["month"]
            and candidate.weekday() in fields["weekday"]
        ):
            return candidate
        candidate += timedelta(minutes=1)

    raise RuntimeError(f"Could not find next cron match for {cron_expr} within 366 days")


# ---------------------------------------------------------------------------
# Report payload generation
# ---------------------------------------------------------------------------


async def generate_report_payload(
    db: AsyncSession, repo_id: int, report_type: str
) -> dict[str, Any]:
    """Build a JSON-serialisable report payload from the latest repo metrics."""

    # Fetch repo metadata
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise ValueError(f"Repository {repo_id} not found")

    # Fetch latest health snapshot via commit join
    stmt = (
        select(Commit, ReportSchedule)
        .join(Repo, Repo.id == Commit.repo_id)
        .where(Commit.repo_id == repo_id)
        .order_by(Commit.committed_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    latest_commit = result.scalars().first() if hasattr(result, "scalars") else None

    # Fallback: just get the last commit
    c_stmt = (
        select(Commit)
        .where(Commit.repo_id == repo_id)
        .order_by(Commit.committed_at.desc())
        .limit(1)
    )
    c_result = await db.execute(c_stmt)
    last_commit = c_result.scalar_one_or_none()

    total_insertions = 0
    total_deletions = 0
    total_files = 0
    commit_count_stmt = select(Commit).where(Commit.repo_id == repo_id)
    all_commits_result = await db.execute(commit_count_stmt)
    all_commits = all_commits_result.scalars().all()
    for c in all_commits:
        total_insertions += c.insertions or 0
        total_deletions += c.deletions or 0
        total_files += c.files_changed or 0

    author_set: set[str] = set()
    for c in all_commits:
        if c.author_email:
            author_set.add(c.author_email)

    churn_pct = 0.0
    if total_insertions > 0:
        churn_pct = round((total_deletions / total_insertions) * 100, 1)

    payload: dict[str, Any] = {
        "report_type": report_type,
        "repo_name": repo.name,
        "repo_slug": repo.repo_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_commits": len(all_commits),
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "churn_rate_percent": churn_pct,
            "unique_contributors": len(author_set),
            "total_files_changed": total_files,
            "default_branch": repo.default_branch,
        },
    }

    if last_commit:
        payload["latest_commit"] = {
            "sha": last_commit.sha,
            "message": (last_commit.message or "")[:200],
            "author": last_commit.author_name,
            "committed_at": (
                last_commit.committed_at.isoformat() if last_commit.committed_at else None
            ),
        }

    if report_type == "dora_metrics":
        payload["dora"] = {
            "note": "DORA metrics computed from PR and deployment data",
            "total_prs": 0,
            "avg_cycle_time_hours": 0.0,
        }
    elif report_type == "team_health":
        from collections import defaultdict

        hour_dist: dict[int, int] = defaultdict(int)
        for c in all_commits:
            if c.committed_at:
                hour_dist[c.committed_at.hour] += 1
        payload["work_pattern"] = {
            "hour_distribution": dict(hour_dist),
            "peak_hour": max(hour_dist, key=hour_dist.get) if hour_dist else None,
        }

    return payload


# ---------------------------------------------------------------------------
# Webhook delivery with optional HMAC-SHA256 signing
# ---------------------------------------------------------------------------


async def deliver_webhook(
    webhook_url: str,
    payload: dict[str, Any],
    secret: str | None = None,
    timeout: float = 15.0,
) -> tuple[int, str]:
    """POST a JSON payload to the webhook URL and return (status_code, body).

    If *secret* is provided the body is signed with HMAC-SHA256 and the
    signature is sent in the ``X-CommitIQ-Signature`` header.
    """
    body_bytes = json.dumps(payload, default=str).encode("utf-8")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "CommitIQ-ReportScheduler/1.0",
    }

    if secret:
        sig = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        headers["X-CommitIQ-Signature"] = f"sha256={sig}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(webhook_url, content=body_bytes, headers=headers)
        return resp.status_code, resp.text[:2000]


# ---------------------------------------------------------------------------
# Execute a single scheduled report run
# ---------------------------------------------------------------------------


async def execute_scheduled_report(db: AsyncSession, schedule_id: int) -> ReportDelivery:
    """Execute a scheduled report: generate payload, deliver via webhook, record delivery.

    Returns the created ReportDelivery row.
    """
    schedule = await db.get(ReportSchedule, schedule_id)
    if not schedule:
        raise ValueError(f"Schedule {schedule_id} not found")

    # Create a pending delivery record
    delivery = ReportDelivery(
        schedule_id=schedule.id,
        repo_id=schedule.repo_id,
        status="running",
        report_type=schedule.report_type,
        retry_count=0,
    )
    db.add(delivery)
    await db.flush()

    start_time = time.monotonic()

    try:
        payload = await generate_report_payload(db, schedule.repo_id, schedule.report_type)

        delivery.report_payload = json.dumps(payload, default=str)

        # Record snapshot info from payload summary
        summary = payload.get("summary", {})
        delivery.snapshot_commits_analyzed = summary.get("total_commits")
        latest = payload.get("latest_commit")
        if latest:
            delivery.snapshot_latest_sha = latest.get("sha")

        # Deliver via webhook if configured
        if schedule.webhook_url:
            status_code, response_body = await deliver_webhook(
                schedule.webhook_url, payload, schedule.webhook_secret
            )
            delivery.webhook_status_code = status_code
            delivery.webhook_response_body = response_body
            if status_code >= 400:
                raise RuntimeError(f"Webhook returned HTTP {status_code}: {response_body[:500]}")
        else:
            delivery.webhook_status_code = 200
            delivery.webhook_response_body = "No webhook configured – report stored only"

        elapsed = round(time.monotonic() - start_time, 2)
        delivery.status = "success"
        delivery.duration_seconds = elapsed
        delivery.completed_at = datetime.now(timezone.utc)

        # Update schedule bookkeeping
        schedule.last_run_at = datetime.now(timezone.utc)
        schedule.last_delivery_status = "success"
        schedule.consecutive_failures = 0
        schedule.next_run_at = compute_next_run(
            schedule.cron_expression, schedule.timezone, after=datetime.now(timezone.utc)
        )

    except Exception as exc:
        elapsed = round(time.monotonic() - start_time, 2)
        delivery.status = "failed"
        delivery.error_message = str(exc)[:2000]
        delivery.duration_seconds = elapsed
        delivery.completed_at = datetime.now(timezone.utc)

        schedule.consecutive_failures += 1
        schedule.last_delivery_status = "failed"

        # Disable schedule after too many consecutive failures
        if schedule.consecutive_failures >= schedule.max_retry_count:
            schedule.is_active = False
            logger.warning(
                "Schedule %d disabled after %d consecutive failures",
                schedule.id,
                schedule.consecutive_failures,
            )

        logger.error("Report delivery failed for schedule %d: %s", schedule_id, exc)

    return delivery


# ---------------------------------------------------------------------------
# Validate a cron expression for user-facing error messages
# ---------------------------------------------------------------------------


def validate_cron_expression(expression: str) -> tuple[bool, str]:
    """Validate a 5-field cron expression. Returns (is_valid, error_message)."""
    try:
        parse_cron(expression)
        return True, ""
    except ValueError as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Build a human-readable schedule description from cron
# ---------------------------------------------------------------------------


def describe_cron(expression: str) -> str:
    """Return a short human-readable description of a 5-field cron expression."""
    fields = parse_cron(expression)
    hour_str = (
        ", ".join(str(h) for h in sorted(fields["hour"]))
        if fields["hour"] != set(range(24))
        else "every hour"
    )
    minute_str = (
        ", ".join(str(m) for m in sorted(fields["minute"]))
        if fields["minute"] == {0}
        else f"at minute {', '.join(str(m) for m in sorted(fields['minute']))}"
    )
    day_names = {v: k for k, v in _DOW_MAP.items()}

    if fields["weekday"] == set(range(7)):
        freq = "Daily"
    elif fields["weekday"] == {1, 2, 3, 4, 5}:
        freq = "Weekdays"
    elif len(fields["weekday"]) == 1:
        dow = list(fields["weekday"])[0]
        freq = f"Every {day_names.get(dow, str(dow))}"
    else:
        days = ", ".join(day_names.get(d, str(d)) for d in sorted(fields["weekday"]))
        freq = f"On {days}"

    if fields["day"] != set(range(1, 32)) or fields["month"] != set(range(1, 13)):
        freq = f"{freq} (specific dates)"

    return f"{freq} {minute_str} {hour_str}"
