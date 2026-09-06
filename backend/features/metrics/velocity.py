"""Commit velocity & delivery cadence metrics.

Aggregates commit history into weekly buckets and computes throughput,
consistency, and contributor distribution signals useful for sprint
planning and engineering-manager reporting.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import Commit

logger = logging.getLogger(__name__)

# ISO weekday start (Monday = 0)
_WEEK_START_OFFSET = 7  # roll back to Monday


def _iso_week_key(dt: datetime) -> str:
    """Return an ISO-week label like ``2025-W27``."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _week_start(dt: datetime) -> datetime:
    """Snap a datetime to the Monday 00:00 of its ISO week."""
    return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def _deviation(values: list[float]) -> float:
    """Population standard deviation; returns 0 for empty / single-value."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


async def compute_velocity(db: AsyncSession, repo_id: int) -> dict[str, Any]:
    """Return a velocity/cadence payload for *repo_id*.

    The response contains:
    - ``weekly``: list of per-week summaries (label, commits, lines, contributors)
    - ``totals``: aggregate averages and streaks
    - ``contributors``: per-author velocity breakdown
    """
    stmt = select(Commit).where(Commit.repo_id == repo_id).order_by(Commit.committed_at.asc())
    commits = (await db.execute(stmt)).scalars().all()

    if not commits:
        return _empty_response()

    # ── bucket by ISO week ──────────────────────────────────────────
    weeks: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "label": "",
            "week_start": None,
            "commits": 0,
            "insertions": 0,
            "deletions": 0,
            "lines_changed": 0,
            "files_changed": 0,
            "contributors": set(),
        }
    )

    author_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "name": "",
            "email": "",
            "commits": 0,
            "insertions": 0,
            "deletions": 0,
            "weeks_active": set(),
        }
    )

    for c in commits:
        if not c.committed_at:
            continue
        key = _iso_week_key(c.committed_at)
        w = weeks[key]
        w["label"] = key
        w["week_start"] = _week_start(c.committed_at)
        w["commits"] += 1
        w["insertions"] += c.insertions or 0
        w["deletions"] += c.deletions or 0
        w["lines_changed"] += (c.insertions or 0) + (c.deletions or 0)
        w["files_changed"] += c.files_changed or 0

        author = c.author_name or c.author_email or "unknown"
        w["contributors"].add(author)

        at = author_totals[author]
        at["name"] = author
        at["email"] = c.author_email or ""
        at["commits"] += 1
        at["insertions"] += c.insertions or 0
        at["deletions"] += c.deletions or 0
        at["weeks_active"].add(key)

    # ── serialise weekly summaries ───────────────────────────────────
    sorted_weeks = sorted(weeks.values(), key=lambda w: w["week_start"])
    weekly_data: list[dict[str, Any]] = []
    for w in sorted_weeks:
        weekly_data.append(
            {
                "label": w["label"],
                "week_start": w["week_start"].strftime("%Y-%m-%d") if w["week_start"] else "",
                "commits": w["commits"],
                "insertions": w["insertions"],
                "deletions": w["deletions"],
                "lines_changed": w["lines_changed"],
                "files_changed": w["files_changed"],
                "contributor_count": len(w["contributors"]),
            }
        )

    # ── aggregate stats ──────────────────────────────────────────────
    commit_counts = [w["commits"] for w in weekly_data]
    lines_per_week = [w["lines_changed"] for w in weekly_data]
    contributor_counts = [w["contributor_count"] for w in weekly_data]

    total_commits = sum(commit_counts)
    total_insertions = sum(w["insertions"] for w in weekly_data)
    total_deletions = sum(w["deletions"] for w in weekly_data)
    num_weeks = max(len(weekly_data), 1)
    num_active_contributors = len(author_totals)

    avg_commits_per_week = round(total_commits / num_weeks, 1)
    avg_lines_per_week = round(sum(lines_per_week) / num_weeks, 0)
    avg_contributors_per_week = round(sum(contributor_counts) / num_weeks, 1)

    # longest commit streak (consecutive weeks with >= 1 commit)
    streak = 0
    max_streak = 0
    prev_label: str | None = None
    for w in weekly_data:
        if prev_label is not None:
            prev_idx = int(prev_label.split("-W")[1])
            cur_idx = int(w["label"].split("-W")[1])
            if cur_idx == prev_idx + 1 or (cur_idx == 1 and prev_idx >= 51):
                streak += 1
            else:
                streak = 1
        else:
            streak = 1
        max_streak = max(max_streak, streak)
        prev_label = w["label"]

    commit_deviation = round(_deviation(commit_counts), 1)
    lines_deviation = round(_deviation(lines_per_week), 0)

    # cadence score: 0-100, higher = more consistent
    if avg_commits_per_week > 0 and commit_deviation is not None:
        cv = commit_deviation / max(avg_commits_per_week, 0.1)
        cadence_score = max(0, min(100, round(100 * (1 - min(cv, 1.0)))))
    else:
        cadence_score = 0

    # ── contributor breakdown ────────────────────────────────────────
    contrib_list: list[dict[str, Any]] = []
    for name, data in sorted(author_totals.items(), key=lambda x: -x[1]["commits"]):
        weeks_active = len(data["weeks_active"])
        contrib_list.append(
            {
                "name": data["name"],
                "email": data["email"],
                "commits": data["commits"],
                "commit_pct": (
                    round(data["commits"] / total_commits * 100, 1) if total_commits else 0
                ),
                "insertions": data["insertions"],
                "deletions": data["deletions"],
                "weeks_active": weeks_active,
                "avg_commits_per_active_week": round(data["commits"] / max(weeks_active, 1), 1),
            }
        )

    # ── response ─────────────────────────────────────────────────────
    return {
        "weekly": weekly_data,
        "totals": {
            "total_commits": total_commits,
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
            "num_weeks": num_weeks,
            "num_active_contributors": num_active_contributors,
            "avg_commits_per_week": avg_commits_per_week,
            "avg_lines_per_week": avg_lines_per_week,
            "avg_contributors_per_week": avg_contributors_per_week,
            "max_commit_streak_weeks": max_streak,
            "commit_deviation": commit_deviation,
            "lines_deviation": lines_deviation,
            "cadence_score": cadence_score,
        },
        "contributors": contrib_list[:20],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "weekly": [],
        "totals": {
            "total_commits": 0,
            "total_insertions": 0,
            "total_deletions": 0,
            "num_weeks": 0,
            "num_active_contributors": 0,
            "avg_commits_per_week": 0,
            "avg_lines_per_week": 0,
            "avg_contributors_per_week": 0,
            "max_commit_streak_weeks": 0,
            "commit_deviation": 0,
            "lines_deviation": 0,
            "cadence_score": 0,
        },
        "contributors": [],
    }
