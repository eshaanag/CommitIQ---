"""Weekly repository health digest – single-endpoint metric aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import BusFactor, Commit, HealthSnapshot, Repo


async def compute_weekly_digest(
    db: AsyncSession, repo_id: int, *, weeks: int = 1
) -> dict[str, Any]:
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise ValueError(f"Repository {repo_id} not found")

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(weeks=weeks)
    prev_start = window_start - timedelta(weeks=weeks)

    # commits in window
    c_stmt = select(Commit).where(Commit.repo_id == repo_id, Commit.committed_at >= window_start)
    commits = (await db.execute(c_stmt.order_by(Commit.committed_at.desc()))).scalars().all()

    # snapshots for trend
    snap_stmt = (
        select(HealthSnapshot)
        .where(HealthSnapshot.repo_id == repo_id)
        .order_by(HealthSnapshot.computed_at.desc())
    )
    snapshots = (await db.execute(snap_stmt)).scalars().all()
    cur_snaps = [s for s in snapshots if s.computed_at and s.computed_at >= window_start]
    prev_snaps = [
        s for s in snapshots if s.computed_at and prev_start <= s.computed_at < window_start
    ]

    # bus factor
    bus_factors = (
        (await db.execute(select(BusFactor).where(BusFactor.repo_id == repo_id))).scalars().all()
    )

    # aggregates
    total_ins = sum(c.insertions or 0 for c in commits)
    total_del = sum(c.deletions or 0 for c in commits)
    total_files = sum(c.files_changed or 0 for c in commits)
    authors: dict[str, dict[str, int]] = defaultdict(
        lambda: {"commits": 0, "insertions": 0, "deletions": 0}
    )
    for c in commits:
        k = c.author_email or c.author_name or "unknown"
        authors[k]["commits"] += 1
        authors[k]["insertions"] += c.insertions or 0
        authors[k]["deletions"] += c.deletions or 0
    top_authors = sorted(authors.items(), key=lambda x: x[1]["commits"], reverse=True)[:10]

    # trend helpers
    def _avg(snaps, attr, default=0.0):
        vals = [getattr(s, attr) for s in snaps if getattr(s, attr, None) is not None]
        return round(sum(vals) / len(vals), 4) if vals else default

    cur_h, prev_h = _avg(cur_snaps, "health_score"), _avg(prev_snaps, "health_score")
    cur_cc, prev_cc = _avg(cur_snaps, "avg_complexity", 0.0), _avg(
        prev_snaps, "avg_complexity", 0.0
    )
    cur_ch, prev_ch = _avg(cur_snaps, "churn_rate"), _avg(prev_snaps, "churn_rate")
    health_trend = round(cur_h - prev_h, 2)
    cc_trend = round(cur_cc - prev_cc, 2)
    churn_trend = round(cur_ch - prev_ch, 4)

    # bus factor risk
    risk = [bf for bf in bus_factors if bf.risk_level in ("critical", "high")]
    bf_summary = {
        "total_modules": len(bus_factors),
        "critical_risk_count": sum(1 for bf in bus_factors if bf.risk_level == "critical"),
        "high_risk_count": sum(1 for bf in bus_factors if bf.risk_level == "high"),
        "top_risk_modules": [
            {
                "module": bf.module_path,
                "risk_level": bf.risk_level,
                "top_contributor": bf.top_contributor,
                "top_contributor_pct": bf.top_contributor_pct,
                "contributor_count": bf.contributor_count,
            }
            for bf in sorted(risk, key=lambda b: b.top_contributor_pct, reverse=True)[:5]
        ],
    }

    # alerts
    alerts: list[dict[str, str]] = []
    if health_trend < -10:
        alerts.append(
            {
                "severity": "critical",
                "metric": "health_score",
                "message": f"Health score dropped {abs(health_trend):.1f} points vs prior window.",
            }
        )
    elif health_trend < -5:
        alerts.append(
            {
                "severity": "warning",
                "metric": "health_score",
                "message": f"Health score declined {abs(health_trend):.1f} points.",
            }
        )
    if cc_trend > 1.0:
        alerts.append(
            {
                "severity": "warning",
                "metric": "complexity",
                "message": f"Average cyclomatic complexity rose by {cc_trend:.1f}.",
            }
        )
    if churn_trend > 0.05:
        alerts.append(
            {
                "severity": "warning",
                "metric": "churn",
                "message": f"Churn rate increased by {churn_trend * 100:.1f}pp.",
            }
        )
    if bf_summary["critical_risk_count"] > 0:
        alerts.append(
            {
                "severity": "critical",
                "metric": "bus_factor",
                "message": f"{bf_summary['critical_risk_count']} module(s) at critical bus-factor risk.",
            }
        )

    # persistent hotspots
    file_churn: dict[str, int] = defaultdict(int)
    for s in snapshots:
        if s.top_files_json:
            try:
                for f in json.loads(s.top_files_json):
                    file_churn[f.get("path", "")] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    hotspots = sorted(file_churn.items(), key=lambda x: x[1], reverse=True)[:8]

    return {
        "repo_name": repo.name,
        "repo_slug": repo.repo_slug,
        "generated_at": now.isoformat(),
        "window_weeks": weeks,
        "summary": {
            "total_commits": len(commits),
            "total_insertions": total_ins,
            "total_deletions": total_del,
            "total_files_changed": total_files,
            "unique_contributors": len(authors),
        },
        "health": {
            "current_avg_score": cur_h,
            "previous_avg_score": prev_h,
            "trend": health_trend,
            "trend_direction": (
                "up" if health_trend > 0 else ("down" if health_trend < 0 else "flat")
            ),
        },
        "complexity": {"current_avg": cur_cc, "previous_avg": prev_cc, "trend": cc_trend},
        "churn": {"current_avg_rate": cur_ch, "previous_avg_rate": prev_ch, "trend": churn_trend},
        "bus_factor": bf_summary,
        "top_contributors": [{"author": a, **s} for a, s in top_authors],
        "persistent_hotspots": [{"path": p, "snapshot_count": c} for p, c in hotspots],
        "alerts": alerts,
    }
