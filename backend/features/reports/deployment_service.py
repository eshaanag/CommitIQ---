"""Deployment timeline computation service.

Queries the ``deployments`` table for a repository and computes
summary statistics and per-deployment timeline entries useful for
visualising release cadence, success rates, and environment breakdown.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import Deployment, Repo

logger = logging.getLogger(__name__)

# Environment colour mapping (matches frontend)
_ENV_COLORS: dict[str, str] = {
    "production": "emerald",
    "staging": "amber",
    "development": "sky",
    "preview": "violet",
}


async def get_deployment_timeline(
    db: AsyncSession, repo_id: int, limit: int = 50
) -> dict[str, Any]:
    """Return deployment timeline and summary for *repo_id*.

    Response shape:
        - ``deployments``: list of recent deployment records
        - ``summary``: aggregate stats (total, success rate, by env)
        - ``daily``: per-day deployment counts for sparkline chart
    """
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise ValueError(f"Repository {repo_id} not found")

    stmt = (
        select(Deployment)
        .where(Deployment.repo_id == repo_id)
        .order_by(Deployment.deployed_at.desc())
        .limit(limit)
    )
    deployments = (await db.execute(stmt)).scalars().all()

    if not deployments:
        return _empty_response()

    # ── per-deployment entries ───────────────────────────────────────
    entries: list[dict[str, Any]] = []
    for d in deployments:
        entries.append(
            {
                "id": d.id,
                "provider": d.provider,
                "environment": d.environment,
                "status": d.status,
                "ref": d.ref or "",
                "sha": (d.sha or "")[:12],
                "pipeline_id": d.pipeline_id or "",
                "deployed_at": (d.deployed_at.isoformat() if d.deployed_at else ""),
                "env_color": _ENV_COLORS.get(d.environment, "slate"),
            }
        )

    # ── summary stats ────────────────────────────────────────────────
    total = len(entries)
    success_count = sum(1 for e in entries if e["status"] == "success")
    failure_count = sum(1 for e in entries if e["status"] in {"failed", "error", "canceled"})
    success_rate = round(success_count / max(total, 1) * 100, 1)

    by_env: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "success": 0, "failure": 0}
    )
    for e in entries:
        env = e["environment"]
        by_env[env]["total"] += 1
        if e["status"] == "success":
            by_env[env]["success"] += 1
        elif e["status"] in {"failed", "error", "canceled"}:
            by_env[env]["failure"] += 1

    by_provider: dict[str, int] = defaultdict(int)
    for e in entries:
        by_provider[e["provider"]] += 1

    # most recent deploy time
    most_recent = entries[0]["deployed_at"] if entries else ""

    # ── daily counts for sparkline ───────────────────────────────────
    daily: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0})
    for e in entries:
        if not e["deployed_at"]:
            continue
        try:
            day = e["deployed_at"][:10]  # YYYY-MM-DD
        except (IndexError, TypeError):
            continue
        daily[day]["total"] += 1
        if e["status"] == "success":
            daily[day]["success"] += 1
        elif e["status"] in {"failed", "error", "canceled"}:
            daily[day]["failure"] += 1

    daily_list = [
        {"date": k, "success": v["success"], "failure": v["failure"], "total": v["total"]}
        for k, v in sorted(daily.items())
    ]

    return {
        "deployments": entries,
        "summary": {
            "total_deploys": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "most_recent": most_recent,
            "by_environment": dict(by_env),
            "by_provider": dict(by_provider),
        },
        "daily": daily_list,
    }


def _empty_response() -> dict[str, Any]:
    return {
        "deployments": [],
        "summary": {
            "total_deploys": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "most_recent": "",
            "by_environment": {},
            "by_provider": {},
        },
        "daily": [],
    }
