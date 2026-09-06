import json
import logging
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.features.llm_analysis.cache import _get_redis
from backend.shared.models import Commit

logger = logging.getLogger(__name__)


async def compute_team_health(db: AsyncSession, repo_id: int) -> dict:
    redis_client = _get_redis()
    cache_key = f"team_health:{repo_id}"
    if redis_client:
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info("Cache hit for team health repo_id=%s", repo_id)
                return json.loads(cached_data)
        except Exception as exc:
            logger.warning("Failed to fetch cached team health: %s", exc)

    stmt = select(Commit).where(Commit.repo_id == repo_id)
    result = await db.execute(stmt)
    commits = result.scalars().all()

    if not commits:
        res = {
            "burnout_risk_score": "Low",
            "weekend_commits_percent": 0.0,
            "after_hours_commits_percent": 0.0,
            "context_switching_score": "Low",
            "avg_files_per_day": 0.0,
        }
        if redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(res), ex=3600)
            except Exception as exc:
                logger.warning("Failed to cache team health: %s", exc)
        return res

    total_commits = len(commits)
    weekend_commits = 0
    after_hours_commits = 0

    author_days = defaultdict(int)
    author_files = defaultdict(int)

    for c in commits:
        if not c.committed_at:
            continue

        if c.committed_at.weekday() >= 5:
            weekend_commits += 1

        hour = c.committed_at.hour
        if hour >= 20 or hour < 8:
            after_hours_commits += 1

        date_str = c.committed_at.strftime("%Y-%m-%d")
        author = c.author_email or c.author_name or "unknown"
        key = f"{author}_{date_str}"
        author_days[key] += 1
        author_files[key] += c.files_changed or 0

    weekend_percent = (weekend_commits / total_commits) * 100
    after_hours_percent = (after_hours_commits / total_commits) * 100

    burnout_score = "Low"
    if weekend_percent > 15 or after_hours_percent > 20:
        burnout_score = "High"
    elif weekend_percent > 5 or after_hours_percent > 10:
        burnout_score = "Medium"

    avg_files = 0.0
    if author_days:
        avg_files = sum(author_files.values()) / len(author_files)

    context_switch_score = "Low"
    if avg_files > 50:
        context_switch_score = "High"
    elif avg_files > 20:
        context_switch_score = "Medium"

    res = {
        "burnout_risk_score": burnout_score,
        "weekend_commits_percent": round(weekend_percent, 1),
        "after_hours_commits_percent": round(after_hours_percent, 1),
        "context_switching_score": context_switch_score,
        "avg_files_per_day": round(avg_files, 1),
    }

    if redis_client:
        try:
            await redis_client.set(cache_key, json.dumps(res), ex=3600)
        except Exception as exc:
            logger.warning("Failed to cache team health: %s", exc)

    return res
