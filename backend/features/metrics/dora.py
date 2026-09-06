from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import Deployment, PullRequest, Repo


def _parse_datetime(dt: datetime | str | None) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        s = dt.strip()
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return None


def _seconds_between(t1: datetime, t2: datetime) -> float:
    """Safely calculate t1 - t2 in seconds handling naive/aware mixed datetimes."""
    if t1.tzinfo is not None and t2.tzinfo is None:
        t1 = t1.replace(tzinfo=None)
    elif t1.tzinfo is None and t2.tzinfo is not None:
        t2 = t2.replace(tzinfo=None)
    return (t1 - t2).total_seconds()


async def compute_dora_metrics(
    db: AsyncSession,
    repo_id: int,
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
) -> dict:
    repo = await db.get(Repo, repo_id)
    default_branch = repo.default_branch if repo else "main"
    parsed_start = _parse_datetime(start_date)
    parsed_end = _parse_datetime(end_date)

    deploy_stmt = select(Deployment).where(
        Deployment.repo_id == repo_id,
        Deployment.status == "success",
        (Deployment.ref == default_branch)
        | (Deployment.ref == f"refs/heads/{default_branch}")
        | (Deployment.ref.is_(None)),
    )
    if parsed_start is not None:
        deploy_stmt = deploy_stmt.where(Deployment.deployed_at >= parsed_start)
    if parsed_end is not None:
        deploy_stmt = deploy_stmt.where(Deployment.deployed_at <= parsed_end)
    deploy_res = await db.execute(deploy_stmt)
    deployments = deploy_res.scalars().all()

    pr_stmt = select(PullRequest).where(
        PullRequest.repo_id == repo_id, PullRequest.merged_at.isnot(None)
    )
    if parsed_start is not None:
        pr_stmt = pr_stmt.where(PullRequest.merged_at >= parsed_start)
    if parsed_end is not None:
        pr_stmt = pr_stmt.where(PullRequest.merged_at <= parsed_end)
    pr_res = await db.execute(pr_stmt)
    prs = pr_res.scalars().all()

    if not prs and not deployments:
        return {
            "deployment_frequency": "Low",
            "deployment_frequency_value": 0.0,
            "change_failure_rate": "Low",
            "change_failure_rate_value": 0.0,
            "mttr_hours": 0.0,
            "mttr_category": "Low",
            "dora_score": "Low",
        }

    # 1. Deployment Frequency (Deployments per week)
    if parsed_start is not None and parsed_end is not None:
        days_span = max(1.0, _seconds_between(parsed_end, parsed_start) / 86400.0)
    elif parsed_start is not None:
        now = datetime.now(timezone.utc) if parsed_start.tzinfo else datetime.now()
        days_span = max(1.0, _seconds_between(now, parsed_start) / 86400.0)
    elif deployments:
        ref_tz = deployments[0].deployed_at.tzinfo if deployments[0].deployed_at else None
        now = parsed_end or (datetime.now(ref_tz) if ref_tz else datetime.now())
        earliest_dep = min(deployments, key=lambda d: d.deployed_at or now)
        earliest_dt = earliest_dep.deployed_at or now
        days_span = max(1.0, _seconds_between(now, earliest_dt) / 86400.0)
    else:
        ref_tz = prs[0].merged_at.tzinfo if prs[0].merged_at else None
        now = parsed_end or (datetime.now(ref_tz) if ref_tz else datetime.now())
        earliest_pr = min(prs, key=lambda p: p.merged_at or now)
        earliest_dt = earliest_pr.merged_at or now
        days_span = max(1.0, _seconds_between(now, earliest_dt) / 86400.0)

    weeks_span = max(1.0, days_span / 7.0)
    event_count = len(deployments) if deployments else len(prs)
    weekly_deployments = event_count / weeks_span

    if weekly_deployments >= 7:
        df_category = "Elite"
    elif weekly_deployments >= 1:
        df_category = "High"
    elif weekly_deployments >= 0.25:
        df_category = "Medium"
    else:
        df_category = "Low"

    # 2. Change Failure Rate
    failure_prs = []
    for pr in prs:
        title = pr.title.lower()
        if "hotfix" in title or "fix" in title or "revert" in title or "bug" in title:
            failure_prs.append(pr)

    cfr_value = (len(failure_prs) / len(prs) * 100.0) if prs else 0.0

    if cfr_value <= 5:
        cfr_category = "Elite"
    elif cfr_value <= 10:
        cfr_category = "High"
    elif cfr_value <= 15:
        cfr_category = "Medium"
    else:
        cfr_category = "Low"

    # 3. MTTR (Mean Time to Recovery)
    mttr_seconds = 0
    valid_mttr_prs = 0
    for pr in failure_prs:
        if pr.created_at and pr.merged_at:
            time_to_resolve = _seconds_between(pr.merged_at, pr.created_at)
            if time_to_resolve > 0:
                mttr_seconds += time_to_resolve
                valid_mttr_prs += 1

    mttr_hours = (mttr_seconds / valid_mttr_prs / 3600.0) if valid_mttr_prs > 0 else 0.0

    if valid_mttr_prs == 0:
        mttr_category = "Elite"
    elif mttr_hours < 1:
        mttr_category = "Elite"
    elif mttr_hours < 24:
        mttr_category = "High"
    elif mttr_hours < 168:
        mttr_category = "Medium"
    else:
        mttr_category = "Low"

    score_map = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
    total_score = score_map[df_category] + score_map[cfr_category] + score_map[mttr_category]
    avg_score = total_score / 3.0

    if avg_score >= 3.5:
        dora_score = "Elite"
    elif avg_score >= 2.5:
        dora_score = "High"
    elif avg_score >= 1.5:
        dora_score = "Medium"
    else:
        dora_score = "Low"

    return {
        "deployment_frequency": df_category,
        "deployment_frequency_value": round(weekly_deployments, 1),
        "change_failure_rate": cfr_category,
        "change_failure_rate_value": round(cfr_value, 1),
        "mttr_hours": round(mttr_hours, 1),
        "mttr_category": mttr_category,
        "dora_score": dora_score,
    }
