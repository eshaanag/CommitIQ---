"""Actionable health recommendations engine – analyses all computed metrics
and produces a prioritised, deduplicated list of suggestions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import BusFactor, Commit, HealthSnapshot, Repo


def _r(id, cat, sev, title, desc, impact, effort, **kw):
    """Shortcut to build a recommendation dict."""
    return {
        "id": id,
        "category": cat,
        "severity": sev,
        "title": title,
        "description": desc,
        "impact": max(0, min(100, impact)),
        "effort": effort,
        "metric": kw.get("metric"),
        "current_value": kw.get("cur"),
        "target_value": kw.get("tgt"),
        "file_path": kw.get("file"),
    }


def _health_trend(snaps):
    if len(snaps) < 4:
        return []
    scores = [s.health_score for s in snaps[:4] if s.health_score is not None]
    older = [s.health_score for s in snaps[4:12] if s.health_score is not None]
    if len(scores) < 3 or not older:
        return []
    d = sum(scores) / len(scores) - sum(older) / len(older)
    cur = sum(scores) / len(scores)
    old = sum(older) / len(older)
    if d < -8:
        return [
            _r(
                "health_decline",
                "health",
                "critical",
                "Sustained health-score decline",
                f"Health score dropped {abs(d):.1f} pts (now {cur:.1f}). Compounding code-health issues.",
                90,
                "medium",
                metric="health_score",
                cur=f"{cur:.1f}",
                tgt=f"{old:.1f}",
            )
        ]
    if d < -4:
        return [
            _r(
                "health_drift",
                "health",
                "high",
                "Health score trending downward",
                f"Score drifted down {abs(d):.1f} pts (now {cur:.1f}). Early intervention recommended.",
                65,
                "low",
                metric="health_score",
                cur=f"{cur:.1f}",
                tgt=f"{old:.1f}",
            )
        ]
    return []


def _complexity(snaps):
    if not snaps:
        return []
    recs, lat = [], snaps[0]
    if lat.avg_complexity > 10:
        recs.append(
            _r(
                "high_complexity",
                "complexity",
                "critical",
                "Avg cyclomatic complexity critically high",
                f"Complexity is {lat.avg_complexity:.1f}. High complexity correlates with defects. Refactor large functions.",
                85,
                "high",
                metric="avg_complexity",
                cur=f"{lat.avg_complexity:.1f}",
                tgt="≤ 5.0",
            )
        )
    elif lat.avg_complexity > 6:
        recs.append(
            _r(
                "moderate_complexity",
                "complexity",
                "medium",
                "Complexity moderately elevated",
                f"Complexity is {lat.avg_complexity:.1f}. Consider refactoring the most complex units.",
                45,
                "medium",
                metric="avg_complexity",
                cur=f"{lat.avg_complexity:.1f}",
                tgt="≤ 5.0",
            )
        )
    if len(snaps) >= 2 and lat.avg_complexity > snaps[1].avg_complexity * 1.25:
        pct = ((lat.avg_complexity / max(snaps[1].avg_complexity, 0.1)) - 1) * 100
        recs.append(
            _r(
                "complexity_spike",
                "complexity",
                "high",
                "Complexity spike in latest commit",
                f"Complexity jumped {pct:.0f}% ({snaps[1].avg_complexity:.1f} → {lat.avg_complexity:.1f}).",
                70,
                "medium",
                metric="avg_complexity",
                cur=f"{lat.avg_complexity:.1f}",
                tgt=f"≤ {snaps[1].avg_complexity:.1f}",
            )
        )
    return recs


def _bus_factor(bfs):
    crit = [b for b in bfs if b.risk_level == "critical"]
    high = [b for b in bfs if b.risk_level == "high"]
    recs = []
    if crit:
        names = ", ".join(b.module_path for b in crit[:3])
        pct = max(b.top_contributor_pct for b in crit)
        recs.append(
            _r(
                "bus_factor_critical",
                "bus_factor",
                "critical",
                f"{len(crit)} module(s) at critical bus-factor risk",
                f"{names} depend on a single contributor (up to {pct:.0f}%). Cross-train to mitigate.",
                80,
                "high",
                metric="bus_factor_min",
                cur=f"{len(crit)} critical",
                tgt="0 critical",
            )
        )
    elif high:
        names = ", ".join(b.module_path for b in high[:3])
        recs.append(
            _r(
                "bus_factor_high",
                "bus_factor",
                "high",
                f"{len(high)} module(s) at high bus-factor risk",
                f"{names} have limited contributor diversity. Schedule knowledge-sharing.",
                55,
                "medium",
                metric="bus_factor_min",
                cur=f"{len(high)} high",
                tgt="0 high",
            )
        )
    return recs


def _churn(snaps):
    if not snaps:
        return []
    c = snaps[0].churn_rate
    if c > 0.4:
        return [
            _r(
                "high_churn",
                "churn",
                "critical",
                "Codebase churn rate critically high",
                f"Churn is {c*100:.0f}%. Code rewritten faster than it matures. Investigate root causes.",
                75,
                "high",
                metric="churn_rate",
                cur=f"{c*100:.0f}%",
                tgt="< 25%",
            )
        ]
    if c > 0.25:
        return [
            _r(
                "elevated_churn",
                "churn",
                "medium",
                "Churn rate elevated",
                f"Churn is {c*100:.0f}%. Tighten code review to catch design issues earlier.",
                40,
                "low",
                metric="churn_rate",
                cur=f"{c*100:.0f}%",
                tgt="< 15%",
            )
        ]
    return []


def _hotspots(snaps):
    if not snaps:
        return []
    s = snaps[0]
    recs = []
    if s.hotspot_persistence_score > 70:
        recs.append(
            _r(
                "persistent_hotspots",
                "hotspots",
                "high",
                "Persistent hotspots need attention",
                f"Hotspot persistence {s.hotspot_persistence_score:.0f}/100 with {s.hotspot_count} files. Architectural refactoring needed.",
                60,
                "high",
                metric="hotspot_persistence_score",
                cur=f"{s.hotspot_persistence_score:.0f}",
                tgt="< 40",
            )
        )
    elif s.hotspot_count > 5:
        recs.append(
            _r(
                "many_hotspots",
                "hotspots",
                "medium",
                f"{s.hotspot_count} hotspot files detected",
                "Multiple high-churn complexity files. Prioritise the top 3 by risk score.",
                45,
                "medium",
                metric="hotspot_count",
                cur=str(s.hotspot_count),
                tgt="< 3",
            )
        )
    return recs


def _dependencies(snaps):
    if not snaps:
        return []
    s = snaps[0]
    recs = []
    if s.has_cycles:
        recs.append(
            _r(
                "dependency_cycles",
                "dependencies",
                "critical",
                "Circular dependency detected",
                "Import cycles cause tight coupling and slow builds. Extract shared interfaces.",
                85,
                "high",
                metric="has_cycles",
                cur="true",
                tgt="false",
            )
        )
    if s.dependency_density > 0.6:
        recs.append(
            _r(
                "high_dep_density",
                "dependencies",
                "high",
                "High dependency density",
                f"Density is {s.dependency_density:.2f}. High coupling makes changes risky.",
                55,
                "high",
                metric="dependency_density",
                cur=f"{s.dependency_density:.2f}",
                tgt="< 0.4",
            )
        )
    return recs


def _team_health(commits):
    if len(commits) < 5:
        return []
    wk = sum(1 for c in commits if c.committed_at and c.committed_at.weekday() >= 5)
    ah = sum(
        1
        for c in commits
        if c.committed_at and (c.committed_at.hour >= 20 or c.committed_at.hour < 8)
    )
    n = len(commits)
    wp, ap = wk / n * 100, ah / n * 100
    recs = []
    if wp > 15 or ap > 20:
        recs.append(
            _r(
                "burnout_risk",
                "team_health",
                "critical",
                "Burnout risk: excessive after-hours commits",
                f"{wp:.0f}% weekend, {ap:.0f}% after-hours. Predicts burnout and attrition.",
                80,
                "medium",
                metric="burnout_risk",
                cur="High",
                tgt="Low",
            )
        )
    elif wp > 5 or ap > 10:
        recs.append(
            _r(
                "burnout_warning",
                "team_health",
                "high",
                "Elevated after-hours commit activity",
                f"{wp:.0f}% weekend, {ap:.0f}% after-hours. Monitor capacity.",
                50,
                "low",
                metric="burnout_risk",
                cur="Medium",
                tgt="Low",
            )
        )
    af = defaultdict(int)
    for c in commits:
        af[c.author_email or "unknown"] += c.files_changed or 0
    if af:
        avg = sum(af.values()) / len(af)
        if avg > 40:
            recs.append(
                _r(
                    "context_switching",
                    "team_health",
                    "high",
                    "High context-switching load",
                    f"Contributors average {avg:.0f} files/day. Consider feature ownership.",
                    50,
                    "medium",
                    metric="context_switching",
                    cur="High",
                    tgt="Low",
                )
            )
    return recs


def _semantic_drift(snaps):
    if not snaps or snaps[0].avg_semantic_drift <= 0.4:
        return []
    d = snaps[0].avg_semantic_drift
    return [
        _r(
            "semantic_drift",
            "documentation",
            "medium",
            "Commit messages diverge from code changes",
            f"Avg drift is {d:.2f}. Enforce conventional commits and review standards.",
            35,
            "low",
            metric="avg_semantic_drift",
            cur=f"{d:.2f}",
            tgt="< 0.2",
        )
    ]


async def generate_recommendations(db: AsyncSession, repo_id: int) -> dict[str, Any]:
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise ValueError(f"Repository {repo_id} not found")

    snaps = (
        (
            await db.execute(
                select(HealthSnapshot)
                .where(HealthSnapshot.repo_id == repo_id)
                .order_by(HealthSnapshot.computed_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    bfs = (await db.execute(select(BusFactor).where(BusFactor.repo_id == repo_id))).scalars().all()
    commits = (
        (
            await db.execute(
                select(Commit)
                .where(Commit.repo_id == repo_id)
                .order_by(Commit.committed_at.desc())
                .limit(200)
            )
        )
        .scalars()
        .all()
    )

    all_recs = (
        _health_trend(snaps)
        + _complexity(snaps)
        + _bus_factor(bfs)
        + _churn(snaps)
        + _hotspots(snaps)
        + _dependencies(snaps)
        + _team_health(commits)
        + _semantic_drift(snaps)
    )
    all_recs.sort(key=lambda r: r["impact"], reverse=True)

    seen: set[str] = set()
    unique = []
    for r in all_recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    crit = sum(1 for r in unique if r["severity"] == "critical")
    high = sum(1 for r in unique if r["severity"] == "high")
    med = sum(1 for r in unique if r["severity"] == "medium")

    return {
        "repo_name": repo.name,
        "repo_slug": repo.repo_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health_score": max(0, 100 - crit * 25 - high * 10 - len(unique) * 2),
        "total_recommendations": len(unique),
        "critical_count": crit,
        "high_count": high,
        "medium_count": med,
        "low_count": len(unique) - crit - high - med,
        "recommendations": unique,
    }
