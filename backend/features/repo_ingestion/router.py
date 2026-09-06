import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated


def _extract_metrics_in_worktree(repo_path: Path, commit_data: dict) -> tuple[str, dict]:
    wt_name = f"wt_{uuid.uuid4().hex[:8]}"
    wt_path = repo_path.parent / wt_name
    res = subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt_path), commit_data["full_sha"]],
        cwd=repo_path,
        capture_output=True,
    )
    target_path = wt_path if res.returncode == 0 else repo_path

    try:
        from backend.features.repo_ingestion.metrics_extractor import extract_commit_metrics

        metrics = extract_commit_metrics(target_path, commit_data)
        return commit_data["full_sha"], metrics
    except ModuleNotFoundError:
        return commit_data["full_sha"], {
            "default": {
                "avg_complexity": 1.0,
                "max_complexity": 1.0,
                "loc": 10,
                "semantic_drift_score": 0.0,
                "drift_method": "none",
            }
        }
    finally:
        if res.returncode == 0:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=repo_path,
                capture_output=True,
            )


from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import MAX_COMMITS
from backend.database import commit_with_retry, get_db
from backend.features.llm_analysis.cache import make_cache_key
from backend.features.repo_ingestion.bus_factor import compute_bus_factor_from_history
from backend.features.repo_ingestion.clone_service import (
    cleanup_repo,
    clone_repo,
    count_available_commits,
    fetch_github_metadata,
    fetch_github_pull_requests,
    make_repo_slug,
    parse_github_url,
    sanitize_repo_url,
)
from backend.features.repo_ingestion.commit_walker import walk_commits
from backend.features.repo_ingestion.graph_builder import build_cochange_edges, build_import_edges
from backend.features.repo_ingestion.health_scorer import assign_health_color, compute_full_snapshot
from backend.shared.models import (
    AnalysisJob,
    BusFactor,
    Commit,
    GraphEdge,
    GraphNode,
    HealthSnapshot,
    LLMNarrative,
    PullRequest,
    Repo,
)
from backend.shared.schemas import (
    BusFactorWrapper,
    CommitDetailResponse,
    GraphResponse,
    HotspotResponse,
    IngestRequest,
    IngestResponse,
    JobProgressOut,
    LLMUsageOut,
    RepoCompareDelta,
    RepoCompareInsight,
    RepoCompareItem,
    RepoCompareMetrics,
    RepoCompareResponse,
    RepoOut,
    RescanResponse,
    TimelineResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/repos", tags=["repos"])
ACTIVE_JOB_STATUSES = {"queued", "cloning", "analyzing", "building_graph", "computing_bus_factor"}
CANCELLED_MESSAGE = "Ingestion cancelled by user."


def _http_error(status_code: int, detail: str, code: str = "request_error") -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail, headers={"X-CommitIQ-Error": code})


_ingestion_semaphore: asyncio.Semaphore | None = None


def get_ingestion_semaphore() -> asyncio.Semaphore:
    """Return an asyncio Semaphore to throttle concurrent repository ingestion and rescan jobs."""
    global _ingestion_semaphore
    if _ingestion_semaphore is None:
        from backend.config import MAX_CONCURRENT_INGESTIONS

        _ingestion_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INGESTIONS)
    return _ingestion_semaphore


class IngestionCancelled(RuntimeError):
    pass


async def _update_job(job_id: int, db: AsyncSession | None = None, **kwargs) -> None:
    from backend.database import AsyncSessionLocal, commit_with_retry

    try:
        if db is not None:
            job = await db.get(AnalysisJob, job_id)
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)
                await commit_with_retry(db)
        else:
            async with AsyncSessionLocal() as session:
                job = await session.get(AnalysisJob, job_id)
                if job:
                    for key, value in kwargs.items():
                        setattr(job, key, value)
                    await commit_with_retry(session)
    except Exception as exc:
        logger.warning("Failed to update job id=%s: %s", job_id, exc)


async def _raise_if_cancelled(job_id: int) -> None:
    from backend.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AnalysisJob.status).where(AnalysisJob.id == job_id))
        status = result.scalar_one_or_none()
        if status == "cancelled":
            raise IngestionCancelled(CANCELLED_MESSAGE)


def _parse_top_files(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _parse_json_list(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _detect_cycles(edges: list[dict]) -> bool:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("edge_type") != "import":
            continue
        graph.setdefault(edge["source_file"], []).append(edge["target_file"])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for neighbor in graph.get(node, []):
            if visit(neighbor):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _duration_seconds(started_at: datetime | None, completed_at: datetime) -> float | None:
    if not started_at:
        return None
    return (_utc_datetime(completed_at) - _utc_datetime(started_at)).total_seconds()


def _hotspot_files(
    commit_history: list[dict],
    current_index: int,
    file_metrics_map: dict,
) -> list[str]:
    recent_commits = commit_history[max(0, current_index - 4) : current_index + 1]
    churn_counts: dict[str, int] = {}
    for commit_data in recent_commits:
        for fpath in set(commit_data.get("files_list", [])):
            churn_counts[fpath] = churn_counts.get(fpath, 0) + 1
    return [
        fpath
        for fpath, count in churn_counts.items()
        if count > 3 and file_metrics_map.get(fpath, {}).get("avg_complexity", 0.0) > 5.0
    ]


def _persistent_hotspots(
    commit_history: list[dict],
    current_index: int,
    file_metrics_map: dict,
    min_recent_commits: int = 3,
) -> list[dict]:
    recent_commits = commit_history[max(0, current_index - 5) : current_index + 1]
    churn_counts: dict[str, int] = {}
    for commit_data in recent_commits:
        for fpath in set(commit_data.get("files_list", [])):
            churn_counts[fpath] = churn_counts.get(fpath, 0) + 1

    hotspots = []
    for fpath, recent_count in churn_counts.items():
        metrics = file_metrics_map.get(fpath, {})
        avg_complexity = float(metrics.get("avg_complexity", 0.0))
        if recent_count < min_recent_commits or avg_complexity <= 5.0:
            continue
        hotspots.append(
            {
                "path": fpath,
                "recent_commit_count": recent_count,
                "complexity": round(avg_complexity, 2),
                "loc": int(metrics.get("loc", 0)),
            }
        )
    return sorted(
        hotspots,
        key=lambda item: (item["recent_commit_count"], item["complexity"], item["loc"]),
        reverse=True,
    )


def _snapshot_payload(commit: Commit, snap: HealthSnapshot) -> dict:
    return {
        "id": snap.id,
        "repo_id": snap.repo_id,
        "commit_id": snap.commit_id,
        "sha": commit.sha,
        "full_sha": commit.full_sha,
        "message": commit.message,
        "author": commit.author_name,
        "author_email": commit.author_email,
        "committed_at": commit.committed_at,
        "health_score": snap.health_score,
        "avg_complexity": snap.avg_complexity,
        "max_complexity": snap.max_complexity,
        "total_loc": snap.total_loc,
        "churn_rate": snap.churn_rate,
        "num_files_changed": snap.num_files_changed,
        "insertions": commit.insertions,
        "deletions": commit.deletions,
        "bus_factor_min": snap.bus_factor_min,
        "health_delta": snap.health_delta,
        "cc_score": snap.cc_score,
        "churn_score": snap.churn_score,
        "bus_score": snap.bus_score,
        "loc_score": snap.loc_score,
        "subscores": {
            "complexity_drift": snap.complexity_drift_score or snap.cc_score,
            "churn_risk": snap.churn_risk_score or snap.churn_score,
            "bus_factor_risk": snap.bus_factor_risk_score or snap.bus_score,
            "dependency_health": snap.dependency_health_score or snap.loc_score,
            "semantic_drift": snap.semantic_health_score,
        },
        "dependency_density": snap.dependency_density,
        "has_cycles": bool(snap.has_cycles),
        "hotspot_count": snap.hotspot_count,
        "avg_semantic_drift": snap.avg_semantic_drift,
        "semantic_health_score": snap.semantic_health_score,
        "high_drift_files": snap.high_drift_files,
        "semantic_drift_method": snap.semantic_drift_method,
        "risk_reasons": _parse_json_list(snap.risk_reasons_json),
        "hotspot_persistence_score": snap.hotspot_persistence_score,
        "persistent_hotspots": _parse_json_list(snap.persistent_hotspots_json),
        "top_files": _parse_top_files(snap.top_files_json),
        "computed_at": snap.computed_at,
    }


async def _find_commit(db: AsyncSession, repo_id: int, sha: str | None = None) -> Commit | None:
    query = select(Commit).where(Commit.repo_id == repo_id)
    if sha:
        query = query.where((Commit.sha.like(f"{sha}%")) | (Commit.full_sha.like(f"{sha}%")))
    else:
        query = query.order_by(desc(Commit.committed_at))
    result = await db.execute(query.limit(1))
    return result.scalar_one_or_none()


async def _graph_payload(db: AsyncSession, repo_id: int, commit: Commit) -> dict:
    nodes_result = await db.execute(
        select(GraphNode).where(GraphNode.repo_id == repo_id, GraphNode.commit_id == commit.id)
    )
    edges_result = await db.execute(
        select(GraphEdge).where(GraphEdge.repo_id == repo_id, GraphEdge.commit_id == commit.id)
    )
    nodes = nodes_result.scalars().all()
    edges = edges_result.scalars().all()

    if not nodes:
        fallback_result = await db.execute(
            select(Commit)
            .join(GraphNode, GraphNode.commit_id == Commit.id)
            .where(Commit.repo_id == repo_id, GraphNode.repo_id == repo_id)
            .group_by(Commit.id)
            .order_by(desc(Commit.committed_at))
            .limit(1)
        )
        fallback_commit = fallback_result.scalar_one_or_none()
        if fallback_commit:
            commit = fallback_commit
            nodes_result = await db.execute(
                select(GraphNode).where(
                    GraphNode.repo_id == repo_id, GraphNode.commit_id == commit.id
                )
            )
            edges_result = await db.execute(
                select(GraphEdge).where(
                    GraphEdge.repo_id == repo_id, GraphEdge.commit_id == commit.id
                )
            )
            nodes = nodes_result.scalars().all()
            edges = edges_result.scalars().all()

    return {
        "repo_id": repo_id,
        "commit_sha": commit.sha,
        "nodes": [
            {
                "id": node.file_path,
                "file": node.file_path,
                "module": node.module_name,
                "loc": node.loc,
                "health": node.avg_complexity,
                "health_color": node.health_color,
                "is_entry_point": node.is_entry_point,
                "semantic_drift_score": node.semantic_drift_score,
                "drift_method": node.drift_method,
            }
            for node in nodes
        ],
        "edges": [
            {
                "source": edge.source_file,
                "target": edge.target_file,
                "type": edge.edge_type,
                "weight": edge.weight,
                "cochange_count": edge.cochange_count,
            }
            for edge in edges
        ],
    }


async def _bus_factor_payload(db: AsyncSession, repo_id: int) -> dict:
    result = await db.execute(
        select(BusFactor)
        .where(BusFactor.repo_id == repo_id)
        .order_by(BusFactor.risk_level, desc(BusFactor.top_contributor_pct))
    )
    entries = result.scalars().all()
    return {
        "repo_id": repo_id,
        "modules": [
            {
                "module_path": entry.module_path,
                "contributor_count": entry.contributor_count,
                "top_contributor": entry.top_contributor,
                "top_contributor_email": entry.top_contributor_email,
                "top_contributor_pct": entry.top_contributor_pct,
                "total_commits_to_module": entry.total_commits_to_module,
                "risk_level": entry.risk_level,
                "last_commit_sha": entry.last_commit_sha,
                "last_updated_at": entry.last_updated_at,
            }
            for entry in entries
        ],
    }


async def _commit_graph_rows(
    db: AsyncSession, repo_id: int, sha: str
) -> tuple[list[GraphNode], list[GraphEdge], Commit]:
    commit = await _find_commit(db, repo_id, sha)
    if not commit:
        raise _http_error(404, "Commit not found.", "commit_not_found")
    nodes_result = await db.execute(
        select(GraphNode).where(GraphNode.repo_id == repo_id, GraphNode.commit_id == commit.id)
    )
    edges_result = await db.execute(
        select(GraphEdge).where(GraphEdge.repo_id == repo_id, GraphEdge.commit_id == commit.id)
    )
    return nodes_result.scalars().all(), edges_result.scalars().all(), commit


async def _clear_repo_data(db: AsyncSession, repo_id: int) -> None:
    for model in (
        LLMNarrative,
        GraphEdge,
        GraphNode,
        HealthSnapshot,
        Commit,
        BusFactor,
        PullRequest,
    ):
        await db.execute(delete(model).where(model.repo_id == repo_id))


async def _latest_active_job(db: AsyncSession, repo_id: int) -> AnalysisJob | None:
    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.repo_id == repo_id, AnalysisJob.status.in_(ACTIVE_JOB_STATUSES))
        .order_by(desc(AnalysisJob.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def run_ingestion(
    repo_id: int,
    job_id: int,
    max_commits: int,
    branch: str | None = None,
    exclude_merges: bool = False,
) -> None:
    from backend.database import AsyncSessionLocal, commit_with_retry
    from backend.features.repo_ingestion.metrics_extractor import (
        checkout_commit,
        extract_commit_metrics,
    )

    async with get_ingestion_semaphore():
        # --- early validation (own session, committed immediately) ---
        async with AsyncSessionLocal() as db:
            repo = await db.get(Repo, repo_id)
            if not repo:
                return

            job = await db.get(AnalysisJob, job_id)
            if not job or job.repo_id != repo_id:
                logger.warning(
                    "Skipping ingestion for repo_id=%s because job_id=%s was not found",
                    repo_id,
                    job_id,
                )
                return

            repo_url = repo.url  # snapshot immutable fields for later use

            await _update_job(
                job_id,
                db=db,
                status="cloning",
                current_stage="Cloning repository",
                started_at=datetime.now(tz=timezone.utc),
            )
            repo.status = "processing"
            repo.error_message = None
            await commit_with_retry(db)

        clone_path = None
        try:
            clone_path = await clone_repo(
                repo_url,
                repo_id,
                max_commits,
                branch=branch,
            )
            available_commits = await count_available_commits(clone_path)
            if available_commits < 1:
                raise RuntimeError(
                    f"Repository must have at least 1 commit for CommitIQ analysis; found {available_commits}."
                )

            await _update_job(job_id, status="analyzing", current_stage="Walking commit history")
            await _raise_if_cancelled(job_id)
            commit_history = list(walk_commits(clone_path, max_commits, exclude_merges))
            if not commit_history:
                raise RuntimeError("No commits were found in this repository.")
            await _update_job(job_id, total_commits=len(commit_history))
            await _raise_if_cancelled(job_id)

            await _update_job(
                job_id, status="computing_bus_factor", current_stage="Computing bus factor"
            )
            await _raise_if_cancelled(job_id)
            checkout_commit(clone_path, commit_history[-1]["full_sha"])
            bus_entries = await asyncio.to_thread(
                compute_bus_factor_from_history, commit_history, clone_path
            )
            min_bus_factor = min((entry["contributor_count"] for entry in bus_entries), default=1)

            owner, repo_name = parse_github_url(repo_url)
            pr_list = await fetch_github_pull_requests(owner, repo_name)

            # --- single atomic transaction: clear old data + write all new data ---
            async with AsyncSessionLocal() as db:
                await _clear_repo_data(db, repo_id)
                await _raise_if_cancelled(job_id)

                for pr_data in pr_list:
                    pr = PullRequest(
                        repo_id=repo_id,
                        pr_number=pr_data["pr_number"],
                        title=pr_data["title"],
                        state=pr_data["state"],
                        author=pr_data["author"],
                        created_at=pr_data["created_at"],
                        first_review_at=pr_data["first_review_at"],
                        merged_at=pr_data["merged_at"],
                        closed_at=pr_data["closed_at"],
                    )
                    db.add(pr)

                prev_health = None
                prev_avg_complexity = 0.0

                await _update_job(
                    job_id, status="analyzing", current_stage="Extracting metrics in parallel..."
                )
                loop = asyncio.get_running_loop()
                max_workers = min(8, (os.cpu_count() or 4) + 4)
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    tasks = [
                        loop.run_in_executor(
                            pool, _extract_metrics_in_worktree, clone_path, commit_data
                        )
                        for commit_data in commit_history
                    ]
                    results = await asyncio.gather(*tasks)
                metrics_by_sha = dict(results)

                for idx, commit_data in enumerate(commit_history):
                    await _raise_if_cancelled(job_id)
                    await _update_job(
                        job_id,
                        db=db,
                        status="analyzing",
                        current_stage=f"Analyzing commit {idx + 1}/{len(commit_history)}",
                        processed_commits=idx,
                        current_sha=commit_data["sha"],
                        progress_pct=round(idx / len(commit_history) * 60, 1),
                    )

                    file_metrics_map = metrics_by_sha.get(
                        commit_data["full_sha"],
                        await asyncio.to_thread(extract_commit_metrics, clone_path, commit_data),
                    )
                    top_files = list(file_metrics_map.keys())[:50]
                    import_edges = build_import_edges(clone_path, top_files)
                    cochange_edges = build_cochange_edges(commit_history[: idx + 1])
                    top_set = set(top_files)
                    filtered_edges = []
                    seen_edges = set()
                    for edge in import_edges + cochange_edges:
                        if edge["source_file"] not in top_set or edge["target_file"] not in top_set:
                            continue
                        edge_key = (edge["source_file"], edge["target_file"], edge["edge_type"])
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)
                        filtered_edges.append(edge)

                    hotspot_files = _hotspot_files(commit_history, idx, file_metrics_map)
                    persistent_hotspots = _persistent_hotspots(
                        commit_history, idx, file_metrics_map
                    )
                    dependency_density = len(filtered_edges) / max(len(top_files), 1)
                    has_cycles = _detect_cycles(filtered_edges)
                    snapshot_data = compute_full_snapshot(
                        commit_data=commit_data,
                        file_metrics_map=file_metrics_map,
                        bus_factor_min=min_bus_factor,
                        prev_health=prev_health,
                        prev_avg_complexity=prev_avg_complexity,
                        dependency_density=dependency_density,
                        has_cycles=has_cycles,
                        hotspot_files=hotspot_files,
                        persistent_hotspots=persistent_hotspots,
                    )

                    commit_obj = Commit(
                        repo_id=repo_id,
                        sha=commit_data["sha"],
                        full_sha=commit_data["full_sha"],
                        message=commit_data["message"],
                        author_name=commit_data["author_name"],
                        author_email=commit_data["author_email"],
                        committed_at=datetime.fromisoformat(commit_data["committed_at"]),
                        insertions=commit_data["insertions"],
                        deletions=commit_data["deletions"],
                        files_changed=commit_data["files_changed"],
                        parent_sha=commit_data["parent_sha"],
                    )
                    db.add(commit_obj)
                    await db.flush()

                    snapshot = HealthSnapshot(
                        repo_id=repo_id, commit_id=commit_obj.id, **snapshot_data
                    )
                    db.add(snapshot)

                    for fpath in top_files:
                        metrics = file_metrics_map.get(fpath, {})
                        db.add(
                            GraphNode(
                                repo_id=repo_id,
                                commit_id=commit_obj.id,
                                full_sha=commit_obj.full_sha,
                                file_path=fpath,
                                module_name=(
                                    str(Path(fpath).parent)
                                    if Path(fpath).parent != Path(".")
                                    else None
                                ),
                                loc=metrics.get("loc", 0),
                                avg_complexity=metrics.get("avg_complexity", 0.0),
                                health_color=assign_health_color(
                                    metrics.get("avg_complexity", 0.0)
                                ),
                                is_entry_point=Path(fpath).stem
                                in {"index", "main", "app", "server"},
                                semantic_drift_score=metrics.get("semantic_drift_score", 0.0),
                                drift_method=metrics.get("drift_method", "none"),
                            )
                        )

                    for edge in filtered_edges:
                        db.add(
                            GraphEdge(
                                repo_id=repo_id,
                                commit_id=commit_obj.id,
                                full_sha=commit_obj.full_sha,
                                **edge,
                            )
                        )

                    prev_health = snapshot_data["health_score"]
                    prev_avg_complexity = snapshot_data["avg_complexity"]

                await _update_job(
                    job_id,
                    db=db,
                    status="computing_bus_factor",
                    current_stage="Computing bus factor",
                )
                for entry in bus_entries:
                    db.add(BusFactor(repo_id=repo_id, **entry))

                await commit_with_retry(db)

            # --- mark repo as ready (own session) ---
            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    repo.status = "ready"
                    repo.analyzed_commits = len(commit_history)
                    repo.total_commits = available_commits
                    repo.last_updated_at = datetime.now(tz=timezone.utc)
                    await commit_with_retry(db)

            completed = datetime.now(tz=timezone.utc)
            await _update_job(
                job_id,
                status="ready",
                current_stage="Complete",
                processed_commits=len(commit_history),
                progress_pct=100.0,
                completed_at=completed,
            )
        except IngestionCancelled:
            logger.info("Repository ingestion cancelled for repo_id=%s job_id=%s", repo_id, job_id)
            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    repo.status = "pending"
                    repo.error_message = CANCELLED_MESSAGE
                    await commit_with_retry(db)
            await _update_job(
                job_id,
                status="cancelled",
                current_stage="Cancelled",
                error_message=CANCELLED_MESSAGE,
                completed_at=datetime.now(tz=timezone.utc),
            )
        except Exception as exc:
            logger.exception("Repository ingestion failed for repo_id=%s", repo_id)
            error_msg = str(exc)[:500]
            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    repo.status = "error"
                    repo.error_message = error_msg
                    await commit_with_retry(db)
            await _update_job(
                job_id,
                status="error",
                current_stage="Error",
                error_message=error_msg,
            )
        finally:
            cleanup_repo(repo_id)


async def run_rescan(repo_id: int, job_id: int, max_commits: int) -> None:
    from backend.database import AsyncSessionLocal, commit_with_retry
    from backend.features.repo_ingestion.metrics_extractor import (
        checkout_commit,
        extract_commit_metrics,
    )

    async with get_ingestion_semaphore():
        async with AsyncSessionLocal() as db:
            repo = await db.get(Repo, repo_id)
            if not repo:
                return

            job = await db.get(AnalysisJob, job_id)
            if not job or job.repo_id != repo_id:
                logger.warning(
                    "Skipping rescan for repo_id=%s because job_id=%s was not found",
                    repo_id,
                    job_id,
                )
                return

            repo_url = repo.url

            await _update_job(
                job_id,
                db=db,
                status="cloning",
                current_stage="Fetching remote repository updates",
                started_at=datetime.now(tz=timezone.utc),
            )
            repo.status = "processing"
            repo.error_message = None
            await commit_with_retry(db)

        clone_path = None
        try:
            clone_path = await clone_repo(repo_url, repo_id, max_commits)
            await _raise_if_cancelled(job_id)
            available_commits = await count_available_commits(clone_path)
            if available_commits < 1:
                raise RuntimeError(
                    f"Repository must have at least 1 commit for CommitIQ analysis; found {available_commits}."
                )

            await _update_job(job_id, status="analyzing", current_stage="Checking for new commits")
            await _raise_if_cancelled(job_id)
            commit_history = list(walk_commits(clone_path, max_commits, exclude_merges))
            if not commit_history:
                raise RuntimeError("No commits were found in this repository.")

            # Find existing commit full_shas in DB to isolate new commits
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(Commit.full_sha).where(Commit.repo_id == repo_id))
                existing_shas = set(result.scalars().all())

                # Retrieve prior health state from latest commit in DB
                latest_commit_result = await db.execute(
                    select(Commit, HealthSnapshot)
                    .join(HealthSnapshot, HealthSnapshot.commit_id == Commit.id)
                    .where(Commit.repo_id == repo_id)
                    .order_by(desc(Commit.committed_at))
                    .limit(1)
                )
                latest_row = latest_commit_result.first()
                prev_health = latest_row[1].health_score if latest_row else None
                prev_avg_complexity = latest_row[1].avg_complexity if latest_row else 0.0

            new_commits = [c for c in commit_history if c["full_sha"] not in existing_shas]

            if not new_commits:
                logger.info("Rescan completed for repo_id=%s: no new commits found.", repo_id)
                async with AsyncSessionLocal() as db:
                    repo = await db.get(Repo, repo_id)
                    if repo:
                        repo.status = "ready"
                        repo.total_commits = available_commits
                        repo.last_updated_at = datetime.now(tz=timezone.utc)
                        await commit_with_retry(db)
                completed = datetime.now(tz=timezone.utc)
                await _update_job(
                    job_id,
                    status="ready",
                    current_stage="Complete (No new commits found)",
                    processed_commits=0,
                    progress_pct=100.0,
                    completed_at=completed,
                )
                return

            await _update_job(job_id, total_commits=len(new_commits))
            await _raise_if_cancelled(job_id)

            # Process new commits incrementally
            checkout_commit(clone_path, commit_history[-1]["full_sha"])
            bus_entries = await asyncio.to_thread(
                compute_bus_factor_from_history, commit_history, clone_path
            )
            min_bus_factor = min((entry["contributor_count"] for entry in bus_entries), default=1)

            await _update_job(
                job_id, status="analyzing", current_stage="Extracting metrics in parallel..."
            )
            loop = asyncio.get_running_loop()
            max_workers = min(8, (os.cpu_count() or 4) + 4)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                tasks = [
                    loop.run_in_executor(
                        pool, _extract_metrics_in_worktree, clone_path, commit_data
                    )
                    for commit_data in new_commits
                ]
                results = await asyncio.gather(*tasks)
            metrics_by_sha = dict(results)

            for idx, commit_data in enumerate(new_commits):
                await _raise_if_cancelled(job_id)

                file_metrics_map = metrics_by_sha.get(
                    commit_data["full_sha"],
                    await asyncio.to_thread(extract_commit_metrics, clone_path, commit_data),
                )
                top_files = list(file_metrics_map.keys())[:50]
                import_edges = build_import_edges(clone_path, top_files)

                c_idx = next(
                    (
                        i
                        for i, c in enumerate(commit_history)
                        if c["full_sha"] == commit_data["full_sha"]
                    ),
                    idx,
                )
                cochange_edges = build_cochange_edges(commit_history[: c_idx + 1])
                top_set = set(top_files)
                filtered_edges = []
                seen_edges = set()
                for edge in import_edges + cochange_edges:
                    if edge["source_file"] not in top_set or edge["target_file"] not in top_set:
                        continue
                    edge_key = (edge["source_file"], edge["target_file"], edge["edge_type"])
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    filtered_edges.append(edge)

                hotspot_files = _hotspot_files(commit_history, c_idx, file_metrics_map)
                persistent_hotspots = _persistent_hotspots(commit_history, c_idx, file_metrics_map)
                dependency_density = len(filtered_edges) / max(len(top_files), 1)
                has_cycles = _detect_cycles(filtered_edges)
                snapshot_data = compute_full_snapshot(
                    commit_data=commit_data,
                    file_metrics_map=file_metrics_map,
                    bus_factor_min=min_bus_factor,
                    prev_health=prev_health,
                    prev_avg_complexity=prev_avg_complexity,
                    dependency_density=dependency_density,
                    has_cycles=has_cycles,
                    hotspot_files=hotspot_files,
                    persistent_hotspots=persistent_hotspots,
                )

                async with AsyncSessionLocal() as db:
                    commit_obj = Commit(
                        repo_id=repo_id,
                        sha=commit_data["sha"],
                        full_sha=commit_data["full_sha"],
                        message=commit_data["message"],
                        author_name=commit_data["author_name"],
                        author_email=commit_data["author_email"],
                        committed_at=datetime.fromisoformat(commit_data["committed_at"]),
                        insertions=commit_data["insertions"],
                        deletions=commit_data["deletions"],
                        files_changed=commit_data["files_changed"],
                        parent_sha=commit_data["parent_sha"],
                    )
                    db.add(commit_obj)
                    await db.flush()

                    snapshot = HealthSnapshot(
                        repo_id=repo_id, commit_id=commit_obj.id, **snapshot_data
                    )
                    db.add(snapshot)

                    for fpath in top_files:
                        metrics = file_metrics_map.get(fpath, {})
                        db.add(
                            GraphNode(
                                repo_id=repo_id,
                                commit_id=commit_obj.id,
                                full_sha=commit_obj.full_sha,
                                file_path=fpath,
                                module_name=(
                                    str(Path(fpath).parent)
                                    if Path(fpath).parent != Path(".")
                                    else None
                                ),
                                loc=metrics.get("loc", 0),
                                avg_complexity=metrics.get("avg_complexity", 0.0),
                                health_color=assign_health_color(
                                    metrics.get("avg_complexity", 0.0)
                                ),
                                is_entry_point=Path(fpath).stem
                                in {"index", "main", "app", "server"},
                                semantic_drift_score=metrics.get("semantic_drift_score", 0.0),
                                drift_method=metrics.get("drift_method", "none"),
                            )
                        )

                    for edge in filtered_edges:
                        db.add(
                            GraphEdge(
                                repo_id=repo_id,
                                commit_id=commit_obj.id,
                                full_sha=commit_obj.full_sha,
                                **edge,
                            )
                        )

                    job = await db.get(AnalysisJob, job_id)
                    if job:
                        job.status = "analyzing"
                        job.current_stage = f"Analyzing new commit {idx + 1}/{len(new_commits)}"
                        job.processed_commits = idx + 1
                        job.current_sha = commit_data["sha"]
                        job.progress_pct = round((idx + 1) / len(new_commits) * 80, 1)

                    await commit_with_retry(db)

                prev_health = snapshot_data["health_score"]
                prev_avg_complexity = snapshot_data["avg_complexity"]

            async with AsyncSessionLocal() as db:
                await db.execute(delete(BusFactor).where(BusFactor.repo_id == repo_id))
                for entry in bus_entries:
                    db.add(BusFactor(repo_id=repo_id, **entry))
                await commit_with_retry(db)

            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    total_db_commits = (
                        (await db.execute(select(Commit).where(Commit.repo_id == repo_id)))
                        .scalars()
                        .all()
                    )
                    repo.status = "ready"
                    repo.analyzed_commits = len(total_db_commits)
                    repo.total_commits = available_commits
                    repo.last_updated_at = datetime.now(tz=timezone.utc)
                    await commit_with_retry(db)

            completed = datetime.now(tz=timezone.utc)
            await _update_job(
                job_id,
                status="ready",
                current_stage="Complete",
                processed_commits=len(new_commits),
                progress_pct=100.0,
                completed_at=completed,
            )
        except IngestionCancelled:
            logger.info("Repository rescan cancelled for repo_id=%s job_id=%s", repo_id, job_id)
            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    repo.status = "ready"
                    await commit_with_retry(db)
            await _update_job(
                job_id,
                status="cancelled",
                current_stage="Cancelled",
                error_message=CANCELLED_MESSAGE,
                completed_at=datetime.now(tz=timezone.utc),
            )
        except Exception as exc:
            logger.exception("Repository rescan failed for repo_id=%s", repo_id)
            error_msg = str(exc)[:500]
            async with AsyncSessionLocal() as db:
                repo = await db.get(Repo, repo_id)
                if repo:
                    repo.status = "error"
                    repo.error_message = error_msg
                    await commit_with_retry(db)
            await _update_job(
                job_id,
                status="error",
                current_stage="Error",
                error_message=error_msg,
            )
        finally:
            cleanup_repo(repo_id)


async def _find_repo_by_slug(db: AsyncSession, slug: str) -> Repo | None:
    normalized = slug.strip().lower()
    result = await db.execute(select(Repo).where(func.lower(Repo.repo_slug) == normalized))
    return result.scalar_one_or_none()


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_repo(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    cleaned_repo_url = sanitize_repo_url(request.repo_url)
    normalized_repo_url = cleaned_repo_url.strip().lower()

    try:
        owner, repo_name = parse_github_url(normalized_repo_url)
    except ValueError as exc:
        raise _http_error(400, str(exc), "invalid_repo_url")

    owner = owner.strip().lower()
    repo_name = repo_name.strip().lower()

    url = f"https://github.com/{owner}/{repo_name}"
    repo_slug = make_repo_slug(owner, repo_name)
    max_c = request.max_commits or MAX_COMMITS

    repo = await _find_repo_by_slug(db, repo_slug)

    if repo:
        active_job = await _latest_active_job(db, repo.id)
        if active_job:
            return IngestResponse(
                repo_id=repo.id,
                repo_slug=repo.repo_slug,
                status="processing",
                job_id=active_job.id,
                message=f"Ingestion already in progress. Poll /api/repos/ingest/progress/{repo.id} for updates.",
            )

    if not repo:
        metadata = await fetch_github_metadata(owner, repo_name)
        repo = Repo(
            url=url,
            name=f"{owner}/{repo_name}",
            owner=owner,
            repo_slug=repo_slug,
            status="pending",
            max_commits_setting=max_c,
            **metadata,
        )
        db.add(repo)
        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
            repo = await _find_repo_by_slug(db, repo_slug)
            if not repo:
                raise
    else:
        metadata = await fetch_github_metadata(owner, repo_name)
        repo.url = url
        repo.name = f"{owner}/{repo_name}"
        repo.owner = owner
        repo.status = "pending"
        repo.error_message = None
        repo.max_commits_setting = max_c
        for k, v in metadata.items():
            setattr(repo, k, v)

    job = AnalysisJob(repo_id=repo.id, status="queued", triggered_by="user")
    db.add(job)
    await commit_with_retry(db)
    await db.refresh(repo)
    await db.refresh(job)

    background_tasks.add_task(
        run_ingestion,
        repo.id,
        job.id,
        request.max_commits,
        request.branch,
        request.exclude_merges,
    )
    return IngestResponse(
        repo_id=repo.id,
        repo_slug=repo.repo_slug,
        status="processing",
        job_id=job.id,
        message=f"Ingestion started. Poll /api/repos/ingest/progress/{repo.id} for updates.",
    )


@router.post("/{repo_id}/rescan", response_model=RescanResponse, status_code=202)
async def rescan_repo(
    repo_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")

    active_job = await _latest_active_job(db, repo.id)
    if active_job:
        return RescanResponse(
            repo_id=repo.id,
            repo_slug=repo.repo_slug,
            status="processing",
            job_id=active_job.id,
            message=f"Analysis update already in progress. Poll /api/repos/ingest/progress/{repo.id} for updates.",
        )

    repo.status = "pending"
    repo.error_message = None

    job = AnalysisJob(repo_id=repo.id, status="queued", triggered_by="rescan")
    db.add(job)
    await commit_with_retry(db)
    await db.refresh(repo)
    await db.refresh(job)

    max_c = repo.max_commits_setting or MAX_COMMITS
    background_tasks.add_task(run_rescan, repo.id, job.id, max_c)

    return RescanResponse(
        repo_id=repo.id,
        repo_slug=repo.repo_slug,
        status="processing",
        job_id=job.id,
        message=f"Rescan started. Poll /api/repos/ingest/progress/{repo.id} for updates.",
    )


@router.post("/ingest/cancel/{repo_id}", response_model=JobProgressOut)
async def cancel_ingestion(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")

    job = await _latest_active_job(db, repo_id)
    if not job:
        raise _http_error(404, "No active ingestion job found.", "job_not_found")

    completed = datetime.now(tz=timezone.utc)
    job.status = "cancelled"
    job.current_stage = "Cancelled"
    job.error_message = CANCELLED_MESSAGE
    job.completed_at = completed
    if job.started_at and not job.duration_seconds:
        job.duration_seconds = _duration_seconds(job.started_at, completed)
    repo.status = "pending"
    repo.error_message = CANCELLED_MESSAGE
    await commit_with_retry(db)
    await db.refresh(job)

    return JobProgressOut(
        current=job.processed_commits,
        total=job.total_commits,
        current_sha=job.current_sha,
        stage=job.current_stage,
        progress_pct=job.progress_pct,
        status=job.status,
        error_message=job.error_message,
    )


@router.get("/ingest/progress/{repo_id}", response_model=None)
async def ingest_progress(repo_id: int):
    async def event_generator():
        from backend.database import AsyncSessionLocal

        try:
            while True:
                async with AsyncSessionLocal() as stream_db:
                    result = await stream_db.execute(
                        select(AnalysisJob)
                        .where(AnalysisJob.repo_id == repo_id)
                        .order_by(desc(AnalysisJob.created_at))
                        .limit(1)
                    )
                    job = result.scalar_one_or_none()

                    if not job:
                        yield f"data: {json.dumps({'status': 'error', 'error_message': 'Job not found'})}\n\n"
                        break

                    payload = JobProgressOut(
                        current=job.processed_commits,
                        total=job.total_commits,
                        current_sha=job.current_sha,
                        stage=job.current_stage,
                        progress_pct=job.progress_pct,
                        status=job.status,
                        error_message=job.error_message,
                    ).model_dump(mode="json")
                    terminal = job.status in {"ready", "error", "cancelled"}

                yield f"data: {json.dumps(payload)}\n\n"

                if terminal:
                    break

                await asyncio.sleep(0.75)
        except asyncio.CancelledError:
            logger.debug("Ingestion progress stream disconnected for repo_id=%d", int(repo_id))
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _count_active_contributors(db: AsyncSession, repo_id: int) -> int:
    result = await db.execute(
        select(
            func.count(func.distinct(func.coalesce(Commit.author_email, Commit.author_name)))
        ).where(Commit.repo_id == repo_id)
    )
    return result.scalar() or 0


async def _active_contributors_map(db: AsyncSession, repo_ids: list[int]) -> dict[int, int]:
    if not repo_ids:
        return {}
    result = await db.execute(
        select(
            Commit.repo_id,
            func.count(func.distinct(func.coalesce(Commit.author_email, Commit.author_name))),
        )
        .where(Commit.repo_id.in_(repo_ids))
        .group_by(Commit.repo_id)
    )
    return {repo_id: count for repo_id, count in result.all()}


async def _latest_completed_job_time(db: AsyncSession, repo_id: int) -> datetime | None:
    result = await db.execute(
        select(AnalysisJob.completed_at)
        .where(
            AnalysisJob.repo_id == repo_id,
            AnalysisJob.status == "ready",
            AnalysisJob.completed_at.isnot(None),
        )
        .order_by(desc(AnalysisJob.completed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _latest_completed_job_map(db: AsyncSession, repo_ids: list[int]) -> dict[int, datetime]:
    if not repo_ids:
        return {}
    subquery = (
        select(
            AnalysisJob.repo_id,
            func.max(AnalysisJob.completed_at).label("max_completed_at"),
        )
        .where(
            AnalysisJob.repo_id.in_(repo_ids),
            AnalysisJob.status == "ready",
            AnalysisJob.completed_at.isnot(None),
        )
        .group_by(AnalysisJob.repo_id)
        .subquery()
    )
    result = await db.execute(select(subquery.c.repo_id, subquery.c.max_completed_at))
    return {repo_id: completed_at for repo_id, completed_at in result.all()}


def _repo_to_out(
    repo: Repo,
    active_contributors_count: int = 0,
    last_job_completed_at: datetime | None = None,
) -> RepoOut:
    out = RepoOut.model_validate(repo)
    out.active_contributors_count = active_contributors_count
    out.last_job_completed_at = last_job_completed_at or repo.last_updated_at
    return out


@router.get("", response_model=list[RepoOut])
async def list_repos(
    slug: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_db),
):
    if limit < 1:
        raise _http_error(422, "limit must be >= 1", "validation_error")
    if offset < 0:
        raise _http_error(422, "offset must be >= 0", "validation_error")
    if limit > 100:
        limit = 100
    query = select(Repo)
    if slug:
        query = query.where(func.lower(Repo.repo_slug) == slug.strip().lower())
    query = query.order_by(desc(Repo.ingested_at)).limit(limit).offset(offset)
    result = await db.execute(query)
    repos = result.scalars().all()
    repo_ids = [repo.id for repo in repos]
    contrib_map = await _active_contributors_map(db, repo_ids)
    completed_job_map = await _latest_completed_job_map(db, repo_ids)
    return [
        _repo_to_out(repo, contrib_map.get(repo.id, 0), completed_job_map.get(repo.id))
        for repo in repos
    ]


@router.get("/by-slug/{slug}", response_model=RepoOut)
async def get_repo_by_slug(slug: str, db: AsyncSession = Depends(get_db)):
    repo = await _find_repo_by_slug(db, slug)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    count = await _count_active_contributors(db, repo.id)
    last_completed = await _latest_completed_job_time(db, repo.id)
    return _repo_to_out(repo, count, last_completed)


@router.get("/{repo_id}", response_model=RepoOut)
async def get_repo(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    count = await _count_active_contributors(db, repo.id)
    last_completed = await _latest_completed_job_time(db, repo.id)
    return _repo_to_out(repo, count, last_completed)


async def _build_repo_compare_item(db: AsyncSession, repo: Repo) -> RepoCompareItem:
    contrib_count = await _count_active_contributors(db, repo.id)
    repo_out = _repo_to_out(repo, contrib_count)

    result = await db.execute(
        select(Commit, HealthSnapshot)
        .join(HealthSnapshot, HealthSnapshot.commit_id == Commit.id)
        .where(Commit.repo_id == repo.id)
        .order_by(Commit.committed_at)
    )
    rows = result.all()
    snapshots_chronological = [_snapshot_payload(commit, snap) for commit, snap in rows]
    latest_snapshot = snapshots_chronological[-1] if snapshots_chronological else None

    bus_factor = await _bus_factor_payload(db, repo.id)
    min_bus = min((m["contributor_count"] for m in bus_factor.get("modules", [])), default=1)

    if latest_snapshot:
        metrics_summary = RepoCompareMetrics(
            health_score=round(latest_snapshot["health_score"], 1),
            avg_complexity=round(latest_snapshot["avg_complexity"], 2),
            max_complexity=round(latest_snapshot["max_complexity"], 2),
            churn_rate=round(latest_snapshot["churn_rate"], 4),
            total_loc=latest_snapshot["total_loc"],
            bus_factor_min=latest_snapshot["bus_factor_min"],
            hotspot_count=latest_snapshot["hotspot_count"],
            active_contributors=contrib_count,
            total_commits=repo.total_commits,
            analyzed_commits=repo.analyzed_commits,
            dependency_density=round(latest_snapshot.get("dependency_density", 0.0), 2),
            has_cycles=bool(latest_snapshot.get("has_cycles", False)),
            avg_semantic_drift=round(latest_snapshot.get("avg_semantic_drift", 0.0), 2),
            cc_score=round(latest_snapshot.get("cc_score", 0.0), 1),
            churn_score=round(latest_snapshot.get("churn_score", 0.0), 1),
            bus_score=round(latest_snapshot.get("bus_score", 0.0), 1),
            loc_score=round(latest_snapshot.get("loc_score", 0.0), 1),
            semantic_health_score=round(latest_snapshot.get("semantic_health_score", 100.0), 1),
        )
    else:
        metrics_summary = RepoCompareMetrics(
            active_contributors=contrib_count,
            total_commits=repo.total_commits,
            analyzed_commits=repo.analyzed_commits,
            bus_factor_min=min_bus,
        )

    return RepoCompareItem(
        repo=repo_out,
        latest_snapshot=latest_snapshot,
        metrics_summary=metrics_summary,
        bus_factor=bus_factor,
        timeline_summary=snapshots_chronological[-30:],
    )


def _generate_comparison_insights(
    base: RepoCompareItem, head: RepoCompareItem
) -> tuple[list[RepoCompareInsight], str]:
    insights: list[RepoCompareInsight] = []
    b_m = base.metrics_summary
    h_m = head.metrics_summary
    b_name = base.repo.name
    h_name = head.repo.name

    # Health score insight
    if h_m.health_score > b_m.health_score + 2.0:
        insights.append(
            RepoCompareInsight(
                category="Health Score",
                winner=head.repo.repo_slug,
                summary=f"{h_name} has a higher health score ({h_m.health_score:.1f} vs {b_m.health_score:.1f}).",
            )
        )
    elif b_m.health_score > h_m.health_score + 2.0:
        insights.append(
            RepoCompareInsight(
                category="Health Score",
                winner=base.repo.repo_slug,
                summary=f"{b_name} has a higher health score ({b_m.health_score:.1f} vs {h_m.health_score:.1f}).",
            )
        )
    else:
        insights.append(
            RepoCompareInsight(
                category="Health Score",
                winner=None,
                summary=f"Both repositories maintain comparable health scores ({h_m.health_score:.1f} vs {b_m.health_score:.1f}).",
            )
        )

    # Complexity insight (lower complexity is better)
    if b_m.avg_complexity > 0 and h_m.avg_complexity > 0:
        if h_m.avg_complexity < b_m.avg_complexity - 0.5:
            insights.append(
                RepoCompareInsight(
                    category="Code Complexity",
                    winner=head.repo.repo_slug,
                    summary=f"{h_name} shows lower cyclomatic complexity ({h_m.avg_complexity:.1f} vs {b_m.avg_complexity:.1f}).",
                )
            )
        elif b_m.avg_complexity < h_m.avg_complexity - 0.5:
            insights.append(
                RepoCompareInsight(
                    category="Code Complexity",
                    winner=base.repo.repo_slug,
                    summary=f"{b_name} shows lower cyclomatic complexity ({b_m.avg_complexity:.1f} vs {h_m.avg_complexity:.1f}).",
                )
            )
        else:
            insights.append(
                RepoCompareInsight(
                    category="Code Complexity",
                    winner=None,
                    summary=f"Both repositories exhibit similar average cyclomatic complexity ({h_m.avg_complexity:.1f} vs {b_m.avg_complexity:.1f}).",
                )
            )

    # Bus factor insight (higher bus factor is better)
    if h_m.bus_factor_min > b_m.bus_factor_min:
        insights.append(
            RepoCompareInsight(
                category="Bus Factor Resilience",
                winner=head.repo.repo_slug,
                summary=f"{h_name} has better key-module contributor redundancy (min bus factor {h_m.bus_factor_min} vs {b_m.bus_factor_min}).",
            )
        )
    elif b_m.bus_factor_min > h_m.bus_factor_min:
        insights.append(
            RepoCompareInsight(
                category="Bus Factor Resilience",
                winner=base.repo.repo_slug,
                summary=f"{b_name} has better key-module contributor redundancy (min bus factor {b_m.bus_factor_min} vs {h_m.bus_factor_min}).",
            )
        )
    else:
        insights.append(
            RepoCompareInsight(
                category="Bus Factor Resilience",
                winner=None,
                summary=f"Both repositories have identical minimum bus factors ({h_m.bus_factor_min}).",
            )
        )

    # Churn & Risk insight (fewer hotspots is better)
    if h_m.hotspot_count < b_m.hotspot_count:
        insights.append(
            RepoCompareInsight(
                category="Hotspots & Churn",
                winner=head.repo.repo_slug,
                summary=f"{h_name} has fewer high-churn hotspots ({h_m.hotspot_count} vs {b_m.hotspot_count}).",
            )
        )
    elif b_m.hotspot_count < h_m.hotspot_count:
        insights.append(
            RepoCompareInsight(
                category="Hotspots & Churn",
                winner=base.repo.repo_slug,
                summary=f"{b_name} has fewer high-churn hotspots ({b_m.hotspot_count} vs {h_m.hotspot_count}).",
            )
        )

    # Activity insight
    if h_m.active_contributors > b_m.active_contributors:
        insights.append(
            RepoCompareInsight(
                category="Team Activity",
                winner=head.repo.repo_slug,
                summary=f"{h_name} has more active contributors ({h_m.active_contributors} vs {b_m.active_contributors}).",
            )
        )
    elif b_m.active_contributors > h_m.active_contributors:
        insights.append(
            RepoCompareInsight(
                category="Team Activity",
                winner=base.repo.repo_slug,
                summary=f"{b_name} has more active contributors ({b_m.active_contributors} vs {h_m.active_contributors}).",
            )
        )

    # Overall verdict
    score_diff = round(h_m.health_score - b_m.health_score, 1)
    if abs(score_diff) < 2.0:
        verdict = f"{h_name} and {b_name} have balanced overall health profiles with minimal score variance ({h_m.health_score:.1f} vs {b_m.health_score:.1f})."
    elif score_diff > 0:
        verdict = f"{h_name} demonstrates superior overall health (+{score_diff:.1f} pts) compared to {b_name}."
    else:
        verdict = f"{b_name} demonstrates superior overall health (+{abs(score_diff):.1f} pts) compared to {h_name}."

    return insights, verdict


@router.get("/compare", response_model=RepoCompareResponse)
async def compare_repos(
    base: str | None = Query(default=None, description="Base repository slug"),
    head: str | None = Query(default=None, description="Head repository slug"),
    base_slug: str | None = Query(default=None, description="Alias for base repository slug"),
    head_slug: str | None = Query(default=None, description="Alias for head repository slug"),
    slug1: str | None = Query(default=None, description="Alias for base repository slug"),
    slug2: str | None = Query(default=None, description="Alias for head repository slug"),
    repo1: str | None = Query(default=None, description="Alias for base repository slug"),
    repo2: str | None = Query(default=None, description="Alias for head repository slug"),
    db: AsyncSession = Depends(get_db),
):
    def _val(param) -> str | None:
        if isinstance(param, str) and param.strip():
            return param.strip()
        return None

    raw_base = _val(base) or _val(base_slug) or _val(slug1) or _val(repo1)
    raw_head = _val(head) or _val(head_slug) or _val(slug2) or _val(repo2)

    if not raw_base or not raw_head:
        raise _http_error(
            422,
            "Two repository slugs must be specified for comparison (e.g. /api/repos/compare?base=repo1&head=repo2).",
            "validation_error",
        )

    def _normalize_slug(slug_val: str) -> str:
        s = slug_val.strip()
        s = re.sub(r"^(?:https?://)?(?:www\.)?github\.com/", "", s, flags=re.IGNORECASE)
        while s.endswith("/") or s.endswith(".git"):
            s = s[:-1] if s.endswith("/") else s[:-4]
        return s.strip("/").replace("/", "_").lower()

    clean_base = _normalize_slug(raw_base)
    clean_head = _normalize_slug(raw_head)

    base_repo_res = await db.execute(
        select(Repo).where((Repo.repo_slug == clean_base) | (Repo.repo_slug == raw_base.strip()))
    )
    base_repo = base_repo_res.scalar_one_or_none()
    if not base_repo:
        raise _http_error(404, f"Base repository '{raw_base}' not found.", "repo_not_found")

    head_repo_res = await db.execute(
        select(Repo).where((Repo.repo_slug == clean_head) | (Repo.repo_slug == raw_head.strip()))
    )
    head_repo = head_repo_res.scalar_one_or_none()
    if not head_repo:
        raise _http_error(404, f"Head repository '{raw_head}' not found.", "repo_not_found")

    base_item = await _build_repo_compare_item(db, base_repo)
    head_item = await _build_repo_compare_item(db, head_repo)

    b_m = base_item.metrics_summary
    h_m = head_item.metrics_summary

    deltas = RepoCompareDelta(
        health_score_delta=round(h_m.health_score - b_m.health_score, 1),
        avg_complexity_delta=round(h_m.avg_complexity - b_m.avg_complexity, 2),
        max_complexity_delta=round(h_m.max_complexity - b_m.max_complexity, 2),
        churn_rate_delta=round(h_m.churn_rate - b_m.churn_rate, 4),
        total_loc_delta=h_m.total_loc - b_m.total_loc,
        bus_factor_min_delta=h_m.bus_factor_min - b_m.bus_factor_min,
        hotspot_count_delta=h_m.hotspot_count - b_m.hotspot_count,
        active_contributors_delta=h_m.active_contributors - b_m.active_contributors,
        total_commits_delta=h_m.total_commits - b_m.total_commits,
    )

    insights, verdict = _generate_comparison_insights(base_item, head_item)

    return RepoCompareResponse(
        base=base_item,
        head=head_item,
        deltas=deltas,
        insights=insights,
        verdict=verdict,
    )


@router.get("/by-slug/{slug}", response_model=RepoOut)
async def get_repo_by_slug(slug: str, db: AsyncSession = Depends(get_db)):
    repo = await _find_repo_by_slug(db, slug)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    count = await _count_active_contributors(db, repo.id)
    return _repo_to_out(repo, count)


@router.get("/{repo_id}", response_model=RepoOut)
async def get_repo(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    count = await _count_active_contributors(db, repo.id)
    return _repo_to_out(repo, count)


@router.delete("/{repo_id}", status_code=204)
async def delete_repo(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")

    active_job = await _latest_active_job(db, repo_id)
    if active_job:
        raise _http_error(
            409,
            "Cannot delete a repository while an analysis job is in progress.",
            "repo_ingestion_active",
        )

    await db.delete(repo)
    await commit_with_retry(db)
    cleanup_repo(repo_id)
    return None


@router.get("/{repo_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    repo_id: int,
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(Commit, HealthSnapshot)
        .join(HealthSnapshot, HealthSnapshot.commit_id == Commit.id)
        .where(Commit.repo_id == repo_id)
    )
    if isinstance(start_date, datetime):
        query = query.where(Commit.committed_at >= start_date)
    if isinstance(end_date, datetime):
        query = query.where(Commit.committed_at <= end_date)
    query = query.order_by(Commit.committed_at)
    result = await db.execute(query)
    return {
        "repo_id": repo_id,
        "commits": [_snapshot_payload(commit, snap) for commit, snap in result.all()],
    }


@router.get("/{repo_id}/commit/{sha}", response_model=CommitDetailResponse)
async def get_commit_detail(repo_id: int, sha: str, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    commit = await _find_commit(db, repo_id, sha)
    if not commit:
        raise _http_error(404, "Commit not found.", "commit_not_found")

    snap_result = await db.execute(
        select(HealthSnapshot).where(HealthSnapshot.commit_id == commit.id)
    )
    snapshot = snap_result.scalar_one_or_none()
    if not snapshot:
        raise _http_error(404, "Health snapshot not found for commit.", "snapshot_not_found")

    narrative_key = make_cache_key(repo_id, commit.full_sha, "explain_drop")
    narrative_result = await db.execute(
        select(LLMNarrative).where(LLMNarrative.cache_key == narrative_key)
    )
    narrative = narrative_result.scalar_one_or_none()

    narrative_payload = None
    if narrative:
        narrative_payload = {
            "repo_id": repo_id,
            "commit_sha": commit.sha,
            "prompt_type": narrative.prompt_type,
            "explanation": narrative.response_text,
            "tokens_used": narrative.tokens_input + narrative.tokens_output,
            "cost_usd": narrative.cost_usd,
            "cached": True,
            "model": narrative.model_used,
            "demo_mode": False,
        }

    count = await _count_active_contributors(db, repo.id)
    return {
        "repo": _repo_to_out(repo, count),
        "commit": commit,
        "snapshot": _snapshot_payload(commit, snapshot),
        "graph": await _graph_payload(db, repo_id, commit),
        "bus_factor": await _bus_factor_payload(db, repo_id),
        "has_narrative": narrative is not None,
        "narrative": narrative_payload,
    }


@router.get("/{repo_id}/graph", response_model=GraphResponse)
async def get_graph(repo_id: int, sha: str | None = None, db: AsyncSession = Depends(get_db)):
    commit = await _find_commit(db, repo_id, sha)
    if not commit:
        raise _http_error(404, "Commit not found.", "commit_not_found")
    return await _graph_payload(db, repo_id, commit)


@router.get("/{repo_id}/graph/diff")
async def get_graph_diff(
    repo_id: int,
    sha_before: str,
    sha_after: str,
    db: AsyncSession = Depends(get_db),
):
    before_nodes, before_edges, _ = await _commit_graph_rows(db, repo_id, sha_before)
    after_nodes, after_edges, _ = await _commit_graph_rows(db, repo_id, sha_after)

    before_files = {node.file_path: node for node in before_nodes}
    after_files = {node.file_path: node for node in after_nodes}

    nodes_added = sorted(fpath for fpath in after_files if fpath not in before_files)
    nodes_removed = sorted(fpath for fpath in before_files if fpath not in after_files)

    nodes_changed = []
    for fpath in set(before_files) & set(after_files):
        before_cx = before_files[fpath].avg_complexity or 0.0
        after_cx = after_files[fpath].avg_complexity or 0.0
        if before_cx > 0:
            delta_pct = (after_cx - before_cx) / before_cx
            if abs(delta_pct) > 0.10:
                nodes_changed.append(
                    {
                        "file": fpath,
                        "before_complexity": round(before_cx, 2),
                        "after_complexity": round(after_cx, 2),
                        "delta_pct": round(delta_pct * 100.0, 1),
                    }
                )

    before_edge_set = {
        (edge.source_file, edge.target_file, edge.edge_type) for edge in before_edges
    }
    after_edge_set = {(edge.source_file, edge.target_file, edge.edge_type) for edge in after_edges}
    added_edges = sorted(after_edge_set - before_edge_set)
    removed_edges = sorted(before_edge_set - after_edge_set)

    return {
        "sha_before": sha_before,
        "sha_after": sha_after,
        "summary": {
            "files_added": len(nodes_added),
            "files_removed": len(nodes_removed),
            "files_changed": len(nodes_changed),
            "edges_added": len(added_edges),
            "edges_removed": len(removed_edges),
        },
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_changed": sorted(
            nodes_changed, key=lambda item: abs(item["delta_pct"]), reverse=True
        )[:20],
        "edges_added": [
            {"source": source, "target": target, "type": edge_type}
            for source, target, edge_type in added_edges[:30]
        ],
        "edges_removed": [
            {"source": source, "target": target, "type": edge_type}
            for source, target, edge_type in removed_edges[:30]
        ],
    }


@router.get("/{repo_id}/bus-factor", response_model=BusFactorWrapper)
async def get_bus_factor(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await db.get(Repo, repo_id)
    if not repo:
        raise _http_error(404, "Repository not found.", "repo_not_found")
    return await _bus_factor_payload(db, repo_id)


@router.get("/{repo_id}/hotspots", response_model=HotspotResponse)
async def get_hotspots(
    repo_id: int,
    sha: str | None = None,
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    limit_val = limit.default if hasattr(limit, "default") else limit
    offset_val = offset.default if hasattr(offset, "default") else offset

    commit = await _find_commit(db, repo_id, sha)
    if not commit:
        raise _http_error(404, "Commit not found.", "commit_not_found")

    nodes_result = await db.execute(
        select(GraphNode).where(
            GraphNode.repo_id == repo_id,
            GraphNode.commit_id == commit.id,
            GraphNode.avg_complexity > 5.0,
        )
    )
    nodes = nodes_result.scalars().all()
    if not nodes:
        return {
            "repo_id": repo_id,
            "commit_sha": commit.sha,
            "hotspots": [],
            "total": 0,
            "limit": limit_val,
            "offset": offset_val,
        }

    file_paths = [node.file_path for node in nodes]
    commits_query = select(Commit).where(Commit.repo_id == repo_id)
    if isinstance(start_date, datetime):
        commits_query = commits_query.where(Commit.committed_at >= start_date)
    if isinstance(end_date, datetime):
        commits_query = commits_query.where(Commit.committed_at <= end_date)

    if not isinstance(start_date, datetime) and not isinstance(end_date, datetime):
        commits_query = commits_query.order_by(desc(Commit.committed_at)).limit(20)
    else:
        commits_query = commits_query.order_by(desc(Commit.committed_at))

    commits_result = await db.execute(commits_query)
    recent_commits = commits_result.scalars().all()
    churn_counts = {fpath: 0 for fpath in file_paths}
    for recent in recent_commits:
        # Querying full historical file lists is not stored; approximate churn by graph node presence.
        node_result = await db.execute(
            select(GraphNode.file_path).where(
                GraphNode.repo_id == repo_id, GraphNode.commit_id == recent.id
            )
        )
        recent_paths = set(node_result.scalars().all())
        for fpath in file_paths:
            if fpath in recent_paths:
                churn_counts[fpath] += 1

    hotspots = []
    for node in nodes:
        churn_count = churn_counts.get(node.file_path, 0)
        if churn_count <= 2:
            continue
        risk_score = min(100.0, node.avg_complexity * 8.0 + churn_count * 6.0)
        hotspots.append(
            {
                "file": node.file_path,
                "complexity": round(node.avg_complexity, 2),
                "churn_count": churn_count,
                "risk_score": round(risk_score, 1),
                "loc": node.loc,
            }
        )

    sorted_hotspots = sorted(hotspots, key=lambda item: item["risk_score"], reverse=True)
    total = len(sorted_hotspots)
    paginated_hotspots = sorted_hotspots[offset_val : offset_val + limit_val]

    return {
        "repo_id": repo_id,
        "commit_sha": commit.sha,
        "hotspots": paginated_hotspots,
        "total": total,
        "limit": limit_val,
        "offset": offset_val,
    }


@router.get("/{repo_id}/llm-usage", response_model=LLMUsageOut)
async def get_llm_usage(repo_id: int, db: AsyncSession = Depends(get_db)):
    from backend.features.llm_analysis.cost_guard import get_usage_summary

    return await get_usage_summary(repo_id, db)
