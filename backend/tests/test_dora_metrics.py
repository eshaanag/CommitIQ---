from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import Base, get_db
from backend.features.metrics.dora import compute_dora_metrics
from backend.main import app
from backend.shared.models import Deployment, PullRequest, Repo


@pytest.fixture()
def anyio_backend():
    return "asyncio"


class AsyncSessionAdapter:
    def __init__(self, session: Session):
        self.session = session

    async def execute(self, *args, **kwargs):
        return self.session.execute(*args, **kwargs)

    async def get(self, *args, **kwargs):
        return self.session.get(*args, **kwargs)

    def add(self, *args, **kwargs):
        return self.session.add(*args, **kwargs)

    async def flush(self):
        return self.session.flush()

    async def commit(self):
        return self.session.commit()

    async def refresh(self, instance):
        return self.session.refresh(instance)


@pytest.fixture()
async def db_session() -> AsyncIterator[AsyncSessionAdapter]:
    engine = create_engine("sqlite:///:memory:")
    session_factory = sessionmaker(engine, expire_on_commit=False)

    Base.metadata.create_all(engine)

    with session_factory() as session:
        adapter = AsyncSessionAdapter(session)
        app.dependency_overrides[get_db] = lambda: adapter
        yield adapter
        app.dependency_overrides.clear()
    engine.dispose()


@pytest.mark.anyio
async def test_compute_dora_metrics_with_time_window_deployments(db_session: AsyncSessionAdapter):
    repo = Repo(
        url="https://github.com/org/sample-dora",
        name="org/sample-dora",
        owner="org",
        repo_slug="org-sample-dora",
        status="ready",
    )
    db_session.add(repo)
    await db_session.flush()

    # Jan 2026 deployment (outside window)
    dep1 = Deployment(
        repo_id=repo.id,
        provider="github",
        environment="production",
        status="success",
        deployed_at=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
    )
    # Mar 2026 deployments (inside window)
    dep2 = Deployment(
        repo_id=repo.id,
        provider="github",
        environment="production",
        status="success",
        deployed_at=datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc),
    )
    dep3 = Deployment(
        repo_id=repo.id,
        provider="github",
        environment="production",
        status="success",
        deployed_at=datetime(2026, 3, 19, 14, 0, tzinfo=timezone.utc),
    )
    # May 2026 deployment (outside window)
    dep4 = Deployment(
        repo_id=repo.id,
        provider="github",
        environment="production",
        status="success",
        deployed_at=datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc),
    )
    db_session.add(dep1)
    db_session.add(dep2)
    db_session.add(dep3)
    db_session.add(dep4)
    await db_session.commit()

    # Full time range
    all_dora = await compute_dora_metrics(db_session, repo.id)
    assert all_dora["deployment_frequency"] in ["Elite", "High", "Medium", "Low"]

    # Filtered strictly to March 2026 window
    march_start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    march_end = datetime(2026, 3, 31, 23, 59, tzinfo=timezone.utc)
    march_dora = await compute_dora_metrics(
        db_session, repo.id, start_date=march_start, end_date=march_end
    )
    # 2 deployments in ~4.4 weeks => ~0.45/week
    assert march_dora["deployment_frequency_value"] > 0.0

    # Filtered to an empty window
    empty_dora = await compute_dora_metrics(
        db_session,
        repo.id,
        start_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    assert empty_dora["deployment_frequency_value"] == 0.0
    assert empty_dora["dora_score"] == "Low"


@pytest.mark.anyio
async def test_compute_dora_metrics_with_pull_requests_and_string_iso(
    db_session: AsyncSessionAdapter,
):
    repo = Repo(
        url="https://github.com/org/pr-dora",
        name="org/pr-dora",
        owner="org",
        repo_slug="org-pr-dora",
        status="ready",
    )
    db_session.add(repo)
    await db_session.flush()

    pr1 = PullRequest(
        repo_id=repo.id,
        pr_number=101,
        title="feat: core architecture",
        state="merged",
        author="alice",
        created_at=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
        merged_at=datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc),
    )
    pr2 = PullRequest(
        repo_id=repo.id,
        pr_number=102,
        title="fix: critical hotfix memory leak",
        state="merged",
        author="bob",
        created_at=datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc),
        merged_at=datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc),
    )
    pr3 = PullRequest(
        repo_id=repo.id,
        pr_number=103,
        title="feat: summer release",
        state="merged",
        author="carol",
        created_at=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        merged_at=datetime(2026, 7, 3, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add(pr1)
    db_session.add(pr2)
    db_session.add(pr3)
    await db_session.commit()

    # Pass ISO string parameters for February window
    feb_dora = await compute_dora_metrics(
        db_session,
        repo.id,
        start_date="2026-02-01T00:00:00Z",
        end_date="2026-02-28T23:59:59Z",
    )
    assert feb_dora["change_failure_rate_value"] == 50.0  # 1 fix out of 2 PRs in Feb
    assert feb_dora["mttr_hours"] == 2.0  # 2 hours resolution for pr2


@pytest.mark.anyio
async def test_get_dora_metrics_endpoint_with_query_params(db_session: AsyncSessionAdapter):
    repo = Repo(
        url="https://github.com/org/endpoint-dora",
        name="org/endpoint-dora",
        owner="org",
        repo_slug="org-endpoint-dora",
        status="ready",
    )
    db_session.add(repo)
    await db_session.flush()

    dep = Deployment(
        repo_id=repo.id,
        provider="github",
        environment="production",
        status="success",
        deployed_at=datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add(dep)
    await db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Default (no query params)
        res = await client.get(f"/api/metrics/repos/{repo.id}/dora")
        assert res.status_code == 200
        data = res.json()
        assert "dora_score" in data
        assert "deployment_frequency_value" in data

        # With start_date and end_date matching
        res_filtered = await client.get(
            f"/api/metrics/repos/{repo.id}/dora?start_date=2026-04-01T00:00:00Z&end_date=2026-04-30T23:59:59Z"
        )
        assert res_filtered.status_code == 200
        data_filtered = res_filtered.json()
        assert data_filtered["deployment_frequency_value"] > 0

        # With window where no deployment happened
        res_empty = await client.get(
            f"/api/metrics/repos/{repo.id}/dora?start_date=2026-05-01T00:00:00Z&end_date=2026-05-31T23:59:59Z"
        )
        assert res_empty.status_code == 200
        data_empty = res_empty.json()
        assert data_empty["deployment_frequency_value"] == 0.0
        assert data_empty["dora_score"] == "Low"

        # 404 non-existent repo
        res_404 = await client.get("/api/metrics/repos/999999/dora")
        assert res_404.status_code == 404
