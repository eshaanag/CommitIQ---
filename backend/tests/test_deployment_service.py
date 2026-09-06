"""Tests for the deployment timeline computation service."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.features.reports.deployment_service import (
    _empty_response,
    get_deployment_timeline,
)


def _make_deployment(
    *,
    status: str = "success",
    environment: str = "production",
    provider: str = "gitlab",
    ref: str = "main",
    sha: str = "abc123def456",
    pipeline_id: str = "101",
    deployed_at: datetime | None = None,
) -> MagicMock:
    d = MagicMock()
    d.id = 1
    d.status = status
    d.environment = environment
    d.provider = provider
    d.ref = ref
    d.sha = sha
    d.pipeline_id = pipeline_id
    d.deployed_at = deployed_at or datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
    return d


class TestEmptyResponse:
    def test_structure(self):
        resp = _empty_response()
        assert resp["deployments"] == []
        assert resp["summary"]["total_deploys"] == 0
        assert resp["summary"]["success_rate"] == 0.0
        assert resp["daily"] == []


@pytest.mark.anyio
async def test_empty_deployments():
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert resp["summary"]["total_deploys"] == 0
    assert resp["deployments"] == []


@pytest.mark.anyio
async def test_repo_not_found():
    db = AsyncMock()
    db.get.return_value = None

    with pytest.raises(ValueError, match="not found"):
        await get_deployment_timeline(db, repo_id=999)


@pytest.mark.anyio
async def test_single_success():
    d = _make_deployment(status="success")
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [d]
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert resp["summary"]["total_deploys"] == 1
    assert resp["summary"]["success_count"] == 1
    assert resp["summary"]["success_rate"] == 100.0
    assert len(resp["deployments"]) == 1
    assert resp["deployments"][0]["status"] == "success"
    assert resp["deployments"][0]["env_color"] == "emerald"


@pytest.mark.anyio
async def test_mixed_status():
    deploys = [
        _make_deployment(status="success", environment="production"),
        _make_deployment(status="failed", environment="staging"),
        _make_deployment(status="success", environment="production"),
    ]
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = deploys
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert resp["summary"]["total_deploys"] == 3
    assert resp["summary"]["success_count"] == 2
    assert resp["summary"]["failure_count"] == 1
    assert resp["summary"]["success_rate"] == pytest.approx(66.7, abs=0.1)
    assert "production" in resp["summary"]["by_environment"]
    assert "staging" in resp["summary"]["by_environment"]


@pytest.mark.anyio
async def test_by_provider():
    deploys = [
        _make_deployment(provider="gitlab"),
        _make_deployment(provider="gitlab"),
        _make_deployment(provider="github-actions"),
    ]
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = deploys
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert resp["summary"]["by_provider"]["gitlab"] == 2
    assert resp["summary"]["by_provider"]["github-actions"] == 1


@pytest.mark.anyio
async def test_daily_counts():
    from datetime import timedelta

    base = datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc)
    deploys = [
        _make_deployment(status="success", deployed_at=base),
        _make_deployment(status="success", deployed_at=base + timedelta(hours=2)),
        _make_deployment(status="failed", deployed_at=base + timedelta(days=1)),
    ]
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = deploys
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert len(resp["daily"]) == 2
    assert resp["daily"][0]["success"] == 2
    assert resp["daily"][0]["failure"] == 0
    assert resp["daily"][1]["success"] == 0
    assert resp["daily"][1]["failure"] == 1


@pytest.mark.anyio
async def test_sha_truncation():
    d = _make_deployment(sha="a" * 40)
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [d]
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert len(resp["deployments"][0]["sha"]) == 12


@pytest.mark.anyio
async def test_env_color_mapping():
    d = _make_deployment(environment="preview")
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [d]
    db.execute.return_value = result_mock

    resp = await get_deployment_timeline(db, repo_id=1)
    assert resp["deployments"][0]["env_color"] == "violet"


@pytest.mark.anyio
async def test_limit_parameter():
    db = AsyncMock()
    repo = MagicMock()
    repo.id = 1
    db.get.return_value = repo
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock

    await get_deployment_timeline(db, repo_id=1, limit=10)
    # Verify execute was called (limit is applied via SQLAlchemy .limit())
    db.execute.assert_called_once()
