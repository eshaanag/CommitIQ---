"""Tests for the weekly health digest service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.features.metrics.weekly_digest import compute_weekly_digest


def _mock_exec(results: list):
    """Build db.execute.side_effect from a list of result lists."""
    side = []
    for r in results:
        m = MagicMock()
        m.scalars.return_value.all.return_value = r
        side.append(m)
    return side


class TestWeeklyDigest:
    @pytest.mark.anyio
    async def test_repo_not_found(self):
        db = AsyncMock()
        db.get.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await compute_weekly_digest(db, 999)

    @pytest.mark.anyio
    async def test_empty_repo(self):
        db = AsyncMock()
        db.get.return_value = MagicMock(name="r", repo_slug="r")
        db.execute.side_effect = _mock_exec([[], [], []])
        d = await compute_weekly_digest(db, 1)
        assert d["summary"]["total_commits"] == 0
        assert d["alerts"] == []

    @pytest.mark.anyio
    async def test_aggregates_commits(self):
        now = datetime.now(timezone.utc)
        commits = [
            MagicMock(
                insertions=100,
                deletions=20,
                files_changed=3,
                author_email="a@x.com",
                committed_at=now,
            ),
            MagicMock(
                insertions=50,
                deletions=10,
                files_changed=2,
                author_email="b@x.com",
                committed_at=now,
            ),
        ]
        db = AsyncMock()
        db.get.return_value = MagicMock(name="r", repo_slug="r")
        db.execute.side_effect = _mock_exec([commits, [], []])
        d = await compute_weekly_digest(db, 1)
        assert d["summary"]["total_insertions"] == 150
        assert d["summary"]["unique_contributors"] == 2

    @pytest.mark.anyio
    async def test_regression_alerts(self):
        now = datetime.now(timezone.utc)
        prev_time = now - timedelta(days=10)
        cur = [MagicMock(health_score=40.0, avg_complexity=12.0, churn_rate=0.3, computed_at=now)]
        prev = [
            MagicMock(health_score=80.0, avg_complexity=5.0, churn_rate=0.05, computed_at=prev_time)
        ]
        db = AsyncMock()
        db.get.return_value = MagicMock(name="r", repo_slug="r")
        db.execute.side_effect = _mock_exec([[], cur + prev, []])
        d = await compute_weekly_digest(db, 1)
        metrics = {a["metric"] for a in d["alerts"]}
        assert "health_score" in metrics
        assert "complexity" in metrics


class TestEndpointValidation:
    def test_weeks_bounds(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend.features.metrics.digest_router import router

        app = FastAPI()
        app.include_router(router, prefix="/api")
        c = TestClient(app)
        assert c.get("/api/repos/1/digest?weeks=0").status_code == 422
        assert c.get("/api/repos/1/digest?weeks=999").status_code == 422
