"""
backend/tests/test_pdf_report.py

Unit tests for the unified PDF report export (Issue #389).

Tests cover:
  - Router returns 200 + application/pdf for a valid repo.
  - Router returns 404 for a missing repo.
  - PDF service raises ValueError for missing repo.
  - PDF service generates valid PDF bytes.
  - PDF bytes start with the %PDF magic header.
  - PDF content includes key metric labels.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import Base, get_db
from backend.shared.models import Repo

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def sync_db():
    """Create an in-memory SQLite DB with synchronous sessions."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def _make_repo(db: Session, slug: str = "test/repo") -> Repo:
    repo = Repo(
        url=f"https://github.com/{slug}",
        name=slug.split("/")[-1],
        owner=slug.split("/")[0],
        repo_slug=slug,
        status="ready",
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    return repo


# ════════════════════════════════════════════════════════════════
# 1. PDF service tests
# ════════════════════════════════════════════════════════════════


class TestPdfService:
    @pytest.mark.asyncio
    async def test_generate_report_raises_for_missing_repo(self):
        """generate_health_report raises ValueError for a non-existent repo."""
        from backend.features.reports.pdf_service import generate_health_report

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Repository 999 not found"):
            await generate_health_report(mock_db, 999)

    @pytest.mark.asyncio
    async def test_generate_report_returns_valid_pdf(self):
        """generate_health_report returns bytes starting with %PDF."""
        from backend.features.reports.pdf_service import generate_health_report

        # NOTE: ``name`` cannot be passed to the MagicMock constructor
        # because it is reserved by MagicMock itself (used for repr).
        # It must be set as an attribute after construction.
        mock_repo = MagicMock(id=1, owner="test", github_language="Python")
        mock_repo.name = "repo"
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_repo)

        # Mock the metric computation functions.
        mock_dora = {
            "dora_score": "High",
            "deployment_frequency": "High",
            "deployment_frequency_value": 3.5,
            "change_failure_rate": "Low",
            "change_failure_rate_value": 5.0,
            "mttr_hours": 2.5,
            "mttr_category": "Elite",
        }
        mock_cycle = {
            "avg_cycle_time_hours": 24.5,
            "total_prs_analyzed": 15,
            "bottlenecks": [
                {
                    "pr_number": 42,
                    "title": "Refactor authentication",
                    "author": "alice",
                    "cycle_time_hours": 96.0,
                }
            ],
        }
        mock_health = {
            "burnout_risk_score": "Medium",
            "weekend_commits_percent": 8.5,
            "after_hours_commits_percent": 15.0,
            "context_switching_score": "Low",
            "avg_files_per_day": 12.3,
        }

        # Mock the latest snapshot query.
        mock_snapshot = MagicMock(health_score=78.5)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_snapshot
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "backend.features.reports.pdf_service.compute_dora_metrics",
                new_callable=AsyncMock,
                return_value=mock_dora,
            ),
            patch(
                "backend.features.reports.pdf_service.compute_cycle_time_metrics",
                new_callable=AsyncMock,
                return_value=mock_cycle,
            ),
            patch(
                "backend.features.reports.pdf_service.compute_team_health",
                new_callable=AsyncMock,
                return_value=mock_health,
            ),
        ):
            pdf_bytes, filename = await generate_health_report(mock_db, 1)

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"
        assert filename == "commitiq-health-report-test-repo.pdf"
        assert len(pdf_bytes) > 1000  # Should be a substantial PDF

    @pytest.mark.asyncio
    async def test_generate_report_includes_metric_labels(self):
        """PDF content includes key metric section labels."""
        from backend.features.reports.pdf_service import generate_health_report

        # ``name`` is reserved by MagicMock - set it after construction.
        mock_repo = MagicMock(id=1, owner="test", github_language=None)
        mock_repo.name = "repo"
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_repo)

        mock_dora = {
            "dora_score": "Elite",
            "deployment_frequency": "Elite",
            "deployment_frequency_value": 10.0,
            "change_failure_rate": "Low",
            "change_failure_rate_value": 2.0,
            "mttr_hours": 1.0,
            "mttr_category": "Elite",
        }
        mock_cycle = {
            "avg_cycle_time_hours": 12.0,
            "total_prs_analyzed": 5,
            "bottlenecks": [],
        }
        mock_health = {
            "burnout_risk_score": "Low",
            "weekend_commits_percent": 2.0,
            "after_hours_commits_percent": 5.0,
            "context_switching_score": "Low",
            "avg_files_per_day": 8.0,
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # no snapshot
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "backend.features.reports.pdf_service.compute_dora_metrics",
                new_callable=AsyncMock,
                return_value=mock_dora,
            ),
            patch(
                "backend.features.reports.pdf_service.compute_cycle_time_metrics",
                new_callable=AsyncMock,
                return_value=mock_cycle,
            ),
            patch(
                "backend.features.reports.pdf_service.compute_team_health",
                new_callable=AsyncMock,
                return_value=mock_health,
            ),
        ):
            pdf_bytes, _ = await generate_health_report(mock_db, 1)

        # The PDF is binary, but section headers are stored as text
        # strings inside the PDF content stream.
        pdf_text = pdf_bytes.decode("latin-1")
        assert "Developer Health Report" in pdf_text
        assert "DORA Metrics" in pdf_text
        assert "Cycle Time Analysis" in pdf_text
        assert "Team Health" in pdf_text
        assert "CommitIQ" in pdf_text


# ════════════════════════════════════════════════════════════════
# 2. Router tests
# ════════════════════════════════════════════════════════════════


class TestReportRouter:
    @pytest.mark.asyncio
    async def test_router_returns_404_for_missing_repo(self):
        """GET /api/repos/{repo_id}/report returns 404 for missing repo."""
        from backend.main import app

        # FastAPI captures the ``get_db`` callable inside ``Depends(...)``
        # at route-definition time, so patching ``backend.database.get_db``
        # has no effect on the dependency the router actually invokes.
        # The supported way to swap a dependency in tests is via
        # ``app.dependency_overrides``.
        async def override_get_db():
            mock_db = AsyncMock()
            mock_db.get = AsyncMock(return_value=None)
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db
        try:
            client = TestClient(app)
            response = client.get("/api/repos/999/report")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 404
        assert "Repository not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_router_returns_pdf_for_valid_repo(self):
        """GET /api/repos/{repo_id}/report returns 200 + application/pdf."""
        from backend.main import app

        # ``name`` is reserved by MagicMock - set it after construction.
        mock_repo = MagicMock(id=1, owner="test", github_language="Python")
        mock_repo.name = "repo"
        mock_dora = {
            "dora_score": "High",
            "deployment_frequency": "High",
            "deployment_frequency_value": 3.0,
            "change_failure_rate": "Low",
            "change_failure_rate_value": 5.0,
            "mttr_hours": 2.0,
            "mttr_category": "Elite",
        }
        mock_cycle = {
            "avg_cycle_time_hours": 20.0,
            "total_prs_analyzed": 10,
            "bottlenecks": [],
        }
        mock_health = {
            "burnout_risk_score": "Low",
            "weekend_commits_percent": 3.0,
            "after_hours_commits_percent": 8.0,
            "context_switching_score": "Low",
            "avg_files_per_day": 10.0,
        }

        # Use the supported FastAPI dependency-override mechanism
        # instead of patching ``backend.database.get_db`` (which is
        # captured by ``Depends(...)`` at route-definition time).
        async def override_get_db():
            mock_db = AsyncMock()
            mock_db.get = AsyncMock(return_value=mock_repo)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_db.execute = AsyncMock(return_value=mock_result)
            yield mock_db

        app.dependency_overrides[get_db] = override_get_db
        try:
            with (
                patch(
                    "backend.features.reports.pdf_service.compute_dora_metrics",
                    new_callable=AsyncMock,
                    return_value=mock_dora,
                ),
                patch(
                    "backend.features.reports.pdf_service.compute_cycle_time_metrics",
                    new_callable=AsyncMock,
                    return_value=mock_cycle,
                ),
                patch(
                    "backend.features.reports.pdf_service.compute_team_health",
                    new_callable=AsyncMock,
                    return_value=mock_health,
                ),
            ):
                client = TestClient(app)
                response = client.get("/api/repos/1/report")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "attachment" in response.headers.get("content-disposition", "")
        assert response.content[:4] == b"%PDF"
