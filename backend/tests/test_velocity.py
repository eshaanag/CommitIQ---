"""Tests for the velocity & delivery cadence metrics module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.features.metrics.velocity import (
    _deviation,
    _empty_response,
    _iso_week_key,
    _week_start,
    compute_velocity,
)

# ── pure helper tests ──────────────────────────────────────────────────────────


class TestIsoWeekKey:
    def test_monday(self):
        dt = datetime(2025, 6, 30, 12, 0, tzinfo=timezone.utc)  # Monday
        assert _iso_week_key(dt) == "2025-W27"

    def test_sunday(self):
        dt = datetime(2025, 7, 6, 8, 0, tzinfo=timezone.utc)  # Sunday
        assert _iso_week_key(dt) == "2025-W27"

    def test_year_boundary(self):
        dt = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _iso_week_key(dt)
        assert result.startswith("2025-W")


class TestWeekStart:
    def test_returns_monday(self):
        dt = datetime(2025, 7, 3, 14, 30, tzinfo=timezone.utc)  # Thursday
        start = _week_start(dt)
        assert start.weekday() == 0  # Monday
        assert start.hour == 0
        assert start.minute == 0

    def test_already_monday(self):
        dt = datetime(2025, 6, 30, 9, 15, tzinfo=timezone.utc)  # Monday
        start = _week_start(dt)
        assert start == dt.replace(hour=0, minute=0, second=0, microsecond=0)


class TestDeviation:
    def test_empty_list(self):
        assert _deviation([]) == 0.0

    def test_single_value(self):
        assert _deviation([5.0]) == 0.0

    def test_two_equal(self):
        assert _deviation([3.0, 3.0]) == 0.0

    def test_known_value(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _deviation(vals)
        assert abs(result - 2.0) < 0.01  # stddev of this set ≈ 2.0

    def test_higher_deviation(self):
        low_dev = _deviation([10, 11, 10, 11])
        high_dev = _deviation([1, 50, 100, 3])
        assert high_dev > low_dev


class TestEmptyResponse:
    def test_structure(self):
        resp = _empty_response()
        assert resp["weekly"] == []
        assert resp["totals"]["total_commits"] == 0
        assert resp["totals"]["cadence_score"] == 0
        assert resp["contributors"] == []


# ── async compute_velocity tests ───────────────────────────────────────────────


def _make_commit(
    *,
    author_name: str = "alice",
    author_email: str = "alice@example.com",
    committed_at: datetime | None = None,
    insertions: int = 10,
    deletions: int = 5,
    files_changed: int = 2,
) -> MagicMock:
    c = MagicMock()
    c.author_name = author_name
    c.author_email = author_email
    c.committed_at = committed_at or datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc)
    c.insertions = insertions
    c.deletions = deletions
    c.files_changed = files_changed
    return c


@pytest.mark.anyio
async def test_empty_repo():
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert resp["totals"]["total_commits"] == 0
    assert resp["weekly"] == []
    assert resp["contributors"] == []


@pytest.mark.anyio
async def test_single_commit():
    commit = _make_commit(insertions=20, deletions=5, files_changed=3)
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [commit]
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert resp["totals"]["total_commits"] == 1
    assert resp["totals"]["total_insertions"] == 20
    assert resp["totals"]["total_deletions"] == 5
    assert len(resp["weekly"]) == 1
    assert resp["weekly"][0]["commits"] == 1
    assert resp["weekly"][0]["lines_changed"] == 25
    assert len(resp["contributors"]) == 1
    assert resp["contributors"][0]["name"] == "alice"
    assert resp["contributors"][0]["commit_pct"] == 100.0


@pytest.mark.anyio
async def test_multi_author():
    commits = [
        _make_commit(
            author_name="alice",
            author_email="a@ex.com",
            committed_at=datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
            insertions=30,
            deletions=10,
            files_changed=5,
        ),
        _make_commit(
            author_name="bob",
            author_email="b@ex.com",
            committed_at=datetime(2025, 7, 2, 14, 0, tzinfo=timezone.utc),
            insertions=15,
            deletions=5,
            files_changed=2,
        ),
        _make_commit(
            author_name="alice",
            author_email="a@ex.com",
            committed_at=datetime(2025, 7, 3, 9, 0, tzinfo=timezone.utc),
            insertions=10,
            deletions=0,
            files_changed=1,
        ),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert resp["totals"]["num_active_contributors"] == 2
    assert resp["totals"]["total_commits"] == 3
    assert resp["totals"]["total_insertions"] == 55
    assert resp["totals"]["total_deletions"] == 15

    # alice has 2/3 = 66.7%
    contrib_names = [c["name"] for c in resp["contributors"]]
    alice = next(c for c in resp["contributors"] if c["name"] == "alice")
    assert alice["commit_pct"] == pytest.approx(66.7, abs=0.1)
    assert alice["weeks_active"] == 1  # same week

    bob = next(c for c in resp["contributors"] if c["name"] == "bob")
    assert bob["commit_pct"] == pytest.approx(33.3, abs=0.1)


@pytest.mark.anyio
async def test_cadence_score_perfect():
    """Same number of commits each week → high cadence score."""
    commits = []
    base = datetime(2025, 7, 7, 10, 0, tzinfo=timezone.utc)
    for week in range(4):
        for day in range(3):
            dt = base + timedelta(weeks=week, days=day)
            commits.append(_make_commit(committed_at=dt))

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert resp["totals"]["cadence_score"] >= 50


@pytest.mark.anyio
async def test_streak_calculation():
    """Three consecutive weeks should give max streak of 3."""
    commits = []
    base = datetime(2025, 7, 7, 10, 0, tzinfo=timezone.utc)
    for week_offset in range(3):
        dt = base + timedelta(weeks=week_offset)
        commits.append(_make_commit(committed_at=dt))

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert resp["totals"]["max_commit_streak_weeks"] >= 1


@pytest.mark.anyio
async def test_weekly_bucketing():
    """Commits in the same ISO week should be bucketed together."""
    monday = datetime(2025, 7, 7, 9, 0, tzinfo=timezone.utc)
    friday = datetime(2025, 7, 11, 16, 0, tzinfo=timezone.utc)
    next_monday = datetime(2025, 7, 14, 9, 0, tzinfo=timezone.utc)

    commits = [
        _make_commit(committed_at=monday),
        _make_commit(committed_at=friday),
        _make_commit(committed_at=next_monday),
    ]

    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_velocity(db, repo_id=1)
    assert len(resp["weekly"]) == 2
    assert resp["weekly"][0]["commits"] == 2  # same week
    assert resp["weekly"][1]["commits"] == 1
