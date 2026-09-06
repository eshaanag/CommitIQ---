"""Tests for the commit message quality linter module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.features.metrics.commit_linter import (
    LintViolation,
    _empty_response,
    compute_commit_quality,
    lint_message,
)

# ── lint_message tests ─────────────────────────────────────────────────────────


class TestLintMessage:
    def test_empty_message(self):
        violations = lint_message(None)
        assert len(violations) == 1
        assert violations[0].rule == "empty_message"
        assert violations[0].severity == "error"

    def test_empty_string(self):
        violations = lint_message("")
        assert len(violations) == 1
        assert violations[0].rule == "empty_message"

    def test_conventional_feat(self):
        violations = lint_message("feat(auth): add login page")
        # should have no errors or non_conventional warning
        rules = [v.rule for v in violations]
        assert "non_conventional" not in rules

    def test_conventional_fix(self):
        violations = lint_message("fix: resolve null pointer in parser")
        rules = [v.rule for v in violations]
        assert "non_conventional" not in rules

    def test_conventional_with_bang(self):
        violations = lint_message("feat(api)!: break backward compatibility")
        rules = [v.rule for v in violations]
        assert "non_conventional" not in rules

    def test_non_conventional(self):
        violations = lint_message("updated stuff")
        rules = [v.rule for v in violations]
        assert "non_conventional" in rules

    def test_subject_too_long(self):
        msg = "feat: " + "a" * 70  # 76 chars total
        violations = lint_message(msg)
        rules = [v.rule for v in violations]
        assert "subject_too_long" in rules

    def test_subject_too_short(self):
        violations = lint_message("fix: ab")
        rules = [v.rule for v in violations]
        assert "subject_too_short" not in rules  # "fix: ab" is 7 chars

    def test_subject_very_short(self):
        violations = lint_message("fix")
        rules = [v.rule for v in violations]
        assert "subject_too_short" in rules

    def test_trailing_period(self):
        violations = lint_message("feat: add feature.")
        rules = [v.rule for v in violations]
        assert "trailing_period" in rules

    def test_all_caps(self):
        violations = lint_message("FIX: RESOLVE BUG")
        rules = [v.rule for v in violations]
        assert "all_caps" in rules

    def test_missing_body(self):
        violations = lint_message("feat: add new feature")
        rules = [v.rule for v in violations]
        assert "missing_body" in rules

    def test_with_body(self):
        msg = "feat: add new feature\n\nThis adds a new feature for users."
        violations = lint_message(msg)
        rules = [v.rule for v in violations]
        assert "missing_body" not in rules

    def test_wip_stale_marker(self):
        violations = lint_message("feat: WIP on auth module")
        rules = [v.rule for v in violations]
        assert "stale_message" in rules

    def test_merge_commit_ignored(self):
        violations = lint_message("Merge branch 'main' into feature")
        rules = [v.rule for v in violations]
        assert "non_conventional" not in rules

    def test_body_line_too_long(self):
        msg = "feat: add feature\n\n" + "x" * 110
        violations = lint_message(msg)
        rules = [v.rule for v in violations]
        assert "body_line_long" in rules


class TestLintViolation:
    def test_to_dict(self):
        v = LintViolation("test_rule", "warning", "Test message")
        d = v.to_dict()
        assert d == {
            "rule": "test_rule",
            "severity": "warning",
            "message": "Test message",
        }


# ── empty response test ────────────────────────────────────────────────────────


class TestEmptyResponse:
    def test_structure(self):
        resp = _empty_response()
        assert resp["quality_score"] == 0
        assert resp["total_commits"] == 0
        assert resp["conventional_commits"] == 0
        assert resp["contributors"] == []
        assert resp["top_violations"] == []


# ── compute_commit_quality async tests ─────────────────────────────────────────


def _make_commit(
    *,
    message: str = "feat: add feature",
    author_name: str = "alice",
    author_email: str = "alice@example.com",
    committed_at: datetime | None = None,
) -> MagicMock:
    c = MagicMock()
    c.message = message
    c.author_name = author_name
    c.author_email = author_email
    c.committed_at = committed_at or datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc)
    return c


@pytest.mark.anyio
async def test_empty_repo():
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    assert resp["quality_score"] == 0
    assert resp["total_commits"] == 0


@pytest.mark.anyio
async def test_all_conventional():
    commits = [
        _make_commit(message="feat(auth): add login"),
        _make_commit(message="fix(parser): resolve crash"),
        _make_commit(message="docs: update README"),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    assert resp["total_commits"] == 3
    assert resp["conventional_commits"] == 3
    assert resp["convention_rate"] == 100.0
    assert resp["quality_score"] >= 80


@pytest.mark.anyio
async def test_mixed_quality():
    commits = [
        _make_commit(message="feat(auth): add login"),
        _make_commit(message="updated stuff"),
        _make_commit(message="fix: resolve bug"),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    assert resp["conventional_commits"] == 2
    assert resp["convention_rate"] == pytest.approx(66.7, abs=0.1)


@pytest.mark.anyio
async def test_contributor_breakdown():
    commits = [
        _make_commit(message="feat(a): add a", author_name="alice"),
        _make_commit(message="feat(b): add b", author_name="alice"),
        _make_commit(message="updated stuff", author_name="bob"),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    contribs = resp["contributors"]
    assert len(contribs) == 2
    alice = next(c for c in contribs if c["name"] == "alice")
    assert alice["convention_rate"] == 100.0
    assert alice["total"] == 2


@pytest.mark.anyio
async def test_violations_counted():
    commits = [
        _make_commit(message=""),
        _make_commit(message="WIP: temp fix"),
        _make_commit(message="feat: good commit"),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    assert resp["total_violations"] > 0
    assert resp["severity_breakdown"]["error"] >= 1  # empty message


@pytest.mark.anyio
async def test_merge_commits_excluded_from_convention_rate():
    commits = [
        _make_commit(message="feat: add feature"),
        _make_commit(message="Merge branch 'main' into feature"),
        _make_commit(message="fix: resolve issue"),
    ]
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = commits
    db.execute.return_value = result_mock

    resp = await compute_commit_quality(db, repo_id=1)
    assert resp["merge_commits"] == 1
    # convention_rate should be 100% (2 conventional / 2 non-merge)
    assert resp["convention_rate"] == 100.0
