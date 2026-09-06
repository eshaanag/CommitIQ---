"""Commit message quality linter and analytics.

Analyses the commit message corpus of a repository to compute quality
scores, detect convention compliance (Conventional Commits / Angular),
flag common anti-patterns, and break down stats per contributor.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.models import Commit

logger = logging.getLogger(__name__)

# ── Convention patterns ────────────────────────────────────────────────────────

# Conventional Commits: type(scope): description   or   type: description
_CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|hotfix|release)"
    r"(?:\([a-zA-Z0-9_./-]+\))?"
    r"!?:\s+\S.+",
    re.IGNORECASE,
)

# Subject-line heuristic (first line of the message)
_MAX_SUBJECT_LENGTH = 72
_MIN_SUBJECT_LENGTH = 5
_MAX_BODY_LINE_LENGTH = 100

# Anti-pattern keywords / markers
_STALE_PATTERNS = [
    re.compile(r"\bwip\b", re.IGNORECASE),
    re.compile(r"\btmp\b", re.IGNORECASE),
    re.compile(r"\btemp\b", re.IGNORECASE),
    re.compile(r"\bwip commit\b", re.IGNORECASE),
    re.compile(r"\bfixup\b", re.IGNORECASE),
    re.compile(r"\bsquash\b", re.IGNORECASE),
    re.compile(r"\baddress review\b", re.IGNORECASE),
    re.compile(r"\bchanges requested\b", re.IGNORECASE),
    re.compile(r"\bminor\b", re.IGNORECASE),
    re.compile(r"\btypo\b", re.IGNORECASE),
    re.compile(r"\bcleanup\b", re.IGNORECASE),
]

_MERGE_MARKERS = re.compile(r"^Merge (branch|pull request|remote|tag)", re.IGNORECASE)


# ── Per-message analysis ──────────────────────────────────────────────────────


class LintViolation:
    """Lightweight lint result for a single commit message."""

    __slots__ = ("rule", "severity", "message")

    def __init__(self, rule: str, severity: str, message: str) -> None:
        self.rule = rule
        self.severity = severity  # 'error' | 'warning' | 'info'
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "severity": self.severity, "message": self.message}


def lint_message(message: str | None) -> list[LintViolation]:
    """Return lint violations for a single commit message."""
    if not message:
        return [LintViolation("empty_message", "error", "Commit message is empty")]

    violations: list[LintViolation] = []
    lines = message.strip().splitlines()
    subject = lines[0].strip()

    # ── subject checks ──────────────────────────────────────────────
    if len(subject) > _MAX_SUBJECT_LENGTH:
        violations.append(
            LintViolation(
                "subject_too_long",
                "warning",
                f"Subject line is {len(subject)} chars (max {_MAX_SUBJECT_LENGTH})",
            )
        )

    if len(subject) < _MIN_SUBJECT_LENGTH:
        violations.append(
            LintViolation(
                "subject_too_short",
                "warning",
                f"Subject line is {len(subject)} chars (min {_MIN_SUBJECT_LENGTH})",
            )
        )

    if subject.startswith("."):
        violations.append(LintViolation("hidden_file_prefix", "info", "Subject starts with a dot"))

    if subject.endswith("."):
        violations.append(LintViolation("trailing_period", "info", "Subject ends with a period"))

    if subject.isupper():
        violations.append(LintViolation("all_caps", "info", "Subject is entirely uppercase"))

    if subject.startswith("Merge ") or subject.startswith("Revert "):
        pass  # merge / revert are legitimate conventional patterns
    elif not _CONVENTIONAL_RE.match(subject):
        violations.append(
            LintViolation(
                "non_conventional",
                "warning",
                "Subject does not follow Conventional Commits format",
            )
        )

    # ── body checks ─────────────────────────────────────────────────
    if len(lines) == 1:
        violations.append(
            LintViolation(
                "missing_body",
                "info",
                "No body provided (consider adding context)",
            )
        )

    for i, line in enumerate(lines[2:], start=3):
        if len(line) > _MAX_BODY_LINE_LENGTH:
            violations.append(
                LintViolation(
                    "body_line_long",
                    "info",
                    f"Body line {i} is {len(line)} chars (recommended ≤ {_MAX_BODY_LINE_LENGTH})",
                )
            )
            break  # report only first long line

    # ── anti-patterns ───────────────────────────────────────────────
    for pat in _STALE_PATTERNS:
        if pat.search(subject):
            violations.append(
                LintViolation(
                    "stale_message",
                    "warning",
                    f'Subject contains stale marker: "{pat.pattern}"',
                )
            )
            break

    return violations


# ── Aggregate computation ──────────────────────────────────────────────────────


async def compute_commit_quality(db: AsyncSession, repo_id: int) -> dict[str, Any]:
    """Return commit message quality analytics for *repo_id*."""
    stmt = select(Commit).where(Commit.repo_id == repo_id).order_by(Commit.committed_at.desc())
    commits = (await db.execute(stmt)).scalars().all()

    if not commits:
        return _empty_response()

    total = len(commits)
    merge_commits = 0
    conventional_count = 0
    total_violations = 0
    violations_by_rule: dict[str, int] = defaultdict(int)
    severity_counts: dict[str, int] = defaultdict(int)

    author_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"name": "", "total": 0, "conventional": 0, "errors": 0, "warnings": 0, "infos": 0}
    )

    subject_lengths: list[int] = []

    for c in commits:
        msg = c.message or ""
        lines = msg.strip().splitlines()
        subject = lines[0].strip() if lines else ""
        subject_lengths.append(len(subject))

        is_merge = bool(_MERGE_MARKERS.match(subject))
        if is_merge:
            merge_commits += 1

        violations = lint_message(msg)
        is_conventional = bool(_CONVENTIONAL_RE.match(subject))

        if is_conventional:
            conventional_count += 1

        author = c.author_name or c.author_email or "unknown"
        at = author_stats[author]
        at["name"] = author
        at["total"] += 1
        if is_conventional:
            at["conventional"] += 1

        for v in violations:
            total_violations += 1
            violations_by_rule[v.rule] += 1
            severity_counts[v.severity] += 1
            if v.severity == "error":
                at["errors"] += 1
            elif v.severity == "warning":
                at["warnings"] += 1
            else:
                at["infos"] += 1

    non_merge = max(total - merge_commits, 1)
    avg_subject_len = (
        round(sum(subject_lengths) / len(subject_lengths), 1) if subject_lengths else 0
    )
    median_subject_len = (
        sorted(subject_lengths)[len(subject_lengths) // 2] if subject_lengths else 0
    )

    # quality score: 100 = perfect, each issue deducts points
    convention_rate = conventional_count / non_merge
    error_rate = severity_counts.get("error", 0) / total
    warning_rate = severity_counts.get("warning", 0) / total
    quality_score = max(
        0,
        min(
            100,
            round(convention_rate * 60 + (1 - error_rate) * 20 + (1 - warning_rate) * 20),
        ),
    )

    # top violations
    sorted_violations = sorted(violations_by_rule.items(), key=lambda x: -x[1])

    # contributor leaderboard
    contrib_list = sorted(
        [
            {
                "name": at["name"],
                "total": at["total"],
                "conventional": at["conventional"],
                "convention_rate": round(at["conventional"] / max(at["total"], 1) * 100, 1),
                "errors": at["errors"],
                "warnings": at["warnings"],
                "infos": at["infos"],
            }
            for at in author_stats.values()
        ],
        key=lambda x: -x["total"],
    )

    return {
        "quality_score": quality_score,
        "total_commits": total,
        "conventional_commits": conventional_count,
        "convention_rate": round(convention_rate * 100, 1),
        "merge_commits": merge_commits,
        "avg_subject_length": avg_subject_len,
        "median_subject_length": median_subject_len,
        "total_violations": total_violations,
        "severity_breakdown": {
            "error": severity_counts.get("error", 0),
            "warning": severity_counts.get("warning", 0),
            "info": severity_counts.get("info", 0),
        },
        "top_violations": [
            {"rule": rule, "count": count} for rule, count in sorted_violations[:10]
        ],
        "contributors": contrib_list[:20],
    }


def _empty_response() -> dict[str, Any]:
    return {
        "quality_score": 0,
        "total_commits": 0,
        "conventional_commits": 0,
        "convention_rate": 0.0,
        "merge_commits": 0,
        "avg_subject_length": 0,
        "median_subject_length": 0,
        "total_violations": 0,
        "severity_breakdown": {"error": 0, "warning": 0, "info": 0},
        "top_violations": [],
        "contributors": [],
    }
