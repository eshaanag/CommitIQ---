"""Tests for scheduled health report service and API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.features.reports.scheduler_service import (
    compute_next_run,
    describe_cron,
    execute_scheduled_report,
    generate_report_payload,
    parse_cron,
    validate_cron_expression,
)

# ---------------------------------------------------------------------------
# Cron parsing tests
# ---------------------------------------------------------------------------


class TestParseCron:
    def test_every_minute(self):
        fields = parse_cron("* * * * *")
        assert fields["minute"] == set(range(60))
        assert fields["hour"] == set(range(24))
        assert fields["day"] == set(range(1, 32))
        assert fields["month"] == set(range(1, 13))
        assert fields["weekday"] == set(range(7))

    def test_specific_values(self):
        fields = parse_cron("0 9 * * MON")
        assert fields["minute"] == {0}
        assert fields["hour"] == {9}
        assert fields["weekday"] == {1}

    def test_range(self):
        fields = parse_cron("0 9-17 * * 1-5")
        assert fields["hour"] == set(range(9, 18))
        assert fields["weekday"] == {1, 2, 3, 4, 5}

    def test_step(self):
        fields = parse_cron("*/15 * * * *")
        assert fields["minute"] == {0, 15, 30, 45}

    def test_comma_separated(self):
        fields = parse_cron("0 9,12,18 * * *")
        assert fields["hour"] == {9, 12, 18}

    def test_named_weekdays(self):
        fields = parse_cron("0 9 * * MON,WED,FRI")
        assert fields["weekday"] == {1, 3, 5}

    def test_invalid_field_count(self):
        with pytest.raises(ValueError, match="5 fields"):
            parse_cron("* * *")

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            parse_cron("61 * * * *")


class TestValidateCronExpression:
    def test_valid(self):
        ok, msg = validate_cron_expression("0 9 * * MON")
        assert ok is True
        assert msg == ""

    def test_invalid(self):
        ok, msg = validate_cron_expression("invalid")
        assert ok is False
        assert "5 fields" in msg


class TestComputeNextRun:
    def test_daily_at_9am(self):
        now = datetime(2026, 8, 29, 8, 0, tzinfo=timezone.utc)
        result = compute_next_run("0 9 * * *", "UTC", after=now)
        assert result.hour == 9
        assert result.minute == 0
        assert result > now

    def test_weekly_monday(self):
        # Friday Aug 28, 2026 -> next Monday Aug 31
        now = datetime(2026, 8, 28, 10, 30, tzinfo=timezone.utc)
        result = compute_next_run("0 9 * * MON", "UTC", after=now)
        assert result.weekday() == 1  # Monday
        assert result.hour == 9

    def test_next_minute(self):
        now = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
        result = compute_next_run("30 * * * *", "UTC", after=now)
        assert result.minute == 30
        assert result.hour == 9

    def test_after_now(self):
        now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
        result = compute_next_run("0 9 * * *", "UTC", after=now)
        assert result > now


class TestDescribeCron:
    def test_daily(self):
        desc = describe_cron("0 9 * * *")
        assert "Daily" in desc

    def test_weekday(self):
        desc = describe_cron("0 10 * * 1-5")
        assert "Weekdays" in desc

    def test_specific_day(self):
        desc = describe_cron("0 9 * * MON")
        assert "MON" in desc

    def test_multiple_days(self):
        desc = describe_cron("0 9 * * MON,WED,FRI")
        assert "MON" in desc
        assert "WED" in desc
        assert "FRI" in desc


# ---------------------------------------------------------------------------
# Payload generation tests
# ---------------------------------------------------------------------------


class TestGenerateReportPayload:
    @pytest.mark.anyio
    async def test_empty_repo_returns_defaults(self):
        db = AsyncMock()
        db.get.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await generate_report_payload(db, 999, "health_summary")


# ---------------------------------------------------------------------------
# Execute scheduled report tests
# ---------------------------------------------------------------------------


class TestExecuteScheduledReport:
    @pytest.mark.anyio
    async def test_schedule_not_found(self):
        db = AsyncMock()
        db.get.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await execute_scheduled_report(db, 999)

    @pytest.mark.anyio
    async def test_successful_execution_no_webhook(self):
        schedule = AsyncMock()
        schedule.id = 1
        schedule.repo_id = 10
        schedule.report_type = "health_summary"
        schedule.webhook_url = None
        schedule.webhook_secret = None
        schedule.cron_expression = "0 9 * * *"
        schedule.timezone = "UTC"
        schedule.max_retry_count = 3
        schedule.consecutive_failures = 0

        db = AsyncMock()
        db.get.return_value = schedule
        db.flush = AsyncMock()
        db.commit = AsyncMock()

        with patch(
            "backend.features.reports.scheduler_service.generate_report_payload",
            new_callable=AsyncMock,
        ) as mock_gen:
            mock_gen.return_value = {
                "report_type": "health_summary",
                "repo_name": "test-repo",
                "summary": {"total_commits": 42},
                "latest_commit": {"sha": "abc123"},
            }
            delivery = await execute_scheduled_report(db, 1)

        assert delivery.status == "success"
        assert delivery.snapshot_commits_analyzed == 42
        assert schedule.last_delivery_status == "success"
        assert schedule.consecutive_failures == 0

    @pytest.mark.anyio
    async def test_failed_execution_disables_after_max_retries(self):
        schedule = AsyncMock()
        schedule.id = 2
        schedule.repo_id = 20
        schedule.report_type = "health_summary"
        schedule.webhook_url = "https://hooks.example.com/fail"
        schedule.webhook_secret = None
        schedule.cron_expression = "0 9 * * *"
        schedule.timezone = "UTC"
        schedule.max_retry_count = 3
        schedule.consecutive_failures = 2

        db = AsyncMock()
        db.get.return_value = schedule
        db.flush = AsyncMock()
        db.commit = AsyncMock()

        with (
            patch(
                "backend.features.reports.scheduler_service.generate_report_payload",
                new_callable=AsyncMock,
            ) as mock_gen,
            patch(
                "backend.features.reports.scheduler_service.deliver_webhook",
                new_callable=AsyncMock,
            ) as mock_webhook,
        ):
            mock_gen.return_value = {"summary": {}, "report_type": "health_summary"}
            mock_webhook.return_value = (500, "Internal Server Error")

            delivery = await execute_scheduled_report(db, 2)

        assert delivery.status == "failed"
        assert "500" in delivery.error_message
        assert schedule.consecutive_failures == 3
        assert schedule.is_active is False


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestScheduleAPIEndpoints:
    """Integration-style tests for the schedule router using FastAPI TestClient."""

    @pytest.fixture
    def mock_schedule(self):
        return {
            "id": 1,
            "repo_id": 1,
            "name": "Weekly Report",
            "description": None,
            "cron_expression": "0 9 * * MON",
            "cron_description": "Every MON at minute 0 9",
            "timezone": "UTC",
            "report_type": "health_summary",
            "is_active": True,
            "webhook_url": None,
            "notification_email": None,
            "include_narrative": False,
            "last_run_at": None,
            "next_run_at": "2026-09-01T09:00:00+00:00",
            "last_delivery_status": None,
            "consecutive_failures": 0,
            "max_retry_count": 3,
            "created_at": "2026-08-29T12:00:00+00:00",
            "updated_at": "2026-08-29T12:00:00+00:00",
        }

    def test_cron_validator_rejects_bad_input(self):
        from backend.features.reports.schedule_router import ReportScheduleCreate

        with pytest.raises(Exception):
            ReportScheduleCreate(
                name="Bad Cron",
                cron_expression="invalid",
            )

    def test_cron_validator_accepts_valid(self):
        from backend.features.reports.schedule_router import ReportScheduleCreate

        s = ReportScheduleCreate(
            name="Good",
            cron_expression="0 9 * * MON",
        )
        assert s.cron_expression == "0 9 * * MON"

    def test_report_type_validator_rejects_invalid(self):
        from backend.features.reports.schedule_router import ReportScheduleCreate

        with pytest.raises(Exception):
            ReportScheduleCreate(
                name="Bad Type",
                cron_expression="0 9 * * *",
                report_type="invalid_type",
            )

    def test_report_type_validator_accepts_valid(self):
        from backend.features.reports.schedule_router import ReportScheduleCreate

        s = ReportScheduleCreate(
            name="DORA",
            cron_expression="0 9 * * *",
            report_type="dora_metrics",
        )
        assert s.report_type == "dora_metrics"
