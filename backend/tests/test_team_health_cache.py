from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.features.metrics.team_health import compute_team_health

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend():
    return "asyncio"


async def test_compute_team_health_uses_cache():
    mock_db = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.get.return_value = '{"burnout_risk_score": "High", "weekend_commits_percent": 10.0, "after_hours_commits_percent": 15.0, "context_switching_score": "Low", "avg_files_per_day": 5.0}'

    with patch("backend.features.metrics.team_health._get_redis", return_value=mock_redis):
        res = await compute_team_health(mock_db, 1)

        assert res["burnout_risk_score"] == "High"
        assert res["weekend_commits_percent"] == 10.0
        mock_redis.get.assert_called_once_with("team_health:1")
        mock_db.execute.assert_not_called()


async def test_compute_team_health_writes_to_cache():
    mock_db = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    mock_commit = MagicMock()
    mock_commit.committed_at = None

    mock_result = MagicMock()
    mock_result.scalars().all.return_value = [mock_commit]
    mock_db.execute.return_value = mock_result

    with patch("backend.features.metrics.team_health._get_redis", return_value=mock_redis):
        res = await compute_team_health(mock_db, 2)

        mock_redis.get.assert_called_once_with("team_health:2")
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "team_health:2"
        assert kwargs["ex"] == 3600
