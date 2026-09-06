"""Tests for the health recommendations engine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.features.metrics.recommendations import (
    _bus_factor,
    _churn,
    _complexity,
    _health_trend,
    generate_recommendations,
)


def _s(**kw):
    defaults = dict(
        health_score=75.0,
        avg_complexity=5.0,
        churn_rate=0.15,
        hotspot_count=2,
        hotspot_persistence_score=20,
        dependency_density=0.3,
        has_cycles=False,
        avg_semantic_drift=0.1,
        computed_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return MagicMock(**defaults)


def _bf(risk="low", pct=30.0):
    return MagicMock(module_path="src/x.py", risk_level=risk, top_contributor_pct=pct)


class TestHealthTrend:
    def test_decline(self):
        snaps = [_s(health_score=40)] * 4 + [_s(health_score=80)] * 8
        r = _health_trend(snaps)
        assert r and r[0]["severity"] == "critical"

    def test_drift(self):
        snaps = [_s(health_score=65)] * 4 + [_s(health_score=70)] * 8
        r = _health_trend(snaps)
        assert r and r[0]["severity"] == "high"

    def test_stable(self):
        assert _health_trend([_s()] * 12) == []


class TestComplexity:
    def test_critical(self):
        r = _complexity([_s(avg_complexity=12.0)])
        assert r and r[0]["severity"] == "critical"

    def test_spike(self):
        r = _complexity([_s(avg_complexity=8.0), _s(avg_complexity=4.0)])
        assert any(x["id"] == "complexity_spike" for x in r)


class TestBusFactor:
    def test_critical(self):
        r = _bus_factor([_bf(risk="critical")])
        assert r and r[0]["severity"] == "critical"

    def test_high_only(self):
        r = _bus_factor([_bf(risk="high")])
        assert r and r[0]["severity"] == "high"

    def test_none(self):
        assert _bus_factor([_bf(risk="low")]) == []


class TestChurn:
    def test_high(self):
        r = _churn([_s(churn_rate=0.5)])
        assert r and r[0]["severity"] == "critical"


class TestGenerate:
    @pytest.mark.anyio
    async def test_not_found(self):
        db = AsyncMock()
        db.get.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await generate_recommendations(db, 999)

    @pytest.mark.anyio
    async def test_empty(self):
        db = AsyncMock()
        db.get.return_value = MagicMock(name="r", repo_slug="r")
        e = MagicMock()
        e.scalars.return_value.all.return_value = []
        db.execute.return_value = e
        r = await generate_recommendations(db, 1)
        assert r["health_score"] == 100
        assert r["total_recommendations"] == 0
