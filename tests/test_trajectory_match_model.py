"""Tests for TrajectoryMatchResult Pydantic model (Task 2.1)."""
from __future__ import annotations

import pytest
from ontology.state import TrajectoryMatchResult, DiseaseStateSnapshot
from datetime import datetime


class TestTrajectoryMatchResult:
    def test_basic_construction(self):
        result = TrajectoryMatchResult(
            cohort_size=200,
            matched_k=47,
            median_months_remaining=28.5,
            p25_months=16.0,
            p75_months=42.0,
        )
        assert result.cohort_size == 200
        assert result.matched_k == 47
        assert result.median_months_remaining == 28.5
        assert result.p25_months == 16.0
        assert result.p75_months == 42.0

    def test_defaults(self):
        result = TrajectoryMatchResult(
            cohort_size=100,
            matched_k=10,
            median_months_remaining=24.0,
            p25_months=12.0,
            p75_months=36.0,
        )
        assert result.window_estimates == {}
        assert result.decline_rate_percentile == 0.0

    def test_window_estimates(self):
        result = TrajectoryMatchResult(
            cohort_size=50,
            matched_k=30,
            median_months_remaining=18.0,
            p25_months=9.0,
            p75_months=27.0,
            window_estimates={"molecular": 12.5, "nmj": 8.0, "functional": 6.0},
        )
        assert result.window_estimates["molecular"] == 12.5
        assert result.window_estimates["nmj"] == 8.0
        assert result.window_estimates["functional"] == 6.0

    def test_decline_rate_percentile(self):
        result = TrajectoryMatchResult(
            cohort_size=150,
            matched_k=50,
            median_months_remaining=22.0,
            p25_months=10.0,
            p75_months=34.0,
            decline_rate_percentile=73.5,
        )
        assert result.decline_rate_percentile == 73.5

    def test_model_dump_keys(self):
        result = TrajectoryMatchResult(
            cohort_size=80,
            matched_k=20,
            median_months_remaining=20.0,
            p25_months=10.0,
            p75_months=30.0,
        )
        d = result.model_dump()
        assert "cohort_size" in d
        assert "matched_k" in d
        assert "median_months_remaining" in d
        assert "p25_months" in d
        assert "p75_months" in d
        assert "window_estimates" in d
        assert "decline_rate_percentile" in d

    def test_is_pydantic_base_model(self):
        """TrajectoryMatchResult is a plain BaseModel, not a BaseEnvelope."""
        from pydantic import BaseModel
        from ontology.base import BaseEnvelope
        result = TrajectoryMatchResult(
            cohort_size=1,
            matched_k=1,
            median_months_remaining=1.0,
            p25_months=0.5,
            p75_months=1.5,
        )
        assert isinstance(result, BaseModel)
        assert not isinstance(result, BaseEnvelope)


class TestDiseaseStateSnapshotTrajectory:
    def test_trajectory_match_defaults_none(self):
        snap = DiseaseStateSnapshot(
            id="snap:test",
            subject_ref="patient:erik",
            as_of=datetime(2025, 1, 1),
        )
        assert snap.trajectory_match is None

    def test_trajectory_match_can_be_set(self):
        result = TrajectoryMatchResult(
            cohort_size=200,
            matched_k=50,
            median_months_remaining=30.0,
            p25_months=18.0,
            p75_months=44.0,
        )
        snap = DiseaseStateSnapshot(
            id="snap:test2",
            subject_ref="patient:erik",
            as_of=datetime(2025, 6, 1),
            trajectory_match=result.model_dump(),
        )
        assert snap.trajectory_match is not None
        assert snap.trajectory_match["cohort_size"] == 200
        assert snap.trajectory_match["median_months_remaining"] == 30.0
