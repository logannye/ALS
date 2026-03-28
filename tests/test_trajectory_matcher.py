"""Tests for TrajectoryMatcher and helper functions (Task 2.2)."""
from __future__ import annotations

import pytest
import numpy as np
from world_model.trajectory_matcher import (
    _dtw_distance,
    _estimate_survival,
    _estimate_windows,
    TrajectoryMatcher,
)


class TestDtwDistance:
    def test_identical_sequences_zero(self):
        seq = [43, 41, 39, 37]
        dist = _dtw_distance(seq, seq)
        assert dist == pytest.approx(0.0)

    def test_different_sequences_positive(self):
        seq_a = [43, 41, 39, 37]
        seq_b = [40, 36, 32, 28]
        dist = _dtw_distance(seq_a, seq_b)
        assert dist > 0.0

    def test_different_lengths(self):
        seq_a = [43, 41, 39]
        seq_b = [43, 41, 39, 37, 35]
        dist = _dtw_distance(seq_a, seq_b)
        assert dist >= 0.0

    def test_single_element_sequences(self):
        dist = _dtw_distance([40], [40])
        assert dist == pytest.approx(0.0)
        dist2 = _dtw_distance([40], [30])
        assert dist2 == pytest.approx(10.0)

    def test_symmetry(self):
        seq_a = [43, 40, 36]
        seq_b = [41, 39, 37, 35]
        dist_ab = _dtw_distance(seq_a, seq_b)
        dist_ba = _dtw_distance(seq_b, seq_a)
        assert dist_ab == pytest.approx(dist_ba, rel=1e-6)

    def test_empty_sequences(self):
        dist = _dtw_distance([], [])
        assert dist == pytest.approx(0.0)

    def test_returns_float(self):
        result = _dtw_distance([43, 41], [42, 40])
        assert isinstance(result, float)


class TestEstimateSurvival:
    def _make_cohort(self, survival_months, vital_statuses=None):
        if vital_statuses is None:
            vital_statuses = [1] * len(survival_months)
        return [
            {"survival_months": m, "vital_status": v}
            for m, v in zip(survival_months, vital_statuses)
        ]

    def test_known_survival_months(self):
        # 10 patients, survival months known
        cohort = self._make_cohort([12, 18, 24, 30, 36, 42, 48, 54, 60, 72])
        result = _estimate_survival(cohort)
        assert "median_months_remaining" in result
        assert "p25_months" in result
        assert "p75_months" in result
        assert result["median_months_remaining"] > 0
        assert result["p25_months"] <= result["median_months_remaining"]
        assert result["median_months_remaining"] <= result["p75_months"]

    def test_uniform_cohort(self):
        cohort = self._make_cohort([24] * 20)
        result = _estimate_survival(cohort)
        assert result["median_months_remaining"] == pytest.approx(24.0, abs=1.0)

    def test_empty_cohort_returns_zeros(self):
        result = _estimate_survival([])
        assert result["median_months_remaining"] == 0.0
        assert result["p25_months"] == 0.0
        assert result["p75_months"] == 0.0

    def test_censored_patients_handled(self):
        # Some patients censored (vital_status=0)
        cohort = [
            {"survival_months": 12, "vital_status": 1},
            {"survival_months": 36, "vital_status": 0},  # censored
            {"survival_months": 24, "vital_status": 1},
            {"survival_months": 48, "vital_status": 0},  # censored
            {"survival_months": 18, "vital_status": 1},
        ]
        result = _estimate_survival(cohort)
        # Should return valid estimates even with censoring
        assert result["median_months_remaining"] >= 0.0

    def test_single_patient(self):
        cohort = [{"survival_months": 30.0, "vital_status": 1}]
        result = _estimate_survival(cohort)
        assert result["median_months_remaining"] >= 0.0

    def test_missing_survival_months_skipped(self):
        cohort = [
            {"survival_months": None, "vital_status": None},
            {"survival_months": 24.0, "vital_status": 1},
            {"survival_months": 36.0, "vital_status": 1},
        ]
        result = _estimate_survival(cohort)
        assert result["median_months_remaining"] > 0


class TestEstimateWindows:
    def _make_trajectories(self, n=10, start_alsfrs=44, rate=-0.5):
        """Make n patients each with 12 monthly time-points."""
        trajs = []
        for i in range(n):
            points = []
            for t in range(12):
                val = start_alsfrs + rate * t
                points.append({"time_months": float(t), "alsfrs_r_total": val})
            trajs.append(points)
        return trajs

    def test_basic_window_estimation(self):
        trajs = self._make_trajectories(n=10, start_alsfrs=44, rate=-1.0)
        thresholds = {"molecular": 40, "nmj": 36, "functional": 32}
        result = _estimate_windows(trajs, current_alsfrs_r=44, thresholds=thresholds)
        assert isinstance(result, dict)
        assert "molecular" in result
        assert "nmj" in result
        assert "functional" in result

    def test_window_times_ordered(self):
        # With declining ALSFRS, molecular > nmj > functional (in time-to-cross terms reversed)
        trajs = self._make_trajectories(n=10, start_alsfrs=44, rate=-1.0)
        # threshold 40: crossed at t=4, 36 at t=8, 32 at t=12
        thresholds = {"early": 40, "mid": 36, "late": 32}
        result = _estimate_windows(trajs, current_alsfrs_r=44, thresholds=thresholds)
        # lower threshold means more time to cross
        assert result["early"] <= result["mid"]
        assert result["mid"] <= result["late"]

    def test_empty_trajectories(self):
        result = _estimate_windows([], current_alsfrs_r=44, thresholds={"layer": 36})
        assert result == {} or result.get("layer", float("inf")) == float("inf")

    def test_threshold_already_crossed(self):
        # All patients already below threshold
        trajs = self._make_trajectories(n=5, start_alsfrs=30, rate=-0.5)
        thresholds = {"layer": 36}
        result = _estimate_windows(trajs, current_alsfrs_r=28, thresholds=thresholds)
        # Should handle gracefully (0.0 or some sentinel)
        assert isinstance(result, dict)

    def test_returns_float_values(self):
        trajs = self._make_trajectories(n=8, start_alsfrs=44, rate=-0.8)
        thresholds = {"test": 38}
        result = _estimate_windows(trajs, current_alsfrs_r=44, thresholds=thresholds)
        for v in result.values():
            assert isinstance(v, float)


class TestTrajectoryMatcher:
    def test_init_defaults(self):
        matcher = TrajectoryMatcher()
        assert matcher.cohort_age_window == 5
        assert matcher.top_k == 50
        assert matcher.thresholds is not None

    def test_init_custom(self):
        matcher = TrajectoryMatcher(cohort_age_window=8, top_k=30)
        assert matcher.cohort_age_window == 8
        assert matcher.top_k == 30

    def test_init_custom_thresholds(self):
        custom = {"layer_a": 36, "layer_b": 28}
        matcher = TrajectoryMatcher(thresholds=custom)
        assert matcher.thresholds == custom

    def test_match_returns_dict(self, monkeypatch):
        """match() returns a dict compatible with TrajectoryMatchResult."""
        matcher = TrajectoryMatcher()

        # Patch the DB query to return empty so we test the no-data path
        monkeypatch.setattr(
            matcher, "_query_cohort", lambda *a, **kw: []
        )
        result = matcher.match(
            age=67, sex="Male", onset_region="Limb",
            alsfrs_r=43, fvc_percent=88.0,
        )
        assert isinstance(result, dict)
        assert "cohort_size" in result
        assert "matched_k" in result
        assert "median_months_remaining" in result
        assert "p25_months" in result
        assert "p75_months" in result
        assert "window_estimates" in result
        assert "decline_rate_percentile" in result

    def test_match_empty_cohort_safe(self, monkeypatch):
        matcher = TrajectoryMatcher()
        monkeypatch.setattr(matcher, "_query_cohort", lambda *a, **kw: [])
        result = matcher.match(age=67, sex="Male", onset_region="Limb",
                               alsfrs_r=43, fvc_percent=88.0)
        assert result["cohort_size"] == 0
        assert result["matched_k"] == 0
        assert result["median_months_remaining"] == 0.0

    def test_match_result_compatible_with_model(self, monkeypatch):
        """Result dict should be accepted by TrajectoryMatchResult.model_validate()."""
        from ontology.state import TrajectoryMatchResult
        matcher = TrajectoryMatcher()
        monkeypatch.setattr(matcher, "_query_cohort", lambda *a, **kw: [])
        result = matcher.match(age=67, sex="Male", onset_region="Limb",
                               alsfrs_r=43, fvc_percent=88.0)
        model = TrajectoryMatchResult.model_validate(result)
        assert model.cohort_size == 0
