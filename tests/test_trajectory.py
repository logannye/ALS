# tests/test_trajectory.py
"""Tests for PRO-ACT cohort matching and trajectory prediction."""
from __future__ import annotations
import pytest
from research.trajectory import ProACTAnalyzer, CohortMatch, _parse_alsfrs_csv

class TestParseAlsfrsCsv:
    def test_parses_rows(self):
        csv_data = "SubjectID,ALSFRS_Delta,ALSFRS_R_Total\n1,0,43\n1,30,41\n1,60,39\n2,0,38\n2,30,35\n"
        records = _parse_alsfrs_csv(csv_data)
        assert len(records) == 5
        assert records[0]["subject_id"] == "1"
        assert records[0]["alsfrs_r_total"] == 43

    def test_empty_csv(self):
        assert _parse_alsfrs_csv("") == []

class TestCohortMatch:
    def test_construction(self):
        match = CohortMatch(n_patients=150, median_decline_rate=-0.8, p25_decline_rate=-1.2, p75_decline_rate=-0.4, median_survival_months=36, erik_percentile=65)
        assert match.n_patients == 150
        assert match.erik_percentile == 65

class TestProACTAnalyzer:
    def test_instantiates(self):
        analyzer = ProACTAnalyzer()
        assert analyzer._loaded is False

    def test_match_cohort_without_data(self):
        analyzer = ProACTAnalyzer()
        match = analyzer.match_cohort(age=67, sex="male", onset_region="lower_limb", baseline_alsfrs_r=43, decline_rate=-0.39)
        assert match.n_patients == 0

    def test_load_from_records(self):
        records = [
            {"subject_id": "1", "alsfrs_delta": 0, "alsfrs_r_total": 44, "age": 65, "sex": "Male", "onset": "Limb"},
            {"subject_id": "1", "alsfrs_delta": 90, "alsfrs_r_total": 38, "age": 65, "sex": "Male", "onset": "Limb"},
            {"subject_id": "2", "alsfrs_delta": 0, "alsfrs_r_total": 42, "age": 68, "sex": "Male", "onset": "Limb"},
            {"subject_id": "2", "alsfrs_delta": 90, "alsfrs_r_total": 36, "age": 68, "sex": "Male", "onset": "Limb"},
        ]
        analyzer = ProACTAnalyzer()
        analyzer._load_from_records(records)
        assert analyzer._loaded is True
        assert len(analyzer._subjects) > 0

    def test_match_cohort_with_data(self):
        records = [
            {"subject_id": str(i), "alsfrs_delta": 0, "alsfrs_r_total": 43, "age": 66, "sex": "Male", "onset": "Limb"}
            for i in range(20)
        ] + [
            {"subject_id": str(i), "alsfrs_delta": 90, "alsfrs_r_total": 40, "age": 66, "sex": "Male", "onset": "Limb"}
            for i in range(20)
        ]
        analyzer = ProACTAnalyzer()
        analyzer._load_from_records(records)
        match = analyzer.match_cohort(age=67, sex="male", onset_region="lower_limb", baseline_alsfrs_r=43, decline_rate=-0.39)
        assert match.n_patients > 0
        assert match.median_decline_rate < 0
