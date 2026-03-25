"""Tests for scripts/ingestion/lab_results.py — lab panel parsing."""

from datetime import date

import pytest

from ingestion.lab_results import parse_lab_panel
from ontology.observation import Observation, LabResult
from ontology.enums import ObservationKind


SUBJECT = "patient:erik_draper"


class TestParseLabPanel:
    """parse_lab_panel converts raw dicts into typed Observation[LabResult]."""

    def test_nfl_abnormal(self):
        raw = [
            {
                "name": "NfL Plasma",
                "value": 5.82,
                "unit": "pg/mL",
                "ref_low": 0.0,
                "ref_high": 3.65,
                "date": "2026-02-20",
            }
        ]
        obs_list = parse_lab_panel(raw, SUBJECT)
        assert len(obs_list) == 1
        obs = obs_list[0]

        # Envelope checks
        assert isinstance(obs, Observation)
        assert obs.observation_kind == ObservationKind.lab_result
        assert obs.subject_ref == SUBJECT
        assert obs.id == "obs:lab:nfl_plasma:2026-02-20"

        # LabResult sub-object checks
        lr = obs.lab_result
        assert lr is not None
        assert lr.value == 5.82
        assert lr.unit == "pg/mL"
        assert lr.reference_high == 3.65
        assert lr.is_high is True
        assert lr.is_abnormal is True
        assert lr.collection_date == date(2026, 2, 20)

    def test_ck_normal(self):
        raw = [
            {
                "name": "CK",
                "value": 200.0,
                "unit": "U/L",
                "ref_low": 51.0,
                "ref_high": 298.0,
                "date": "2026-02-20",
            }
        ]
        obs_list = parse_lab_panel(raw, SUBJECT)
        assert len(obs_list) == 1
        obs = obs_list[0]
        lr = obs.lab_result
        assert lr is not None
        assert lr.is_abnormal is False
        assert lr.is_high is False
        assert lr.is_low is False

    def test_multiple_labs(self):
        raw = [
            {"name": "NfL Plasma", "value": 5.82, "unit": "pg/mL",
             "ref_low": 0.0, "ref_high": 3.65, "date": "2026-02-20"},
            {"name": "CK", "value": 200.0, "unit": "U/L",
             "ref_low": 51.0, "ref_high": 298.0, "date": "2026-02-20"},
            {"name": "Glucose", "value": 128.0, "unit": "mg/dL",
             "ref_low": 74.0, "ref_high": 99.0, "date": "2026-02-20"},
        ]
        obs_list = parse_lab_panel(raw, SUBJECT)
        assert len(obs_list) == 3
        # Glucose should be flagged high
        glucose = [o for o in obs_list if "glucose" in o.id][0]
        assert glucose.lab_result.is_high is True

    def test_empty_panel(self):
        obs_list = parse_lab_panel([], SUBJECT)
        assert obs_list == []

    def test_id_snake_case(self):
        raw = [
            {"name": "Sed Rate", "value": 2.0, "unit": "mm/hr",
             "ref_low": 0.0, "ref_high": 15.0, "date": "2026-02-20"},
        ]
        obs_list = parse_lab_panel(raw, SUBJECT)
        assert obs_list[0].id == "obs:lab:sed_rate:2026-02-20"

    def test_low_flag(self):
        raw = [
            {"name": "HDL", "value": 36.0, "unit": "mg/dL",
             "ref_low": 39.0, "ref_high": None, "date": "2025-06-09"},
        ]
        obs_list = parse_lab_panel(raw, SUBJECT)
        lr = obs_list[0].lab_result
        assert lr.is_low is True
        assert lr.is_abnormal is True
