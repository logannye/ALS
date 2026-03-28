"""Tests for ProactConnector (Task 2.2)."""
from __future__ import annotations

import pytest
from connectors.proact import ProactConnector


class TestParseRow:
    def test_valid_full_row(self):
        row = {
            "patient_id": "42",
            "age_onset": "63",
            "sex": "Male",
            "onset_region": "Limb",
            "alsfrs_r_total": "41",
            "fvc_percent": "88.5",
            "time_months": "12",
            "vital_status": "1",
            "survival_months": "36",
        }
        result = ProactConnector._parse_row(row)
        assert result is not None
        assert result["patient_id"] == "42"
        assert result["age_onset"] == 63.0
        assert result["sex"] == "Male"
        assert result["onset_region"] == "Limb"
        assert result["alsfrs_r_total"] == 41.0
        assert result["fvc_percent"] == 88.5
        assert result["time_months"] == 12.0
        assert result["vital_status"] == 1
        assert result["survival_months"] == 36.0

    def test_missing_optional_fields_use_none(self):
        row = {
            "patient_id": "7",
            "time_months": "6",
        }
        result = ProactConnector._parse_row(row)
        assert result is not None
        assert result["patient_id"] == "7"
        assert result["age_onset"] is None
        assert result["sex"] is None
        assert result["onset_region"] is None
        assert result["alsfrs_r_total"] is None
        assert result["fvc_percent"] is None
        assert result["time_months"] == 6.0
        assert result["vital_status"] is None
        assert result["survival_months"] is None

    def test_empty_row_returns_none(self):
        result = ProactConnector._parse_row({})
        assert result is None

    def test_non_numeric_fields_handled_gracefully(self):
        row = {
            "patient_id": "99",
            "age_onset": "unknown",
            "alsfrs_r_total": "n/a",
            "fvc_percent": "",
            "time_months": "0",
        }
        result = ProactConnector._parse_row(row)
        assert result is not None
        assert result["patient_id"] == "99"
        assert result["age_onset"] is None
        assert result["alsfrs_r_total"] is None
        assert result["fvc_percent"] is None
        assert result["time_months"] == 0.0

    def test_sex_normalization_preserved(self):
        row = {"patient_id": "5", "sex": "female", "time_months": "3"}
        result = ProactConnector._parse_row(row)
        assert result is not None
        assert result["sex"] == "female"

    def test_vital_status_integer(self):
        row = {"patient_id": "8", "vital_status": "0", "time_months": "10"}
        result = ProactConnector._parse_row(row)
        assert result is not None
        assert result["vital_status"] == 0

    def test_parse_row_returns_expected_keys(self):
        row = {"patient_id": "1", "time_months": "5"}
        result = ProactConnector._parse_row(row)
        expected_keys = {
            "patient_id", "age_onset", "sex", "onset_region",
            "alsfrs_r_total", "fvc_percent", "time_months",
            "vital_status", "survival_months",
        }
        assert set(result.keys()) == expected_keys


class TestProactConnectorInit:
    def test_init_with_data_dir(self):
        connector = ProactConnector(data_dir="/tmp/proact_test")
        assert connector.data_dir == "/tmp/proact_test"

    def test_init_with_none_data_dir(self):
        connector = ProactConnector(data_dir=None)
        assert connector.data_dir is None
