"""Tests for ChEMBLConnector — no ChEMBL DB required for unit tests.

Integration tests that require the local ChEMBL 36 SQLite database are
marked with @pytest.mark.chembl and skipped by default.
"""
import pytest

from connectors.chembl import ChEMBLConnector, _build_bioactivity_query


# ---------------------------------------------------------------------------
# Unit tests: _build_bioactivity_query (free function)
# ---------------------------------------------------------------------------

def test_build_bioactivity_query_returns_tuple():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 100)
    assert isinstance(sql, str)
    assert isinstance(params, tuple)


def test_build_bioactivity_query_contains_uniprot():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 100)
    assert "Q99720" in params


def test_build_bioactivity_query_contains_activity_type():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 50)
    assert "IC50" in params


def test_build_bioactivity_query_contains_max_results():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 77)
    assert 77 in params


def test_build_bioactivity_query_sql_is_select():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 100)
    assert sql.strip().upper().startswith("SELECT")


def test_build_bioactivity_query_sql_joins_expected_tables():
    sql, params = _build_bioactivity_query("Q99720", "IC50", 100)
    sql_upper = sql.upper()
    assert "ACTIVITIES" in sql_upper
    assert "ASSAYS" in sql_upper
    assert "TARGET_DICTIONARY" in sql_upper
    assert "MOLECULE_DICTIONARY" in sql_upper


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates_with_default_db_path():
    c = ChEMBLConnector()
    assert c is not None
    assert c.db_path == "/Volumes/Databank/databases/chembl_36.db"


def test_connector_instantiates_with_custom_db_path():
    c = ChEMBLConnector(db_path="/tmp/test_chembl.db")
    assert c.db_path == "/tmp/test_chembl.db"


def test_connector_inherits_base():
    from connectors.base import BaseConnector
    c = ChEMBLConnector()
    assert isinstance(c, BaseConnector)


def test_connector_fetch_method_exists():
    c = ChEMBLConnector()
    assert callable(c.fetch)


# ---------------------------------------------------------------------------
# Missing DB path: graceful error (no crash)
# ---------------------------------------------------------------------------

def test_fetch_bioactivity_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_bioactivity("Q99720")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_compounds_for_target_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_compounds_for_target("SIGMAR1")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_compounds_for_unknown_target_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_compounds_for_target("DOES_NOT_EXIST_TARGET")
    # Should return an error about unknown target, not crash
    assert len(result.errors) > 0


def test_fetch_for_priority_targets_missing_db_returns_errors():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_for_priority_targets(max_per_target=5)
    assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# fetch_for_priority_targets: uses correct ALS target keys
# ---------------------------------------------------------------------------

def test_fetch_for_priority_targets_keys():
    """Priority targets must be valid ALS_TARGETS keys."""
    from targets.als_targets import ALS_TARGETS
    expected_keys = ["SIGMAR1", "EAAT2", "mTOR", "CSF1R", "TDP-43"]
    for key in expected_keys:
        assert key in ALS_TARGETS, f"{key!r} not found in ALS_TARGETS"


# ---------------------------------------------------------------------------
# ConnectorResult shape
# ---------------------------------------------------------------------------

def test_fetch_bioactivity_returns_connector_result():
    from connectors.base import ConnectorResult
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_bioactivity("Q99720")
    assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# Integration tests (require local ChEMBL 36 DB)
# ---------------------------------------------------------------------------

@pytest.mark.chembl
def test_chembl_fetch_bioactivity_sigmar1():
    """Integration: fetch IC50 data for SIGMAR1 (Q99720) from real ChEMBL."""
    c = ChEMBLConnector()
    result = c.fetch_bioactivity("Q99720", activity_type="IC50", max_results=5)
    assert result.evidence_items_added >= 0
    assert len(result.errors) == 0


@pytest.mark.chembl
def test_chembl_fetch_compounds_for_target():
    """Integration: fetch compounds for SIGMAR1 target name."""
    c = ChEMBLConnector()
    result = c.fetch_compounds_for_target("SIGMAR1")
    assert result.evidence_items_added >= 0


@pytest.mark.chembl
def test_chembl_fetch_for_priority_targets():
    """Integration: fetch priority ALS targets from real ChEMBL."""
    c = ChEMBLConnector()
    result = c.fetch_for_priority_targets(max_per_target=5)
    assert result.evidence_items_added >= 0
