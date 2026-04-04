"""Tests for ChEMBLConnector — no ChEMBL DB required for unit tests.

Integration tests that require the local ChEMBL 36 SQLite database are
marked with @pytest.mark.chembl and skipped by default.
"""
import pytest

from connectors.chembl import (
    ChEMBLConnector,
    _build_bioactivity_query,
    _build_compound_properties_query,
    _build_drug_mechanism_query,
    _build_metabolism_query,
    _build_drug_indication_query,
    _build_properties_by_target_query,
)


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


# ---------------------------------------------------------------------------
# ADME/Tox expansion: query builder unit tests
# ---------------------------------------------------------------------------

class TestBuildCompoundPropertiesQuery:
    def test_returns_tuple(self):
        sql, params = _build_compound_properties_query("CHEMBL25")
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_sql_contains_expected_tables(self):
        sql, _ = _build_compound_properties_query("CHEMBL25")
        sql_upper = sql.upper()
        assert "MOLECULE_DICTIONARY" in sql_upper
        assert "COMPOUND_PROPERTIES" in sql_upper
        assert "COMPOUND_STRUCTURES" in sql_upper

    def test_params_contain_chembl_id(self):
        _, params = _build_compound_properties_query("CHEMBL25")
        assert "CHEMBL25" in params

    def test_sql_is_select(self):
        sql, _ = _build_compound_properties_query("CHEMBL25")
        assert sql.strip().upper().startswith("SELECT")


class TestBuildDrugMechanismQuery:
    def test_returns_tuple_with_target(self):
        sql, params = _build_drug_mechanism_query("CHEMBL1862", 50)
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_returns_tuple_without_target(self):
        sql, params = _build_drug_mechanism_query(None, 50)
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_sql_contains_expected_tables(self):
        sql, _ = _build_drug_mechanism_query("CHEMBL1862", 50)
        sql_upper = sql.upper()
        assert "DRUG_MECHANISM" in sql_upper
        assert "MOLECULE_DICTIONARY" in sql_upper
        assert "TARGET_DICTIONARY" in sql_upper

    def test_params_with_target_include_target_id(self):
        _, params = _build_drug_mechanism_query("CHEMBL1862", 50)
        assert "CHEMBL1862" in params
        assert 50 in params

    def test_params_without_target_include_only_limit(self):
        _, params = _build_drug_mechanism_query(None, 100)
        assert params == (100,)

    def test_sql_is_select(self):
        sql, _ = _build_drug_mechanism_query(None, 50)
        assert sql.strip().upper().startswith("SELECT")


class TestBuildMetabolismQuery:
    def test_returns_tuple(self):
        sql, params = _build_metabolism_query("CHEMBL25")
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_sql_contains_expected_tables(self):
        sql, _ = _build_metabolism_query("CHEMBL25")
        sql_upper = sql.upper()
        assert "METABOLISM" in sql_upper
        assert "COMPOUND_RECORDS" in sql_upper
        assert "MOLECULE_DICTIONARY" in sql_upper

    def test_params_contain_chembl_id(self):
        _, params = _build_metabolism_query("CHEMBL25")
        assert "CHEMBL25" in params

    def test_sql_is_select(self):
        sql, _ = _build_metabolism_query("CHEMBL25")
        assert sql.strip().upper().startswith("SELECT")


class TestBuildDrugIndicationQuery:
    def test_returns_tuple(self):
        sql, params = _build_drug_indication_query("CHEMBL25")
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_sql_contains_expected_tables(self):
        sql, _ = _build_drug_indication_query("CHEMBL25")
        sql_upper = sql.upper()
        assert "DRUG_INDICATION" in sql_upper
        assert "MOLECULE_DICTIONARY" in sql_upper

    def test_params_contain_chembl_id(self):
        _, params = _build_drug_indication_query("CHEMBL25")
        assert "CHEMBL25" in params

    def test_sql_orders_by_max_phase(self):
        sql, _ = _build_drug_indication_query("CHEMBL25")
        assert "max_phase_for_ind" in sql.lower()
        assert "desc" in sql.lower()

    def test_sql_is_select(self):
        sql, _ = _build_drug_indication_query("CHEMBL25")
        assert sql.strip().upper().startswith("SELECT")


class TestBuildPropertiesByTargetQuery:
    def test_returns_tuple(self):
        sql, params = _build_properties_by_target_query("Q99720", 50)
        assert isinstance(sql, str)
        assert isinstance(params, tuple)

    def test_sql_contains_expected_tables(self):
        sql, _ = _build_properties_by_target_query("Q99720", 50)
        sql_upper = sql.upper()
        assert "ACTIVITIES" in sql_upper
        assert "ASSAYS" in sql_upper
        assert "TARGET_DICTIONARY" in sql_upper
        assert "COMPOUND_PROPERTIES" in sql_upper
        assert "COMPONENT_SEQUENCES" in sql_upper

    def test_params_contain_uniprot_and_limit(self):
        _, params = _build_properties_by_target_query("Q99720", 50)
        assert "Q99720" in params
        assert 50 in params

    def test_sql_is_select(self):
        sql, _ = _build_properties_by_target_query("Q99720", 50)
        assert sql.strip().upper().startswith("SELECT")


# ---------------------------------------------------------------------------
# ADME/Tox expansion: evidence converter tests
# ---------------------------------------------------------------------------

class _FakeRow(dict):
    """Dict subclass that supports bracket-access like sqlite3.Row."""
    def __getitem__(self, key):
        return self.get(key)


class TestPropsToEvidenceItem:
    def test_creates_evidence_item(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name="ASPIRIN",
            mw_freebase=180.16,
            full_mwt=180.16,
            alogp=1.31,
            hba=4,
            hbd=1,
            psa=63.60,
            num_ro5_violations=0,
            aromatic_rings=1,
            heavy_atoms=13,
            qed_weighted=0.55,
            canonical_smiles="CC(=O)Oc1ccccc1C(=O)O",
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._props_to_evidence_item(row)
        assert item.id == "evi:chembl_props:CHEMBL25"
        assert "ASPIRIN" in item.claim
        assert item.body["pch_layer"] == 2
        assert item.body["alogp"] == 1.31

    def test_handles_none_molecule_name(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name=None,
            mw_freebase=180.16,
            full_mwt=180.16,
            alogp=1.31,
            hba=4,
            hbd=1,
            psa=63.60,
            num_ro5_violations=0,
            aromatic_rings=1,
            heavy_atoms=13,
            qed_weighted=0.55,
            canonical_smiles=None,
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._props_to_evidence_item(row)
        assert "CHEMBL25" in item.claim


class TestMoaToEvidenceItem:
    def test_creates_evidence_item(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name="ASPIRIN",
            target_chembl_id="CHEMBL1862",
            target_name="Cyclooxygenase-2",
            mechanism_of_action="Cyclooxygenase inhibitor",
            action_type="INHIBITOR",
            direct_interaction=1,
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._moa_to_evidence_item(row)
        assert item.id == "evi:chembl_moa:CHEMBL25_CHEMBL1862"
        assert "Cyclooxygenase inhibitor" in item.claim
        assert item.body["direct_interaction"] == 1


class TestMetabolismToEvidenceItem:
    def test_creates_evidence_item(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name="ASPIRIN",
            enzyme_name="CYP2C9",
            met_conversion="Hydroxylation",
            metabolite_chembl_id="CHEMBL1234",
            metabolite_name="SALICYLIC ACID",
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._metabolism_to_evidence_item(row)
        assert item.id == "evi:chembl_met:CHEMBL25_cyp2c9"
        assert "CYP2C9" in item.claim
        assert item.body["met_conversion"] == "Hydroxylation"

    def test_normalizes_enzyme_name_in_id(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name="ASPIRIN",
            enzyme_name="CYP3A4/5",
            met_conversion="Dealkylation",
            metabolite_chembl_id="CHEMBL999",
            metabolite_name="MET999",
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._metabolism_to_evidence_item(row)
        assert item.id == "evi:chembl_met:CHEMBL25_cyp3a4_5"


class TestIndicationToEvidenceItem:
    def test_creates_evidence_item(self):
        row = _FakeRow(
            molecule_chembl_id="CHEMBL25",
            molecule_name="ASPIRIN",
            max_phase_for_ind=4.0,
            mesh_id="D006261",
            mesh_heading="Headache",
            efo_id="EFO_0003821",
            efo_term="headache",
        )
        c = ChEMBLConnector(db_path="/nonexistent")
        item = c._indication_to_evidence_item(row)
        assert item.id == "evi:chembl_ind:CHEMBL25_D006261"
        assert "Headache" in item.claim
        assert item.body["max_phase_for_ind"] == 4.0
        assert item.body["efo_id"] == "EFO_0003821"


# ---------------------------------------------------------------------------
# ADME/Tox expansion: fetch method error handling
# ---------------------------------------------------------------------------

def test_fetch_compound_properties_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_compound_properties("CHEMBL25")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_drug_mechanisms_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_drug_mechanisms(target_chembl_id="CHEMBL1862")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_metabolism_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_metabolism("CHEMBL25")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_drug_indications_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_drug_indications("CHEMBL25")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_full_profile_missing_db_returns_error():
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_full_profile("Q99720")
    assert len(result.errors) > 0
    assert result.evidence_items_added == 0


def test_fetch_full_profile_returns_connector_result():
    from connectors.base import ConnectorResult
    c = ChEMBLConnector(db_path="/nonexistent/path/chembl.db")
    result = c.fetch_full_profile("Q99720")
    assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# Loop executor: _exec_query_chembl should use fetch_full_profile
# ---------------------------------------------------------------------------

class TestChEMBLExecutorFullProfile:
    """The loop executor should use fetch_full_profile when a UniProt ID is available."""

    def test_executor_calls_fetch_full_profile(self):
        """_exec_query_chembl should call fetch_full_profile with a UniProt ID."""
        from unittest.mock import patch, MagicMock
        from connectors.base import ConnectorResult
        from research.state import initial_state, ResearchState
        from research.loop import _exec_query_chembl

        state = initial_state("traj:test")
        state = ResearchState(**{**state.to_dict(), "step_count": 0})

        mock_connector = MagicMock()
        mock_connector.fetch_full_profile.return_value = ConnectorResult()

        with patch("connectors.chembl.ChEMBLConnector", return_value=mock_connector):
            result = _exec_query_chembl({}, state, None, None)

        mock_connector.fetch_full_profile.assert_called_once()
        call_kwargs = mock_connector.fetch_full_profile.call_args
        # Should be called with a uniprot_id from ALS_TARGETS
        assert call_kwargs is not None

    def test_executor_rotates_targets(self):
        """Different step_count values should query different UniProt IDs."""
        from unittest.mock import patch, MagicMock
        from connectors.base import ConnectorResult
        from research.state import initial_state, ResearchState
        from research.loop import _exec_query_chembl

        uniprot_ids = set()
        for step in range(5):
            state = initial_state("traj:test")
            state = ResearchState(**{**state.to_dict(), "step_count": step})

            mock_connector = MagicMock()
            mock_connector.fetch_full_profile.return_value = ConnectorResult()

            with patch("connectors.chembl.ChEMBLConnector", return_value=mock_connector):
                _exec_query_chembl({}, state, None, None)

            call_args = mock_connector.fetch_full_profile.call_args
            if call_args:
                # Extract the uniprot_id positional or keyword arg
                uid = call_args.kwargs.get("uniprot_id") or (call_args.args[0] if call_args.args else None)
                uniprot_ids.add(uid)

        # Should have queried at least 2 different targets across 5 steps
        assert len(uniprot_ids) >= 2

    def test_executor_returns_action_result(self):
        """_exec_query_chembl should return an ActionResult."""
        from unittest.mock import patch, MagicMock
        from connectors.base import ConnectorResult
        from research.state import initial_state, ResearchState
        from research.actions import ActionResult, ActionType
        from research.loop import _exec_query_chembl

        state = initial_state("traj:test")
        state = ResearchState(**{**state.to_dict(), "step_count": 0})

        mock_connector = MagicMock()
        cr = ConnectorResult()
        cr.evidence_items_added = 3
        mock_connector.fetch_full_profile.return_value = cr

        with patch("connectors.chembl.ChEMBLConnector", return_value=mock_connector):
            result = _exec_query_chembl({}, state, None, None)

        assert isinstance(result, ActionResult)
        assert result.action == ActionType.QUERY_CHEMBL
        assert result.success is True
        assert result.evidence_items_added == 3
