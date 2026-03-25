"""Tests for canonical ALS drug target definitions."""
import pytest

from targets.als_targets import ALS_TARGETS, get_target, get_targets_for_subtype, get_targets_for_protocol_layer


REQUIRED_FIELDS = {"name", "gene", "uniprot_id", "description", "subtypes", "protocol_layers", "druggable", "druggability_notes"}


class TestALSTargetsDict:
    def test_has_at_least_fifteen_entries(self):
        assert len(ALS_TARGETS) >= 15

    def test_has_sixteen_entries(self):
        assert len(ALS_TARGETS) == 16

    def test_all_targets_have_required_fields(self):
        for name, target in ALS_TARGETS.items():
            missing = REQUIRED_FIELDS - set(target.keys())
            assert not missing, f"Target '{name}' missing fields: {missing}"

    def test_all_subtypes_are_lists(self):
        for name, target in ALS_TARGETS.items():
            assert isinstance(target["subtypes"], list), f"Target '{name}' subtypes must be a list"

    def test_all_protocol_layers_are_lists(self):
        for name, target in ALS_TARGETS.items():
            assert isinstance(target["protocol_layers"], list), f"Target '{name}' protocol_layers must be a list"

    def test_all_druggable_are_bool(self):
        for name, target in ALS_TARGETS.items():
            assert isinstance(target["druggable"], bool), f"Target '{name}' druggable must be bool"

    def test_all_have_nonempty_name(self):
        for name, target in ALS_TARGETS.items():
            assert target["name"], f"Target '{name}' has empty name"


class TestGetTarget:
    def test_get_tdp43_returns_correct_uniprot(self):
        t = get_target("TDP-43")
        assert t is not None
        assert t["uniprot_id"] == "Q13148"

    def test_get_sod1_returns_correct_uniprot(self):
        t = get_target("SOD1")
        assert t is not None
        assert t["uniprot_id"] == "P00441"

    def test_get_nonexistent_returns_none(self):
        assert get_target("NONEXISTENT") is None

    def test_get_target_returns_dict(self):
        t = get_target("TDP-43")
        assert isinstance(t, dict)

    def test_get_fus(self):
        t = get_target("FUS")
        assert t is not None
        assert t["uniprot_id"] == "P35637"

    def test_get_c9orf72(self):
        t = get_target("C9orf72")
        assert t is not None
        assert t["uniprot_id"] == "Q96LT7"

    def test_get_stmn2(self):
        t = get_target("STMN2")
        assert t is not None
        assert t["uniprot_id"] == "Q93045"

    def test_get_unc13a(self):
        t = get_target("UNC13A")
        assert t is not None
        assert t["uniprot_id"] == "Q9UPW8"

    def test_get_sigmar1(self):
        t = get_target("SIGMAR1")
        assert t is not None
        assert t["uniprot_id"] == "Q99720"

    def test_get_eaat2(self):
        t = get_target("EAAT2")
        assert t is not None
        assert t["uniprot_id"] == "P43004"

    def test_get_bdnf(self):
        t = get_target("BDNF")
        assert t is not None
        assert t["uniprot_id"] == "P23560"

    def test_get_gdnf(self):
        t = get_target("GDNF")
        assert t is not None
        assert t["uniprot_id"] == "P39905"

    def test_get_optn(self):
        t = get_target("OPTN")
        assert t is not None
        assert t["uniprot_id"] == "Q96CV9"
        assert t["druggable"] is False

    def test_get_tbk1(self):
        t = get_target("TBK1")
        assert t is not None
        assert t["uniprot_id"] == "Q9UHD2"

    def test_get_nek1(self):
        t = get_target("NEK1")
        assert t is not None
        assert t["uniprot_id"] == "Q96PY6"
        assert t["druggable"] is False

    def test_get_complement_c5(self):
        t = get_target("Complement C5")
        assert t is not None
        assert t["uniprot_id"] == "P01031"

    def test_get_csf1r(self):
        t = get_target("CSF1R")
        assert t is not None
        assert t["uniprot_id"] == "P07333"

    def test_get_mtor(self):
        t = get_target("mTOR")
        assert t is not None
        assert t["uniprot_id"] == "P42345"


class TestGetTargetsForSubtype:
    def test_sporadic_tdp43_includes_tdp43(self):
        results = get_targets_for_subtype("sporadic_tdp43")
        names = [t["name"] for t in results]
        assert "TDP-43" in names

    def test_sporadic_tdp43_includes_stmn2(self):
        results = get_targets_for_subtype("sporadic_tdp43")
        names = [t["name"] for t in results]
        assert "STMN2" in names

    def test_sporadic_tdp43_includes_unc13a(self):
        results = get_targets_for_subtype("sporadic_tdp43")
        names = [t["name"] for t in results]
        assert "UNC13A" in names

    def test_sod1_includes_sod1_target(self):
        results = get_targets_for_subtype("sod1")
        names = [t["name"] for t in results]
        assert "SOD1" in names

    def test_unknown_subtype_returns_empty_list(self):
        results = get_targets_for_subtype("nonexistent_subtype")
        assert results == []

    def test_returns_list_of_dicts(self):
        results = get_targets_for_subtype("sod1")
        assert isinstance(results, list)
        for t in results:
            assert isinstance(t, dict)

    def test_all_subtype_results_have_required_fields(self):
        results = get_targets_for_subtype("sporadic_tdp43")
        for t in results:
            missing = REQUIRED_FIELDS - set(t.keys())
            assert not missing


class TestGetTargetsForProtocolLayer:
    def test_root_cause_suppression_includes_sod1(self):
        results = get_targets_for_protocol_layer("root_cause_suppression")
        names = [t["name"] for t in results]
        assert "SOD1" in names

    def test_root_cause_suppression_includes_tdp43_or_sod1(self):
        results = get_targets_for_protocol_layer("root_cause_suppression")
        names = [t["name"] for t in results]
        assert "TDP-43" in names or "SOD1" in names

    def test_pathology_reversal_includes_stmn2(self):
        results = get_targets_for_protocol_layer("pathology_reversal")
        names = [t["name"] for t in results]
        assert "STMN2" in names

    def test_regeneration_reinnervation_includes_bdnf(self):
        results = get_targets_for_protocol_layer("regeneration_reinnervation")
        names = [t["name"] for t in results]
        assert "BDNF" in names

    def test_circuit_stabilization_includes_eaat2(self):
        results = get_targets_for_protocol_layer("circuit_stabilization")
        names = [t["name"] for t in results]
        assert "EAAT2" in names

    def test_unknown_layer_returns_empty_list(self):
        results = get_targets_for_protocol_layer("nonexistent_layer")
        assert results == []

    def test_returns_list_of_dicts(self):
        results = get_targets_for_protocol_layer("root_cause_suppression")
        assert isinstance(results, list)
        for t in results:
            assert isinstance(t, dict)
