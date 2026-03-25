"""tests/test_seed_completeness.py

Validates the structure and completeness of seed JSON files without any DB dependency.
These tests must pass on the raw JSON alone.
"""
import json
from pathlib import Path

SEED_DIR = Path(__file__).parent.parent / "data" / "seed"


def test_all_seed_files_exist():
    for f in [
        "interventions.json",
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
        "drug_design_targets.json",
    ]:
        assert (SEED_DIR / f).exists(), f"Missing: {f}"


def test_interventions_have_required_fields():
    data = json.loads((SEED_DIR / "interventions.json").read_text())
    for item in data:
        assert "id" in item
        assert "name" in item
        assert "intervention_class" in item
        assert "body" in item
        assert "applicable_subtypes" in item["body"]
        assert "regulatory_status" in item["body"]


def test_evidence_items_have_required_fields():
    for layer_file in [
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
    ]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        for item in data:
            assert "id" in item
            assert "claim" in item
            assert "direction" in item
            assert "strength" in item
            body = item.get("body", {})
            assert "protocol_layer" in body, f"Missing protocol_layer in {item['id']}"
            assert "mechanism_target" in body, f"Missing mechanism_target in {item['id']}"
            assert "applicable_subtypes" in body, f"Missing subtypes in {item['id']}"
            assert "pch_layer" in body, f"Missing pch_layer in {item['id']}"


def test_erik_eligibility_assessed():
    for layer_file in [
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
    ]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        for item in data:
            body = item.get("body", {})
            assert "erik_eligible" in body, f"Missing erik_eligible in {item['id']}"


def test_total_evidence_item_count():
    total = 0
    for layer_file in [
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
    ]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        total += len(data)
    assert total >= 80, f"Only {total} evidence items"
