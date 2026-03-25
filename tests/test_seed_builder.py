"""tests/test_seed_builder.py

DB-backed tests for the seed builder.  All tests require a live PostgreSQL
connection and are skipped automatically when the DB is not reachable (via the
``db_available`` session fixture in conftest.py).
"""


def test_load_interventions(db_available):
    from evidence.seed_builder import load_seed
    stats = load_seed()
    assert stats["interventions_loaded"] >= 20


def test_load_evidence_items(db_available):
    from evidence.seed_builder import load_seed
    stats = load_seed()
    assert stats["evidence_items_loaded"] >= 80


def test_all_layers_have_evidence(db_available):
    from evidence.seed_builder import load_seed
    from evidence.evidence_store import EvidenceStore
    load_seed()
    store = EvidenceStore()
    for layer in [
        "root_cause_suppression",
        "pathology_reversal",
        "circuit_stabilization",
        "regeneration_reinnervation",
        "adaptive_maintenance",
    ]:
        items = store.query_by_protocol_layer(layer)
        assert len(items) >= 5, f"Layer {layer} has only {len(items)} items"


def test_seed_is_idempotent(db_available):
    from evidence.seed_builder import load_seed
    stats1 = load_seed()
    stats2 = load_seed()
    assert stats1["interventions_loaded"] == stats2["interventions_loaded"]
