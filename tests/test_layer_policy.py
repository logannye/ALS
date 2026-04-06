"""Tests for layer-aware query selection in the research policy."""
from research.layer_orchestrator import ResearchLayer, get_layer_queries


def test_normal_biology_queries_dont_mention_als():
    """Layer 1 queries should be about normal biology, not ALS drugs."""
    queries = get_layer_queries(ResearchLayer.NORMAL_BIOLOGY)
    assert len(queries) >= 8
    for q in queries:
        assert "therapy" not in q.lower() or "normal" in q.lower(), \
            f"Layer 1 query should not be therapy-focused: {q}"


def test_als_mechanism_queries_mention_als():
    """Layer 2 queries should focus on ALS disease mechanisms."""
    queries = get_layer_queries(ResearchLayer.ALS_MECHANISMS)
    assert len(queries) >= 8
    for q in queries:
        assert "als" in q.lower(), f"Layer 2 query should mention ALS: {q}"


def test_erik_specific_queries_use_genetic_profile():
    """Layer 3 queries should reference Erik's specific gene/variant."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=profile)
    assert len(queries) >= 5
    assert any("SOD1" in q for q in queries), "Layer 3 queries should mention Erik's gene"
    assert any("G93A" in q for q in queries), "Layer 3 queries should mention Erik's variant"


def test_drug_design_queries_use_validated_targets():
    """Layer 4 queries should reference specific drug targets."""
    targets = ["SOD1", "EAAT2"]
    queries = get_layer_queries(ResearchLayer.DRUG_DESIGN, validated_targets=targets)
    assert len(queries) >= 4
    assert any("SOD1" in q for q in queries)
    assert any("EAAT2" in q for q in queries)
    assert any("binding" in q.lower() or "drug" in q.lower() for q in queries)


def test_erik_specific_without_profile_falls_back():
    """Layer 3 without a profile should fall back to Layer 2 queries."""
    queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=None)
    assert len(queries) >= 8  # Should get ALS_MECHANISMS fallback


def test_drug_design_without_targets_falls_back():
    """Layer 4 without targets should fall back to generic drug design queries."""
    queries = get_layer_queries(ResearchLayer.DRUG_DESIGN, validated_targets=None)
    assert len(queries) >= 2
