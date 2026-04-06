"""Tests for the layer orchestrator — determines which research phase the system is in."""
import pytest
from research.layer_orchestrator import ResearchLayer, determine_layer


def test_layer_enum_has_four_values():
    assert len(ResearchLayer) == 4
    assert ResearchLayer.NORMAL_BIOLOGY.value == "normal_biology"
    assert ResearchLayer.ALS_MECHANISMS.value == "als_mechanisms"
    assert ResearchLayer.ERIK_SPECIFIC.value == "erik_specific"
    assert ResearchLayer.DRUG_DESIGN.value == "drug_design"


def test_fresh_start_is_normal_biology():
    """A brand-new system with zero evidence starts in Layer 1."""
    layer = determine_layer(evidence_count=0, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.NORMAL_BIOLOGY


def test_early_evidence_stays_normal_biology():
    """With few evidence items, still building the normal biology model."""
    layer = determine_layer(evidence_count=30, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.NORMAL_BIOLOGY


def test_sufficient_evidence_advances_to_als_mechanisms():
    """Once enough basic biology evidence exists, advance to Layer 2."""
    layer = determine_layer(evidence_count=200, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.ALS_MECHANISMS


def test_large_evidence_without_genetics_stays_als_mechanisms():
    """Even with lots of evidence, can't advance to Layer 3 without genetics."""
    layer = determine_layer(evidence_count=5000, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.ALS_MECHANISMS


def test_genetics_received_advances_to_erik_specific():
    """Once genetic profile is uploaded, advance to Layer 3."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    layer = determine_layer(evidence_count=500, genetic_profile=profile, validated_targets=0)
    assert layer == ResearchLayer.ERIK_SPECIFIC


def test_validated_targets_advances_to_drug_design():
    """Once causal targets are validated, advance to Layer 4."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    layer = determine_layer(evidence_count=1000, genetic_profile=profile, validated_targets=3)
    assert layer == ResearchLayer.DRUG_DESIGN


def test_genetics_with_low_evidence_still_advances():
    """Genetics trump evidence count for Layer 2→3 transition."""
    profile = {"gene": "C9orf72", "variant": "repeat_expansion", "subtype": "C9orf72"}
    layer = determine_layer(evidence_count=100, genetic_profile=profile, validated_targets=0)
    assert layer == ResearchLayer.ERIK_SPECIFIC


def test_validated_targets_without_genetics_stays_als_mechanisms():
    """Can't jump to drug design without genetics (even if targets exist from prior work)."""
    layer = determine_layer(evidence_count=2000, genetic_profile=None, validated_targets=5)
    assert layer == ResearchLayer.ALS_MECHANISMS
