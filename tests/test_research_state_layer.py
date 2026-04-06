"""Tests for ResearchState layer field."""
from research.state import ResearchState, initial_state


def test_initial_state_has_normal_biology_layer():
    state = initial_state(subject_ref="test")
    assert state.research_layer == "normal_biology"


def test_state_roundtrip_preserves_layer():
    state = initial_state(subject_ref="test")
    state_dict = state.to_dict()
    assert state_dict["research_layer"] == "normal_biology"
    restored = ResearchState.from_dict(state_dict)
    assert restored.research_layer == "normal_biology"


def test_state_roundtrip_with_erik_specific():
    state = initial_state(subject_ref="test")
    from dataclasses import replace
    state = replace(state, research_layer="erik_specific")
    state_dict = state.to_dict()
    restored = ResearchState.from_dict(state_dict)
    assert restored.research_layer == "erik_specific"


def test_genetic_profile_roundtrip():
    state = initial_state(subject_ref="test")
    from dataclasses import replace
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    state = replace(state, genetic_profile=profile)
    state_dict = state.to_dict()
    restored = ResearchState.from_dict(state_dict)
    assert restored.genetic_profile == profile
    assert restored.genetic_profile["gene"] == "SOD1"


def test_state_without_layer_field_defaults():
    """Old state dicts (pre-layer) should default to normal_biology."""
    old_dict = {"subject_ref": "test", "step_count": 100}
    state = ResearchState.from_dict(old_dict)
    assert state.research_layer == "normal_biology"
