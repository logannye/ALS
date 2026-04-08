"""Tests for provisional genetic profile inference and Layer 3 soft-gating."""
import pytest
from research.provisional_genetics import infer_provisional_profile
from research.layer_orchestrator import ResearchLayer, determine_layer, get_layer_queries


# ---------------------------------------------------------------------------
# infer_provisional_profile()
# ---------------------------------------------------------------------------

class TestInferProvisionalProfile:
    """Unit tests for the clinical inference function."""

    def test_default_returns_sals_tardbp(self):
        """Default (no family history, normal onset age) → TDP-43 proteinopathy."""
        profile = infer_provisional_profile()
        assert profile["gene"] == "TARDBP"
        assert profile["variant"] == "tdp43_proteinopathy"
        assert profile["subtype"] == "sALS"

    def test_default_confidence_with_nfl(self):
        """Default + NfL elevated → confidence 0.70 + 0.05 = 0.75."""
        profile = infer_provisional_profile(nfl_elevated=True)
        assert profile["confidence"] == pytest.approx(0.75)

    def test_default_confidence_without_nfl(self):
        """Default without NfL elevation → base confidence 0.70."""
        profile = infer_provisional_profile(nfl_elevated=False)
        assert profile["confidence"] == pytest.approx(0.70)

    def test_family_history_returns_fals_c9orf72(self):
        """Family history of ALS → C9orf72 fALS inference."""
        profile = infer_provisional_profile(family_history=True)
        assert profile["gene"] == "C9orf72"
        assert profile["variant"] == "hexanucleotide_repeat_expansion"
        assert profile["subtype"] == "fALS"

    def test_family_history_confidence(self):
        """Family history base confidence 0.40, +0.05 with NfL elevation."""
        profile_no_nfl = infer_provisional_profile(family_history=True, nfl_elevated=False)
        assert profile_no_nfl["confidence"] == pytest.approx(0.40)

        profile_nfl = infer_provisional_profile(family_history=True, nfl_elevated=True)
        assert profile_nfl["confidence"] == pytest.approx(0.45)

    def test_young_onset_flags_fus(self):
        """Age of onset < 45 → FUS sALS inference."""
        profile = infer_provisional_profile(age_onset=35)
        assert profile["gene"] == "FUS"
        assert profile["subtype"] == "sALS"

    def test_young_onset_boundary_at_44(self):
        """Age 44 is still young-onset (< 45)."""
        profile = infer_provisional_profile(age_onset=44, family_history=False)
        assert profile["gene"] == "FUS"

    def test_age_45_is_not_young_onset(self):
        """Age 45 is NOT young-onset → falls to default TARDBP path."""
        profile = infer_provisional_profile(age_onset=45, family_history=False)
        assert profile["gene"] == "TARDBP"

    def test_family_history_overrides_young_onset(self):
        """Family history takes precedence over young onset age."""
        profile = infer_provisional_profile(age_onset=30, family_history=True)
        assert profile["gene"] == "C9orf72"
        assert profile["subtype"] == "fALS"

    def test_provisional_flag_always_true(self):
        """provisional=True regardless of input combination."""
        for kwargs in [
            {},
            {"family_history": True},
            {"age_onset": 30},
            {"nfl_elevated": False},
        ]:
            profile = infer_provisional_profile(**kwargs)
            assert profile["provisional"] is True, f"provisional should be True for {kwargs}"

    def test_source_always_clinical_inference(self):
        """source='clinical_inference' for all cases."""
        assert infer_provisional_profile()["source"] == "clinical_inference"
        assert infer_provisional_profile(family_history=True)["source"] == "clinical_inference"

    def test_clinical_features_recorded(self):
        """Input parameters are echoed in clinical_features dict."""
        profile = infer_provisional_profile(
            age_onset=67,
            site_onset="limb",
            alsfrs_r=43,
            nfl_elevated=True,
            family_history=False,
        )
        cf = profile["clinical_features"]
        assert cf["age_onset"] == 67
        assert cf["site_onset"] == "limb"
        assert cf["alsfrs_r"] == 43
        assert cf["nfl_elevated"] is True
        assert cf["family_history"] is False

    def test_confidence_capped_at_1(self):
        """Confidence never exceeds 1.0 even with NfL boost."""
        # Manually craft a scenario where base would be near 1.0 — default path
        # starts at 0.70 + 0.05 = 0.75, well within cap; verify cap logic is safe.
        profile = infer_provisional_profile(nfl_elevated=True)
        assert profile["confidence"] <= 1.0

    def test_required_keys_present(self):
        """All required keys are present in the returned dict."""
        profile = infer_provisional_profile()
        for key in ("gene", "variant", "subtype", "confidence", "provisional", "source", "clinical_features"):
            assert key in profile, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# determine_layer() — provisional soft-gate
# ---------------------------------------------------------------------------

class TestDetermineLayerProvisional:
    """Tests for provisional genetics integration in determine_layer()."""

    def test_layer3_accessible_with_provisional_when_sufficient_evidence(self):
        """Layer 3 is reachable without confirmed genetics when provisional enabled + evidence >= 500."""
        layer = determine_layer(
            evidence_count=500,
            genetic_profile=None,
            validated_targets=0,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.ERIK_SPECIFIC

    def test_layer3_accessible_with_high_evidence(self):
        """Much more evidence than threshold also unlocks Layer 3 provisionally."""
        layer = determine_layer(
            evidence_count=5000,
            genetic_profile=None,
            validated_targets=0,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.ERIK_SPECIFIC

    def test_layer3_blocked_when_provisional_disabled(self):
        """When provisional_genetics_enabled=False, Layer 3 requires confirmed genetics."""
        layer = determine_layer(
            evidence_count=5000,
            genetic_profile=None,
            validated_targets=0,
            provisional_genetics_enabled=False,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.ALS_MECHANISMS

    def test_layer3_blocked_when_evidence_below_threshold(self):
        """Provisional is enabled but evidence count is too low → stays at Layer 2."""
        layer = determine_layer(
            evidence_count=499,
            genetic_profile=None,
            validated_targets=0,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.ALS_MECHANISMS

    def test_drug_design_requires_confirmed_genetics_not_provisional(self):
        """Drug design (Layer 4) is never unlocked by provisional profile alone."""
        layer = determine_layer(
            evidence_count=10000,
            genetic_profile=None,
            validated_targets=10,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        # Must be ERIK_SPECIFIC, NOT DRUG_DESIGN
        assert layer == ResearchLayer.ERIK_SPECIFIC
        assert layer != ResearchLayer.DRUG_DESIGN

    def test_drug_design_still_works_with_confirmed_genetics(self):
        """Confirmed genetics + validated targets still reaches Layer 4."""
        profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
        layer = determine_layer(
            evidence_count=1000,
            genetic_profile=profile,
            validated_targets=3,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.DRUG_DESIGN

    def test_default_provisional_disabled(self):
        """Without explicit kwargs, provisional path is off by default (backward compat)."""
        layer = determine_layer(
            evidence_count=5000,
            genetic_profile=None,
            validated_targets=0,
        )
        assert layer == ResearchLayer.ALS_MECHANISMS

    def test_layer1_unaffected_by_provisional_flag(self):
        """Layer 1 threshold is unaffected by provisional genetics setting."""
        layer = determine_layer(
            evidence_count=50,
            genetic_profile=None,
            validated_targets=0,
            provisional_genetics_enabled=True,
            provisional_genetics_min_evidence=500,
        )
        assert layer == ResearchLayer.NORMAL_BIOLOGY


# ---------------------------------------------------------------------------
# get_layer_queries() — provisional profile in query generation
# ---------------------------------------------------------------------------

class TestGetLayerQueriesProvisional:
    """Tests for Layer 3 query generation with no confirmed genetics."""

    def test_layer3_queries_generated_from_provisional_when_no_profile(self):
        """When genetic_profile is None, provisional inference drives query generation."""
        queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=None)
        assert len(queries) > 0
        # Default provisional path → TARDBP gene
        combined = " ".join(queries)
        assert "TARDBP" in combined

    def test_layer3_queries_with_confirmed_profile(self):
        """Confirmed genetics still generates correct queries."""
        profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
        queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=profile)
        combined = " ".join(queries)
        assert "SOD1" in combined
        assert "G93A" in combined
