"""Tests for adversarial protocol verification (Task 4.1)."""
from __future__ import annotations

import pytest
from research.adversarial import (
    generate_adversarial_queries,
    classify_adversarial_result,
    select_challenge_target,
)


# ---------------------------------------------------------------------------
# TestGenerateAdversarialQueries
# ---------------------------------------------------------------------------

class TestGenerateAdversarialQueries:

    def test_produces_three_queries(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        assert len(queries) == 3

    def test_drug_name_in_failure_query(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        assert "Riluzole" in queries[0]

    def test_drug_name_in_harm_query(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        assert "Riluzole" in queries[1]

    def test_failure_query_contains_negative_terms(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        failure_q = queries[0]
        assert any(t in failure_q for t in ["failed", "negative", "ineffective", "discontinued"])

    def test_harm_query_contains_adverse_terms(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        harm_q = queries[1]
        assert any(t in harm_q for t in ["neurotoxicity", "adverse", "contraindicated"])

    def test_mechanism_dispute_uses_first_word(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        dispute_q = queries[2]
        # First word of mechanism is "glutamate"
        assert "glutamate" in dispute_q

    def test_mechanism_dispute_contains_dispute_terms(self):
        queries = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        dispute_q = queries[2]
        assert any(t in dispute_q for t in ["disputed", "disproven", "no effect", "insufficient"])

    def test_mechanism_dispute_falls_back_to_drug_name(self):
        """When mechanism is a single word equal to drug name or empty fallback."""
        queries = generate_adversarial_queries("Edaravone", "")
        dispute_q = queries[2]
        # mechanism_key should fall back to drug_name when mechanism string is empty/blank
        assert "Edaravone" in dispute_q

    def test_mechanism_key_single_word(self):
        """Single-word mechanism: key = that word."""
        queries = generate_adversarial_queries("Tofersen", "antisense")
        dispute_q = queries[2]
        assert "antisense" in dispute_q

    def test_all_queries_are_strings(self):
        queries = generate_adversarial_queries("Masitinib", "tyrosine kinase inhibition")
        assert all(isinstance(q, str) for q in queries)

    def test_returns_list(self):
        result = generate_adversarial_queries("Riluzole", "glutamate excitotoxicity inhibition")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestClassifyAdversarialResult
# ---------------------------------------------------------------------------

class TestClassifyAdversarialResult:

    def test_contradicts_failed_phase3(self):
        title = "Riluzole Failed Phase 3 ALS Trial: No Benefit Observed"
        abstract = (
            "This randomized controlled trial of Riluzole in amyotrophic lateral sclerosis "
            "failed to meet the primary endpoint. No improvement in ALSFRS-R was observed. "
            "The drug showed no benefit in slowing ALS progression."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "contradicts"

    def test_irrelevant_huntington_study(self):
        title = "Riluzole in Huntington Disease: A Randomized Trial"
        abstract = (
            "We studied Riluzole in 120 patients with Huntington disease. "
            "The treatment showed modest benefit on motor scores in HD patients."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "irrelevant"

    def test_irrelevant_drug_not_in_text(self):
        title = "Motor Neuron Disease Progression Study"
        abstract = (
            "This study examined biomarkers in amyotrophic lateral sclerosis patients. "
            "TDP-43 aggregation was correlated with faster progression."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "irrelevant"

    def test_weakens_neurotoxicity(self):
        title = "Neurotoxicity of Riluzole in ALS Motor Neurons"
        abstract = (
            "Riluzole administration at high doses caused neurotoxicity in amyotrophic "
            "lateral sclerosis patients. Severe adverse events were reported in 15% of cases."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "weakens"

    def test_context_dependent_als_no_clear_signal(self):
        title = "Riluzole Pharmacokinetics in ALS Patients"
        abstract = (
            "We examined the pharmacokinetic profile of Riluzole in amyotrophic lateral "
            "sclerosis patients. Drug absorption was variable across individuals."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "context_dependent"

    def test_contradicts_negative_result_als(self):
        title = "Negative Result: Edaravone ALS RCT"
        abstract = (
            "Edaravone did not meet the primary endpoint in this ALS trial. "
            "The treatment arm showed negative result for motor neuron disease progression."
        )
        result = classify_adversarial_result(title, abstract, "Edaravone")
        assert result == "contradicts"

    def test_contradicts_discontinued_als(self):
        title = "Trial Discontinued: Masitinib in ALS"
        abstract = (
            "The Masitinib trial was discontinued due to lack of efficacy in "
            "amyotrophic lateral sclerosis. The drug was ineffective."
        )
        result = classify_adversarial_result(title, abstract, "Masitinib")
        assert result == "contradicts"

    def test_weakens_contraindicated(self):
        title = "Contraindications for Tofersen in Motor Neuron Disease"
        abstract = (
            "Tofersen is contraindicated in patients with active infection. "
            "Motor neuron disease patients with comorbidities showed worsened outcomes."
        )
        result = classify_adversarial_result(title, abstract, "Tofersen")
        assert result == "weakens"

    def test_case_insensitive_drug_name(self):
        title = "riluzole als trial: failed endpoint"
        abstract = (
            "riluzole failed to reach the primary endpoint in amyotrophic lateral sclerosis. "
            "The trial did not meet the prespecified success criteria."
        )
        result = classify_adversarial_result(title, abstract, "Riluzole")
        assert result == "contradicts"

    def test_returns_string(self):
        result = classify_adversarial_result("foo", "bar baz", "Drug")
        assert isinstance(result, str)

    def test_valid_return_values(self):
        valid = {"contradicts", "weakens", "irrelevant", "context_dependent"}
        for title, abstract, drug, _ in [
            ("T1", "no drug mention here", "DrugX", "irrelevant"),
            ("DrugX failed ALS trial", "DrugX showed no benefit in ALS", "DrugX", "contradicts"),
            ("DrugX neurotoxicity", "DrugX caused neurotoxicity in ALS", "DrugX", "weakens"),
        ]:
            result = classify_adversarial_result(title, abstract, drug)
            assert result in valid


# ---------------------------------------------------------------------------
# TestSelectChallengeTarget
# ---------------------------------------------------------------------------

class TestSelectChallengeTarget:

    def test_selects_least_challenged(self):
        """Highest score * (1 / (1 + count)) → pick unchallenged highest scorer."""
        scores = {"Riluzole": 0.9, "Edaravone": 0.8, "Tofersen": 0.7}
        counts = {"Riluzole": 2, "Edaravone": 0, "Tofersen": 0}
        # Edaravone: 0.8 * (1/1) = 0.8  Tofersen: 0.7 * 1 = 0.7  Riluzole: 0.9/3 = 0.3
        result = select_challenge_target(scores, counts)
        assert result == "Edaravone"

    def test_returns_none_when_all_fully_challenged(self):
        scores = {"Riluzole": 0.9, "Edaravone": 0.8}
        counts = {"Riluzole": 3, "Edaravone": 3}
        result = select_challenge_target(scores, counts, max_challenges=3)
        assert result is None

    def test_empty_scores_returns_none(self):
        result = select_challenge_target({}, {})
        assert result is None

    def test_single_intervention_not_yet_challenged(self):
        result = select_challenge_target({"Riluzole": 0.9}, {})
        assert result == "Riluzole"

    def test_single_intervention_fully_challenged(self):
        result = select_challenge_target({"Riluzole": 0.9}, {"Riluzole": 3}, max_challenges=3)
        assert result is None

    def test_picks_highest_priority_when_equal_counts(self):
        scores = {"Riluzole": 0.9, "Edaravone": 0.5}
        counts = {"Riluzole": 0, "Edaravone": 0}
        result = select_challenge_target(scores, counts)
        assert result == "Riluzole"

    def test_max_challenges_respected(self):
        scores = {"A": 1.0, "B": 0.5}
        counts = {"A": 5, "B": 1}
        result = select_challenge_target(scores, counts, max_challenges=3)
        # A is over max, B is under
        assert result == "B"

    def test_missing_count_treated_as_zero(self):
        """Interventions without a count entry default to 0 challenges."""
        scores = {"A": 0.5, "B": 0.8}
        counts = {"A": 2}  # B not in counts → defaults to 0
        result = select_challenge_target(scores, counts)
        assert result == "B"

    def test_returns_optional_str(self):
        result = select_challenge_target({"X": 1.0}, {"X": 0})
        assert isinstance(result, str) or result is None
